"""Compare forecast families and data sources on expanding-season holdouts."""

from collections import Counter
from math import fsum

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .lineup import SQUAD_COUNTS, optimize
from .model import FPLModel, _fit, _historical_appearances, _predict_fixtures, _scores

FAMILIES = ("hist_gradient_boosting", "regularized_linear", "catboost")
SEED = 61117
REPLAY_SQUADS_PER_ROUND = 20


def _linear_inputs(features):
    numeric = [name for name in features if name != "position"]
    transformations = []
    if "position" in features:
        transformations.append(("position", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["position"]))
    if numeric:
        transformations.append(("numeric", make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
        ), numeric))
    return ColumnTransformer(transformations)


def fit_candidate(observations, features, family):
    """Train one availability classifier and conditional-points regressor."""
    if family == FAMILIES[0]:
        return _fit(observations, features)

    inputs = observations[list(features)].astype(float)
    outcomes = observations["_outcome"].astype(int)
    appearances = outcomes > 0
    if not appearances.any():
        raise ValueError("Training requires at least one recorded player appearance.")
    if family == FAMILIES[1]:
        availability = make_pipeline(
            _linear_inputs(features), LogisticRegression(C=0.02, tol=0.001, max_iter=500),
        )
        points = make_pipeline(_linear_inputs(features), Ridge(alpha=100))
    elif family == FAMILIES[2]:
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as error:
            raise RuntimeError("Install the optional CatBoost benchmark dependency.") from error

        settings = {
            "depth": 5, "learning_rate": 0.1, "l2_leaf_reg": 3.0, "random_seed": SEED,
            "thread_count": 4, "verbose": False, "allow_writing_files": False,
        }
        availability = CatBoostClassifier(iterations=130, **settings)
        points = CatBoostRegressor(iterations=170, **settings)
    else:
        raise ValueError(f"Unknown model family {family!r}.")

    if outcomes.nunique() == 1:
        availability = DummyClassifier(strategy="prior")
    with threadpool_limits(limits=4):
        availability.fit(inputs, outcomes)
        points.fit(inputs.loc[appearances], observations.loc[appearances, "_points"])
    return FPLModel(
        availability, points, tuple(features), {}, len(observations),
        sorted(observations["_season"].astype(int).unique().tolist()),
    )


def _folds(observations, seasons):
    available = sorted(observations["_season"].unique())
    if seasons is None:
        complete = [
            season for season in available
            if observations.loc[observations["_season"] == season, "_gameweek"].max() >= 38
        ]
        seasons = (complete or available)[-3:]
    elif set(seasons) - set(available):
        raise ValueError("Some requested validation seasons have no observations.")

    folds = []
    for season in sorted(set(seasons)):
        test = observations.loc[observations["_season"] == season]
        first_cutoff = test["_cutoff"].min()
        train = observations.loc[
            (observations["_season"] < season) & (observations["_observed"] <= first_cutoff)
        ]
        if not train.empty:
            folds.append((int(season), train, test))
    if not folds:
        raise ValueError("No validation season has eligible earlier training results.")
    return folds


def _pick_squads(players, rng, sample_count):
    squads = []
    position_players = {
        position: (group.index.to_numpy(), group["team"].to_numpy(), group["weight"].to_numpy())
        for position, group in players.groupby("position")
    }
    for _ in range(sample_count):
        squad = {}
        clubs = Counter()
        for position, count in SQUAD_COUNTS.items():
            pool = position_players.get(position)
            if pool is None:
                break
            players, teams, weights = pool
            for _ in range(count):
                allowed = np.fromiter(
                    (player not in squad and clubs[club] < 3 for player, club in zip(players, teams)),
                    dtype=bool, count=len(players),
                )
                indices = np.flatnonzero(allowed)
                if not len(indices):
                    break
                eligible_weights = weights[indices]
                chosen = rng.choice(indices, p=eligible_weights / eligible_weights.sum())
                clubs[teams[chosen]] += 1
                squad[int(players[chosen])] = position
        if len(squad) == 15:
            squads.append(squad)
    return squads


def replay_squads(history, test, sample_count=REPLAY_SQUADS_PER_ROUND, player_context=None):
    """Build identical comparison squads from prior recorded appearances."""
    if sample_count < 1:
        raise ValueError("Replay requires at least one squad per week.")
    snapshots = {} if player_context is None else dict(tuple(player_context.groupby(["season", "gameweek"])))
    squads = {}
    for (season, week), targets in test.groupby(["_season", "_gameweek"]):
        # Target rows contribute only the calendar and deadline, never squad membership.
        cutoff = targets["_cutoff"].min()
        eligible = history.loc[
            (history["season"] == season) & (history["gameweek"] < week)
            & (history["_observed"] <= cutoff)
        ]
        if eligible.empty:
            continue
        latest = eligible.drop_duplicates("player", keep="last").set_index("player")
        recent = eligible.groupby("player", sort=False).tail(6)
        still_playing = recent.loc[(recent["gameweek"] >= week - 6) & (recent["minutes"] > 0), "player"]
        player_roster = latest[["team", "position"]]
        if (snapshot := snapshots.get((season, week))) is not None:
            eligible_snapshot = snapshot.loc[snapshot["context_asof"] < cutoff]
            if not eligible_snapshot.empty:
                player_roster = (
                    eligible_snapshot.sort_values("context_asof").drop_duplicates("player", keep="last")
                    .set_index("player")[["context_team", "context_position"]]
                    .set_axis(["team", "position"], axis=1).dropna()
                )
        roster = player_roster.loc[player_roster.index.isin(still_playing)].sort_index()
        recent_points = roster.index.map(recent.groupby("player")["points"].mean()).to_numpy(dtype=float)
        roster["weight"] = 1 + np.maximum(recent_points, 0)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, int(season), int(week)]))
        squads[int(season), int(week)] = _pick_squads(roster, rng, sample_count)
    return squads


def _lineup_scores(totals, squads, column):
    realised = []
    for key, group in totals.groupby(level=["_season", "_gameweek"]):
        week = group.droplevel(["_season", "_gameweek"])
        projected = week[column].to_dict()
        actual = week["_points"].to_dict()
        for squad in squads.get(key, ()):
            players = {
                player: {"id": player, "position": position, "points": projected.get(player, 0.0)}
                for player, position in squad.items()
            }
            lineup = optimize(list(squad), players)["starters"]
            captain = next(player["id"] for player in lineup if player["role"] == "C")
            captain_points = actual.get(captain, 0.0)
            starter_points = fsum(actual.get(player["id"], 0.0) for player in lineup)
            realised.append((starter_points + captain_points, captain_points))
    if not realised:
        return {"squads": 0, "mean_lineup_points": None, "mean_captain_points": None}
    totals, captains = np.asarray(realised).mean(axis=0)
    return {
        "squads": len(realised), "mean_lineup_points": float(totals),
        "mean_captain_points": float(captains),
    }


def _coverage(folds, columns):
    samples = []
    for season, train, test in folds:
        train_rows = train[list(columns)].notna().any(axis=1) if columns else np.zeros(len(train), dtype=bool)
        test_rows = test[list(columns)].notna().any(axis=1) if columns else np.zeros(len(test), dtype=bool)
        samples.append({
            "season": season, "training_rows": int(train_rows.sum()),
            "test_rows": int(test_rows.sum()), "total_test_rows": len(test),
            "varying_training_columns": int(train[list(columns)].nunique().gt(1).sum()),
        })
    return {
        "usable": any(sample["varying_training_columns"] and sample["test_rows"] for sample in samples),
        "columns": list(columns), "folds": samples,
    }


def _evaluate_candidate(folds, features, family, squads, progress):
    results, seasons = [], []
    for season, train, test in folds:
        progress(f"Fitting {family} on {len(train):,} rows for {season}/{str(season + 1)[-2:]}.")
        # Constant or entirely missing training columns carry no effect in this fold.
        varying = train[list(features)].nunique() > 1
        fold_features = tuple(varying.index[varying])
        if not fold_features:
            raise ValueError(f"No training feature varies for the {season} validation fold.")
        model = fit_candidate(train, fold_features, family)
        # Cast once so sklearn pipelines and CatBoost receive the same numeric representation.
        inputs = test[list(model.feature_columns)].astype(float)
        prediction = _predict_fixtures(model, inputs)["points"]
        if "flag_probability" in test and "flag_probability" not in model.feature_columns:
            # This is the same FPL flag multiplier used by the original live forecast.
            prediction *= test["flag_probability"].fillna(1.0)
        scored = test.assign(prediction=prediction)
        totals = scored.groupby(["_season", "_gameweek", "_player"])[["_points", "prediction", "baseline"]].sum()
        results.append(totals)
        seasons.append({
            "season": season, "first_gameweek": int(test["_gameweek"].min()),
            "last_gameweek": int(test["_gameweek"].max()), **_scores(totals),
            **_lineup_scores(totals, squads, "prediction"),
        })
    totals = pd.concat(results)
    return {
        "family": family, **_scores(totals), **_lineup_scores(totals, squads, "prediction"),
        "folds": seasons,
    }, totals


def run_benchmark(observations, *, history=None, player_context=None, feature_groups=None, families=FAMILIES,
                  seasons=None, progress=None):
    """Score model families and cumulative input additions on common held-out rows."""
    features = tuple(name for name in observations if not name.startswith("_") and name != "baseline")
    feature_groups = feature_groups or {}
    progress = progress or (lambda _message: None)
    folds = _folds(observations, seasons)
    squads = {}
    if history is not None:
        past = _historical_appearances(history)
        for _, _, test in folds:
            squads.update(replay_squads(past, test, player_context=player_context))

    additions = {column for columns in feature_groups.values() for column in columns}
    base = tuple(column for column in features if column not in additions)
    coverage, stages = {}, [("base", base)]
    for name, candidates in feature_groups.items():
        columns = tuple(column for column in candidates if column in features)
        coverage[name] = _coverage(folds, columns)
        if coverage[name]["usable"]:
            last_name, last_features = stages[-1]
            joined = tuple(dict.fromkeys(last_features + columns))
            stages.append((f"{last_name}+{name}", joined))

    last_stage, full = stages[-1]
    models, baseline_totals = [], None
    for family in families:
        result, totals = _evaluate_candidate(folds, full, family, squads, progress)
        models.append(dict(result, feature_set=last_stage))
        if baseline_totals is None:
            baseline_totals = totals
    if baseline_totals is None:
        raise ValueError("Choose at least one forecast family to benchmark.")
    ablations = []
    if len(stages) > 1:
        for name, columns in stages[:-1]:
            result, _ = _evaluate_candidate(folds, columns, FAMILIES[0], squads, progress)
            ablations.append(dict(result, feature_set=name))
        final = next((result for result in models if result["family"] == FAMILIES[0]), None)
        if final is None:
            final, _ = _evaluate_candidate(folds, full, FAMILIES[0], squads, progress)
        ablations.append(dict(final, feature_set=last_stage))
    return {
        "models": models, "ablations": ablations, "feature_coverage": coverage,
        "availability_policy": "Apply the archived flag as a multiplier when it is absent from model features.",
        "baseline": {
            "mae": float((baseline_totals["baseline"] - baseline_totals["_points"]).abs().mean()),
            "rmse": float(np.sqrt((baseline_totals["baseline"] - baseline_totals["_points"]).pow(2).mean())),
            **_lineup_scores(baseline_totals, squads, "baseline"),
        },
        "replay": {
            "source": "reconstructed from prior appearances and timestamped rosters when available",
            "squads_per_round": REPLAY_SQUADS_PER_ROUND, "seed": SEED,
            "rounds": sum(bool(round_squads) for round_squads in squads.values()),
            "score": "XI points plus one additional copy of the captain's points",
            "squad_limits": "Standard positional totals and a maximum of three players per club",
        } if squads else None,
    }
