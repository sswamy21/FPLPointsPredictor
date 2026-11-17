"""Point-in-time fixture forecasting and chronological backtesting."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .data import player_identity
from .player_workload import FEATURES as WORKLOAD_FEATURES, workload_features


MODEL_VERSION = 3
RESULT_LAG = pd.Timedelta(hours=3)
DEADLINE_OFFSET = pd.Timedelta(minutes=90)
POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_NUMBER = {position: code for code, position in POSITION.items()}
WINDOWS = (3, 6, 20)
PLAYER_STATS = (
    "points", "minutes", "appearance", "start", "sixty", "goals_scored", "assists",
    "bonus", "bps", "expected_goals", "expected_assists", "expected_goals_conceded",
)
TEAM_XG = ("xg_for", "xg_against")
TEAM_STATS = ("goals_for", "goals_against", *TEAM_XG)
MARKET_QUOTES = ("home_expected_goals", "away_expected_goals", "home_clean_sheet_probability", "away_clean_sheet_probability")
FEATURE_GROUPS = {
    "team_xg": tuple(
        f"{side}_{stat}_{window}"
        for side in ("team", "opponent") for stat in TEAM_XG
        for window in ("6", "20", "last_season")
    ) + tuple(f"opponent_{stat}_venue" for stat in TEAM_XG),
    "fpl_availability": ("flag_probability", "flag_age_days"),
    "set_pieces": ("penalty_order", "direct_fk_order", "corner_fk_order"),
    "predicted_minutes": ("expected_minutes", "predicted_start_probability"),
    "other_competitions": ("other_matches_7", "other_matches_14", "days_since_other_match"),
    "player_cup_workload": tuple(WORKLOAD_FEATURES.values()),
    "market": ("market_team_xg", "market_opponent_xg", "market_team_clean_sheet", "market_opponent_clean_sheet"),
}


def _date(value):
    return pd.to_datetime(value, utc=True, format="mixed", errors="coerce")


def _event_order(season, gameweek):
    return season * 100 + gameweek


def _position_code(position):
    return position.map(POSITION_NUMBER)


def _historical_appearances(history):
    rows = history.copy()
    rows["kickoff"] = _date(rows["kickoff"])
    rows["deadline"] = _date(rows.get("deadline", pd.NaT))
    rows = rows.sort_values(["kickoff", "fixture"], kind="stable")
    rows["_observed"] = rows["kickoff"] + RESULT_LAG
    rows["_event"] = _event_order(rows["season"], rows["gameweek"])
    for name in ("points", "minutes", "starts", *PLAYER_STATS[5:]):
        rows[name] = pd.to_numeric(rows.get(name, np.nan), errors="coerce")
    rows = rows.dropna(subset=["player", "points", "minutes", "kickoff"])
    rows["appearance"] = (rows["minutes"] > 0).astype("int8")
    rows["sixty"] = (rows["minutes"] >= 60).astype("int8")
    # Starts were not reported in earlier seasons. At least 45 minutes is a proxy then.
    starts = rows["starts"].fillna(rows["minutes"].ge(45).astype("int8"))
    rows["start"] = (starts.gt(0) & rows["appearance"].astype(bool)).astype("int8")
    return rows


def _fixture_teams(fixtures, players):
    matches = fixtures.copy()
    matches["kickoff"] = _date(matches["kickoff"])
    matches["deadline"] = _date(matches.get("deadline", pd.NaT))
    matches["_observed"] = matches["kickoff"] + RESULT_LAG
    matches["_event"] = _event_order(matches["season"], matches["gameweek"])
    for side in ("home", "away"):
        column = f"{side}_team_code"
        if column not in matches:
            matches[column] = pd.NA
    result = matches.dropna(subset=["home_goals", "away_goals", "kickoff"])
    if "finished" in result:
        result = result.loc[result["finished"].fillna(False)]
    scope = ["season", "fixture", "kickoff", "_observed", "_event"]
    home = result[scope + ["home_team", "home_team_code", "home_goals", "away_goals"]]
    away = result[scope + ["away_team", "away_team_code", "away_goals", "home_goals"]]
    context = scope + ["team", "team_code", "goals_for", "goals_against"]
    home = home.set_axis(context, axis=1)
    away = away.set_axis(context, axis=1)
    team_results = pd.concat([home.assign(home=True), away.assign(home=False)], ignore_index=True)
    xg = players.loc[players["minutes"] > 0].groupby(["season", "fixture", "home"])["expected_goals"]
    totals = xg.sum(min_count=1).where(xg.count() >= xg.size() * 0.8).rename("xg_for")
    team_results = team_results.join(totals, on=["season", "fixture", "home"])
    team_results["opposition_home"] = ~team_results["home"]
    team_results = team_results.join(
        totals.rename("xg_against"), on=["season", "fixture", "opposition_home"]
    ).drop(columns="opposition_home")
    return matches, team_results.sort_values("kickoff", kind="stable")


def _available(rows, event, cutoff):
    return rows.loc[(rows["_event"] < event) & (rows["_observed"] <= cutoff)]


def _by_round(frame):
    return dict(tuple(frame.groupby(["season", "gameweek"], sort=False))) if frame is not None else {}


def _join_context(targets, frame, context, cutoff, key, fields, timestamp="context_asof"):
    if context is None:
        return frame.assign(**{field: np.nan for field in fields})
    eligible = context.loc[_date(context[timestamp]) < cutoff].sort_values(timestamp)
    latest = eligible.drop_duplicates(key, keep="last").set_index(key)
    values = latest.reindex(columns=fields)
    return frame.assign(**{field: targets[key].map(values[field]) for field in fields})


class FeatureHistory:
    """Build features from completed prior-Gameweek observations at one cutoff."""

    def __init__(self, history, fixtures, *, player_context=None, market_context=None,
                 minutes_context=None, other_fixtures=None, player_workload=None):
        self.players = _historical_appearances(history)
        self.fixtures, self.teams = _fixture_teams(fixtures, self.players)
        self.player_context = _by_round(player_context)
        self.minutes_context = _by_round(minutes_context)
        self.market_context = dict(tuple(market_context.groupby("season", sort=False))) if market_context is not None else {}
        self.other_fixtures = other_fixtures
        self.player_workload = player_workload
        self.club_codes = pd.concat([
            self.fixtures[["season", f"{side}_team", f"{side}_team_code"]].set_axis(["season", "team", "team_code"], axis=1)
            for side in ("home", "away")
        ]).dropna(subset=["team_code"]).drop_duplicates(["season", "team"])
        fixture_state = self.fixtures.set_index(["season", "fixture"])
        if "finished" in fixture_state:
            # Element summaries may briefly contain an in-progress result.
            key = pd.MultiIndex.from_frame(self.players[["season", "fixture"]])
            finished = fixture_state["finished"].reindex(key)
            explicitly_incomplete = finished.eq(False).fillna(False).to_numpy(dtype=bool)
            self.players = self.players.loc[~explicitly_incomplete]

    def deadline(self, season, gameweek, targets):
        event = _event_order(season, gameweek)
        event_fixtures = self.fixtures.loc[self.fixtures["_event"] == event]
        explicit = event_fixtures["deadline"].min()
        if pd.isna(explicit):
            explicit = targets["deadline"].min()
        if pd.notna(explicit):
            return explicit
        first_kickoff = pd.concat([event_fixtures["kickoff"], targets["kickoff"]]).min()
        return first_kickoff - DEADLINE_OFFSET

    def snapshot(self, targets, season, gameweek, cutoff):
        cutoff = _date(cutoff)
        if pd.isna(cutoff):
            raise ValueError("A Gameweek feature snapshot requires a valid UTC deadline.")
        event = _event_order(season, gameweek)
        frame = targets[["player", "position", "team", "opponent", "home"]].copy()
        frame["position"] = _position_code(frame["position"])
        frame["home"] = frame["home"].astype("int8")

        prior = _available(self.players, event, cutoff)
        prior = prior.loc[prior["player"].isin(frame["player"])]
        recent = prior.groupby("player", sort=False).tail(max(WINDOWS)).copy()
        recent["_reverse"] = recent.groupby("player", sort=False).cumcount(ascending=False)
        for window in WINDOWS:
            window_stats = recent.loc[recent["_reverse"] < window].groupby("player", sort=False)[list(PLAYER_STATS)].mean()
            frame = frame.join(window_stats.add_suffix(f"_{window}"), on="player")
        frame["prior_fixtures"] = frame["player"].map(recent.groupby("player", sort=False).size()).fillna(0)
        frame["season_fixtures"] = frame["player"].map(prior.loc[prior["season"] == season].groupby("player", sort=False).size()).fillna(0)
        last_played = recent.loc[recent["appearance"] > 0].groupby("player", sort=False)["kickoff"].max()
        frame["days_since_appearance"] = (cutoff - _date(frame["player"].map(last_played))).dt.total_seconds() / 86400

        team_results = _available(self.teams, event, cutoff)
        season_results = team_results.loc[team_results["season"] == season]
        previous_season = team_results.loc[team_results["season"] == season - 1]
        club_codes = self.club_codes.loc[self.club_codes["season"] == season].set_index("team")["team_code"]
        frame = self._team_context(frame, season_results, previous_season, club_codes, cutoff)
        frame = _join_context(
            targets, frame, self.player_context.get((season, gameweek)), cutoff, "player",
            FEATURE_GROUPS["fpl_availability"] + FEATURE_GROUPS["set_pieces"],
        )
        frame = _join_context(
            targets, frame, self.minutes_context.get((season, gameweek)), cutoff, "player",
            FEATURE_GROUPS["predicted_minutes"],
        )
        frame = self._market_features(targets, frame, cutoff, season)
        frame = self._other_competitions(frame, club_codes, cutoff)
        frame = frame.join(workload_features(self.player_workload, targets["player"], cutoff))
        # Points-per-fixture form, with fixed positional priors for debuts.
        frame["baseline"] = frame["points_6"].fillna(frame["position"].map({1: 2.7, 2: 1.9, 3: 2.0, 4: 2.1})).fillna(2.0)
        return frame.drop(columns=["player", "team", "opponent"])

    def _market_features(self, targets, frame, cutoff, season):
        quotes = _join_context(targets, targets.iloc[:, :0], self.market_context.get(season),
                               cutoff, "fixture", MARKET_QUOTES, "market_asof")
        home = frame["home"].astype(bool)
        home_xg, away_xg, home_cs, away_cs = (quotes[column] for column in MARKET_QUOTES)
        return frame.assign(
            market_team_xg=home_xg.where(home, away_xg), market_opponent_xg=away_xg.where(home, home_xg),
            market_team_clean_sheet=home_cs.where(home, away_cs), market_opponent_clean_sheet=away_cs.where(home, home_cs),
        )

    def _other_competitions(self, frame, club_codes, cutoff):
        empty = {field: np.nan for field in FEATURE_GROUPS["other_competitions"]}
        if self.other_fixtures is None or self.other_fixtures.empty:
            return frame.assign(**empty)
        attrs = self.other_fixtures.attrs
        start = cutoff - pd.Timedelta(days=30)
        if not {start.year, cutoff.year}.issubset(attrs["covered_years"]):
            return frame.assign(**empty)
        club = frame["team"].map(club_codes)
        covered = club.isin(attrs["club_codes"])
        known = self.other_fixtures.loc[self.other_fixtures["kickoff"] + RESULT_LAG <= cutoff]
        features = {}
        for days in (7, 14):
            recent = known.loc[known["kickoff"] >= cutoff - pd.Timedelta(days=days)]
            features[f"other_matches_{days}"] = club.map(recent.groupby("team_code").size()).fillna(0).where(covered)
        last_played = club.map(known.groupby("team_code")["kickoff"].max())
        features["days_since_other_match"] = (cutoff - last_played).dt.total_seconds().div(86400).clip(upper=30)
        return frame.assign(**features)

    @staticmethod
    def _team_context(frame, rows, previous_season, club_codes, cutoff):
        group = rows.groupby("team", sort=False)
        for window in (6, 20):
            recent = group.tail(window)
            stats = recent.groupby("team", sort=False)[list(TEAM_STATS)].mean()
            frame = frame.join(stats.add_prefix("team_").add_suffix(f"_{window}"), on="team")
            frame = frame.join(stats.add_prefix("opponent_").add_suffix(f"_{window}"), on="opponent")
        venue_groups = rows.groupby(["team", "home"], sort=False)
        venue_stats = venue_groups.tail(6).groupby(["team", "home"], sort=False)[list(TEAM_STATS)].mean()
        frame["opponent_home"] = 1 - frame["home"]
        venue_stats.index = venue_stats.index.set_names(["opponent", "opponent_home"])
        frame = frame.join(venue_stats.add_prefix("opponent_").add_suffix("_venue"), on=["opponent", "opponent_home"])
        previous_form = previous_season.groupby("team_code", sort=False).tail(20)
        previous_form = previous_form.groupby("team_code", sort=False)[list(TEAM_STATS)].mean()
        for side in ("team", "opponent"):
            clubs = frame[side].map(club_codes)
            for stat in TEAM_STATS:
                frame[f"{side}_{stat}_last_season"] = clubs.map(previous_form[stat])
        frame["days_since_team_fixture"] = (cutoff - _date(frame["team"].map(group["kickoff"].max()))).dt.total_seconds() / 86400
        # This pre-match difficulty proxy uses prior goals and venue. Archive FDR can be revised after the fact.
        attack = frame["opponent_goals_for_venue"].fillna(frame["opponent_goals_for_20"])
        attack = attack.fillna(frame["opponent_goals_for_last_season"]).fillna(1.4)
        defence = frame["opponent_goals_against_venue"].fillna(frame["opponent_goals_against_20"])
        defence = defence.fillna(frame["opponent_goals_against_last_season"]).fillna(1.4)
        frame["opponent_difficulty"] = attack - defence + (frame["opponent_home"] - 0.5) * 0.45
        return frame.drop(columns="opponent_home")


def _observations(builder, as_of=None):
    observations = builder.players
    if as_of is not None:
        observations = observations.loc[observations["_observed"] <= _date(as_of)]
    chunks = []
    for (season, gameweek), targets in observations.groupby(["season", "gameweek"], sort=True):
        cutoff = builder.deadline(season, gameweek, targets)
        targets = targets.loc[targets["kickoff"] >= cutoff]
        roster = _join_context(
            targets, targets.iloc[:, :0], builder.player_context.get((season, gameweek)),
            cutoff, "player", ("context_team", "context_position"),
        )
        # A transfer after the deadline cannot change the club on a historical prediction row.
        valid = roster["context_team"].isna() | roster["context_team"].eq(targets["team"])
        positions = roster["context_position"].combine_first(targets["position"])
        targets = targets.loc[valid].assign(position=positions.loc[valid])
        if targets.empty:
            continue
        snapshot = builder.snapshot(targets, season, gameweek, cutoff)
        snapshot["_cutoff"] = cutoff
        snapshot["_outcome"] = targets[["appearance", "start", "sixty"]].dot(np.array([1, 1, 2]))
        snapshot[["_observed", "_event", "_season", "_gameweek", "_player", "_points"]] = targets[
            ["_observed", "_event", "season", "gameweek", "player", "points"]
        ]
        chunks.append(snapshot)
    if not chunks:
        raise ValueError("No completed player fixtures are available for training.")
    return pd.concat(chunks, ignore_index=True)


@dataclass
class FPLModel:
    availability: object
    points_if_appearing: object
    feature_columns: tuple[str, ...]
    metrics: dict
    training_rows: int
    training_seasons: list[int]
    version: int = MODEL_VERSION


def _fit(observations, features):
    appearances = observations["_outcome"] > 0
    if not appearances.any():
        raise ValueError("Training requires at least one recorded player appearance.")
    features = tuple(column for column in features if observations[column].nunique() > 1)
    inputs = observations[list(features)]
    classes = observations["_outcome"]
    min_leaf = max(5, min(40, len(observations) // 100))
    availability = HistGradientBoostingClassifier(
        max_iter=65, max_leaf_nodes=19, min_samples_leaf=min_leaf,
        learning_rate=0.11, l2_regularization=1.5, early_stopping=False,
    ) if classes.nunique() > 1 else DummyClassifier(strategy="prior")
    points = HistGradientBoostingRegressor(
        max_iter=110, max_leaf_nodes=19, min_samples_leaf=min_leaf,
        learning_rate=0.065, l2_regularization=2.0, early_stopping=False,
    )
    with threadpool_limits(limits=4):
        availability.fit(inputs, classes)
        points.fit(inputs.loc[appearances], observations.loc[appearances, "_points"])
    return FPLModel(
        availability, points, features, {}, len(observations),
        sorted(observations["_season"].astype(int).unique().tolist()),
    )


def _predict_fixtures(model, inputs):
    inputs = inputs[list(model.feature_columns)]
    with threadpool_limits(limits=4):
        probabilities = model.availability.predict_proba(inputs)
        raw_points = model.points_if_appearing.predict(inputs)
    classes = model.availability.classes_
    appeared = probabilities[:, classes > 0].sum(axis=1)
    started = probabilities[:, np.isin(classes, [2, 4])].sum(axis=1)
    sixty = probabilities[:, classes >= 3].sum(axis=1)
    return pd.DataFrame({
        "points": raw_points * appeared, "appearance": appeared, "start": started, "sixty": sixty,
    }, index=inputs.index)


def _scores(weeks):
    error = weeks["prediction"] - weeks["_points"]
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(error.pow(2).mean())),
        "baseline_mae": float((weeks["baseline"] - weeks["_points"]).abs().mean()),
        "rows": len(weeks),
    }


def _evaluate(observations, features):
    seasons = sorted(observations["_season"].unique())
    completed = [season for season in seasons if observations.loc[observations["_season"] == season, "_gameweek"].max() >= 38]
    candidates = (completed or seasons)[-3:]
    folds, all_weeks = [], []
    for season in candidates:
        test = observations.loc[observations["_season"] == season].copy()
        first_cutoff = test["_cutoff"].min()
        train = observations.loc[(observations["_season"] < season) & (observations["_observed"] <= first_cutoff)]
        if train.empty:
            continue
        model = _fit(train, features)
        test["prediction"] = _predict_fixtures(model, test)["points"]
        totals = test.groupby(["_season", "_gameweek", "_player"])[["_points", "prediction", "baseline"]].sum()
        all_weeks.append(totals)
        folds.append({
            "season": int(season), "first_gameweek": int(test["_gameweek"].min()),
            "last_gameweek": int(test["_gameweek"].max()), **_scores(totals),
        })
    return {**_scores(pd.concat(all_weeks)), "folds": folds} if all_weeks else {"folds": [], "rows": 0}


def fit_model(history_df, fixtures_df, validate=True, as_of=None, exclude_groups=(), **context):
    """Fit on complete fixtures, optionally evaluating expanding-season holdouts."""
    observations = _observations(FeatureHistory(history_df, fixtures_df, **context), as_of)
    excluded = {feature for group in exclude_groups for feature in FEATURE_GROUPS[group]}
    features = tuple(column for column in observations if not column.startswith("_")
                     and column != "baseline" and column not in excluded and observations[column].notna().any())
    model = _fit(observations, features)
    if validate:
        model.metrics = _evaluate(observations, features)
    return model


def _status_multiplier(element, is_next):
    field = "chance_of_playing_next_round" if is_next else "chance_of_playing_this_round"
    chance = element.get(field)
    if chance is not None:
        return min(1.0, max(0.0, chance / 100))
    return {"i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0, "d": 0.75}.get(element.get("status"), 1.0)


def _team_label(team):
    return team.get("short_name") or team.get("name") or str(team["id"])


def _fixture_targets(matches, elements, season):
    club_matches = {}
    for match in matches.itertuples(index=False):
        for home, team, opponent, difficulty in (
            (True, match.home_team, match.away_team, getattr(match, "home_difficulty", None)),
            (False, match.away_team, match.home_team, getattr(match, "away_difficulty", None)),
        ):
            club_matches.setdefault(team, []).append({
                "team": team, "opponent": opponent, "home": home, "fixture": match.fixture,
                "difficulty": difficulty, "kickoff": match.kickoff,
            })
    return pd.DataFrame([
        {**match, "player": player_identity(element.get("code"), season, element_id),
         "element": element_id, "position": POSITION[element["element_type"]]}
        for element_id, element in elements.items() for match in club_matches.get(element["team"], ())
    ])


def _at_least_one(probabilities, key, multiplier):
    return multiplier * float(np.clip(1 - math.prod(1 - probability[key] for probability in probabilities), 0, 1))


def forecast(model, history_df, fixtures_df, bootstrap, season, gameweek, deadline, **context):
    """Predict each official player for every scheduled fixture at the same deadline."""
    cutoff = _date(deadline)
    if pd.isna(cutoff):
        raise ValueError("A forecast requires a valid upcoming Gameweek deadline.")
    builder = FeatureHistory(history_df, fixtures_df, **context)
    matches = builder.fixtures.loc[builder.fixtures["_event"] == _event_order(season, gameweek)]
    matches = matches.loc[matches["kickoff"].isna() | (matches["kickoff"] >= cutoff)].sort_values("kickoff", kind="stable")
    if "finished" in matches:
        matches = matches.loc[~matches["finished"].eq(True)]
    teams = {team["id"]: team for team in bootstrap["teams"]}
    elements = {element["id"]: element for element in bootstrap["elements"] if element["element_type"] in POSITION}
    event = next((event for event in bootstrap["events"] if event["id"] == gameweek), {})
    is_next = event.get("is_next", False) or not event.get("is_current", False)
    targets = _fixture_targets(matches, elements, season)
    by_element = {}
    if not targets.empty:
        features = builder.snapshot(targets, season, gameweek, cutoff)
        probabilities = _predict_fixtures(model, features)
        probabilities["flag_used"] = "flag_probability" in model.feature_columns and features["flag_probability"].notna()
        by_element = {
            element: group.to_dict("records") for element, group in targets.join(probabilities).groupby("element", sort=False)
        }
    predictions = {}
    for element_id, element in elements.items():
        player_fixtures = by_element.get(element_id, [])
        used_flag = any(match["flag_used"] for match in player_fixtures)
        multiplier = 1.0 if used_flag else _status_multiplier(element, is_next)
        club = teams[element["team"]]
        fixture_details = []
        for match in player_fixtures:
            opponent = teams[match["opponent"]]
            fixture_details.append({
                "fixture_id": int(match["fixture"]), "home": bool(match["home"]),
                "opponent": _team_label(opponent), "opponent_name": opponent.get("name", _team_label(opponent)),
                "points": float(match["points"]) * multiplier,
                "difficulty": int(match["difficulty"]) if pd.notna(match["difficulty"]) else None,
                "kickoff": match["kickoff"].isoformat() if pd.notna(match["kickoff"]) else None,
            })
        predictions[element_id] = {
            "id": element_id, "code": element.get("code"), "name": element["web_name"],
            "full_name": " ".join(part for part in (element.get("first_name"), element.get("second_name")) if part),
            "position": POSITION[element["element_type"]], "team": _team_label(club),
            "team_name": club.get("name", _team_label(club)), "team_id": element["team"],
            "points": math.fsum(detail["points"] for detail in fixture_details), "fixtures": fixture_details,
            "appearance_chance": _at_least_one(player_fixtures, "appearance", multiplier),
            "start_chance": _at_least_one(player_fixtures, "start", multiplier),
            "sixty_chance": _at_least_one(player_fixtures, "sixty", multiplier),
            "status": element.get("status"), "news": element.get("news", ""),
        }
    return predictions
