from collections import Counter

import pandas as pd

from fpl.benchmark import FAMILIES, _folds, _lineup_scores, replay_squads, run_benchmark
from fpl.lineup import SQUAD_COUNTS


def test_model_family_folds_share_rows_and_skip_unobserved_data():
    first = pd.Timestamp("2023-08-01", tz="UTC")
    second = pd.Timestamp("2024-08-01", tz="UTC")
    rows = pd.DataFrame([
        {"_season": season, "_gameweek": 1, "_cutoff": date, "_observed": date + pd.Timedelta(days=1),
         "_player": player, "_points": float(player % 5), "_outcome": player % 5,
         "position": player % 4 + 1, "form": player % 5, "baseline": 2.0, "market": float("nan")}
        for season, date in [(2023, first), (2024, second)] for player in range(30)
    ])
    late = rows.iloc[[0]].assign(_observed=second + pd.Timedelta(days=1), _player=101)
    rows = pd.concat([rows, late], ignore_index=True)
    _, training, validation = _folds(rows, [2024])[0]
    assert set(training["_season"]) == {2023} and 101 not in training["_player"].to_list()
    assert set(validation["_season"]) == {2024}

    result = run_benchmark(
        rows, seasons=[2024], feature_groups={"market": ("market",)}, families=FAMILIES[:2],
    )
    assert result["feature_coverage"]["market"]["usable"] is False
    assert result["ablations"] == []
    assert [(model["family"], model["rows"], model["feature_set"]) for model in result["models"]] == [
        (family, 30, "base") for family in FAMILIES[:2]
    ]


def test_replay_pool_uses_prior_available_rows_and_fixed_club_limits():
    cutoff = pd.Timestamp("2024-09-01", tz="UTC")
    history = pd.DataFrame([
        {"season": 2024, "gameweek": 1, "_observed": cutoff - pd.Timedelta(days=1),
         "player": i * 10 + j, "team": j, "position": pos, "minutes": 90, "points": 2}
        for i, pos in enumerate(SQUAD_COUNTS) for j in range(1, 9)
    ])
    impossible = history.iloc[[0]].assign(player=99, _observed=cutoff + pd.Timedelta(seconds=1), points=1000)
    after_deadline_gameweek = impossible.assign(player=100, gameweek=2, _observed=cutoff - pd.Timedelta(days=1))
    history = pd.concat([history, impossible, after_deadline_gameweek], ignore_index=True)
    targets = pd.DataFrame({"_season": [2024], "_gameweek": [2], "_cutoff": [cutoff], "_player": [101]})
    squads = replay_squads(history, targets, sample_count=3)
    assert squads == replay_squads(history, targets, sample_count=3)
    assert len(squads[2024, 2]) == 3
    for squad in squads[2024, 2]:
        assert not set(squad) & {99, 100, 101}
        assert Counter(squad.values()) == SQUAD_COUNTS
        assert max(Counter(player % 10 for player in squad).values()) <= 3

    fixture_player = next(iter(squads[2024, 2][0]))
    totals = pd.DataFrame({"_season": [2024], "_gameweek": [2], "_player": [fixture_player],
                           "_points": [8.0], "prediction": [3.0]}).set_index(["_season", "_gameweek", "_player"])
    score = _lineup_scores(totals, {(2024, 2): squads[2024, 2][:1]}, "prediction")
    assert score == {"squads": 1, "mean_lineup_points": 16.0, "mean_captain_points": 8.0}

    snapshot = history.loc[history["player"] < 99, ["season", "player", "team", "position"]].rename(
        columns={"team": "context_team", "position": "context_position"},
    )
    snapshot = snapshot.assign(gameweek=2, context_asof=cutoff - pd.Timedelta(hours=1))
    # A prior regular absent from the dated roster cannot be restored by a later snapshot.
    snapshot = snapshot.loc[snapshot["player"] != fixture_player]
    late_readdition = snapshot.iloc[[0]].assign(player=fixture_player, context_asof=cutoff)
    snapshot = pd.concat([snapshot, late_readdition], ignore_index=True)
    squads_at_cutoff = replay_squads(history, targets, sample_count=3, player_context=snapshot)
    assert len(squads_at_cutoff[2024, 2]) == 3
    assert all(fixture_player not in squad for squad in squads_at_cutoff[2024, 2])
