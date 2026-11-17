from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from fpl.data import DataStore, _normalize_fixtures, _normalize_players


def test_postponed_rows_and_retrospective_difficulty_are_not_used():
    raw = pd.DataFrame([{
        "id": 275, "event": 39, "kickoff_time": "2020-06-17T19:15:00Z",
        "team_h": 11, "team_a": 1, "team_h_score": 3, "team_a_score": 0,
        "finished": True, "team_h_difficulty": 3, "team_a_difficulty": 5,
    }])
    fixtures = _normalize_fixtures(raw, 2019, {11: 43, 1: 3})
    assert fixtures["home_difficulty"].isna().all()
    assert fixtures.loc[0, "deadline"] == pd.Timestamp("2020-06-17T18:15:00Z")

    player_rows = pd.DataFrame([
        {"fixture": 275, "element": 214, "round": 29, "was_home": True, "opponent_team": 1,
         "total_points": 0, "minutes": 0},
        {"fixture": 275, "element": 214, "round": 39, "was_home": True, "opponent_team": 1,
         "total_points": 10, "minutes": 90},
    ])
    roster = pd.DataFrame([{"id": 214, "code": 103955, "element_type": 3, "team": 4}])
    history = _normalize_players(player_rows, roster, fixtures, 2019)
    assert list(history["gameweek"]) == [39]
    assert list(history["player"]) == [103955]
    assert list(history["team"]) == [11]
    assert list(history["points"]) == [10]
    assert history["expected_goals"].isna().all()


def test_public_squad_reverts_free_hit_and_excludes_unpublished_gameweek(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    live = SimpleNamespace(bootstrap={"events": [
        {"id": 1, "deadline_time": (now - timedelta(days=8)).isoformat()},
        {"id": 2, "deadline_time": (now - timedelta(days=1)).isoformat()},
        {"id": 3, "deadline_time": (now + timedelta(days=6)).isoformat()},
    ]})
    store = DataStore(tmp_path)
    identity = [23 + i for i in range(15)]
    temporary = [80 + i for i in range(15)]
    responses = {
        "entry/42": {"started_event": 1},
        "entry/42/event/2/picks": {
            "picks": [{"element": player} for player in temporary],
            "active_chip": "freehit",
        },
        "entry/42/event/1/picks": {"picks": [{"element": player} for player in identity]},
    }
    monkeypatch.setattr(store, "_api_json", responses.__getitem__)
    assert store.public_picks(42, live) == (identity, 1)
