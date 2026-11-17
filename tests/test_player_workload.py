from types import SimpleNamespace

import pandas as pd

from fpl.player_workload import _event_workload, _season_lookup, workload_features


def _entry(id_, name, *, starter=False, in_=False, out=False):
    return {
        "starter": starter, "subbedIn": in_, "subbedOut": out,
        "athlete": {"id": id_, "fullName": name},
    }


def _play(kind, period, clock, *players):
    return {
        "type": {"type": kind}, "period": {"number": period}, "clock": {"value": clock},
        "participants": [{"athlete": {"id": player}} for player in players],
    }


def test_workload_uses_only_a_recorded_substitution_clock_for_minutes():
    summary = {
        "header": {"competitions": [{"status": {"type": {"completed": True, "shortDetail": "FT"}}}]},
        "keyEvents": [_play("substitution", 2, 3540, "2", "1"), _play("red-card", 2, 4800, "3")],
        "rosters": [{"team": {"displayName": "Arsenal"}, "roster": [
            _entry("1", "Player Off", starter=True, out=True),
            _entry("2", "Player On", in_=True),
            _entry("3", "Dismissed", starter=True),
            _entry("4", "Unused"),
            _entry("5", "Missing Time", in_=True),
        ]}],
    }
    result = _event_workload(summary)
    assert [row["minutes"] for row in result] == [59.0, 31.0, 80.0, 0.0, None]
    assert [row["starts"] for row in result] == [1, 0, 1, 0, 0]
    assert [row["appearances"] for row in result] == [1, 1, 1, 0, 1]
    assert {row["team_code"] for row in result} == {3}


def test_season_identity_matches_accents_and_rejects_ambiguous_aliases(tmp_path):
    live = SimpleNamespace(season=2026, bootstrap={"elements": [
        {"id": 1, "code": 111, "first_name": "Robert", "second_name": "Sánchez", "web_name": "Sánchez"},
        {"id": 2, "code": 222, "first_name": "Álex", "second_name": "Sánchez", "web_name": "Sánchez"},
    ]})
    aliases = _season_lookup(tmp_path, live, 2026)
    assert aliases == {"robertsanchez": 111, "alexsanchez": 222}


def test_recent_player_workload_ignores_future_and_unobserved_results():
    cutoff = pd.Timestamp("2026-11-21T11:00:00Z")
    data = pd.DataFrame({
        "player": [42, 42, 42, 42, 43, 43],
        "kickoff": pd.to_datetime([
            "2026-11-13T19:00Z", "2026-11-17T18:00Z", "2026-11-21T09:00Z",
            "2026-11-21T12:00Z", "2026-11-17T18:00Z", "2026-11-19T20:00Z",
        ]),
        "observed_at": pd.to_datetime([
            "2026-11-13T23:00Z", "2026-11-17T22:00Z", "2026-11-21T13:00Z",
            "2026-11-21T16:00Z", "2026-11-17T22:00Z", "2026-11-20T00:00Z",
        ]),
        "minutes": [90, 60, 90, 90, None, 60], "starts": [1, 1, 1, 1, 0, 1], "appearances": [1, 1, 1, 1, 1, 1],
    })
    players = pd.Series([42, 43, 44])
    features = workload_features(data, players, cutoff)
    assert features.iloc[0].tolist() == [60, 1, 1]
    assert pd.isna(features.iloc[1, 0]) and features.iloc[1, 1:].tolist() == [1, 2]
    assert features.iloc[2].isna().all()
