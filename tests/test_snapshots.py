from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from fpl import snapshots


SEASON = 2025
DEADLINE = pd.Timestamp("2025-08-22T17:30:00Z")


def _bootstrap(deadline=DEADLINE, *, finished=False, is_next=True):
    return {
        "events": [
            {"id": 1, "deadline_time": "2025-08-15T17:30:00Z", "finished": True},
            {"id": 2, "deadline_time": deadline.isoformat(), "finished": finished, "is_next": is_next},
        ],
        "elements": [
            {
                "id": 10, "code": 42, "element_type": 3, "team": 5,
                "chance_of_playing_next_round": 75, "chance_of_playing_this_round": 0,
                "status": "d", "news": "Doubtful", "news_added": "2025-08-20T12:00:00Z",
                "penalties_order": 1, "direct_freekicks_order": None, "ep_next": "4.5",
            },
            {
                "id": 11, "code": 43, "element_type": 2, "team": 5,
                "chance_of_playing_next_round": None, "status": "a", "news": "Updated",
                "news_added": "2025-08-22T18:00:00Z",
            },
        ],
    }


def test_round_capture_requires_the_exact_deadline_and_target_event():
    previous = DEADLINE - pd.Timedelta(hours=1)
    assert snapshots._accept_capture(_bootstrap(), SEASON, 2, DEADLINE, previous, historical=True)
    assert not snapshots._accept_capture(_bootstrap(), SEASON, 2, DEADLINE, DEADLINE, historical=True)
    assert not snapshots._accept_capture(_bootstrap(is_next=False), SEASON, 2, DEADLINE, previous, historical=True)
    assert not snapshots._accept_capture(_bootstrap(finished=True), SEASON, 2, DEADLINE, previous, historical=True)
    assert not snapshots._accept_capture(
        _bootstrap(DEADLINE + pd.Timedelta(days=1)), SEASON, 2, DEADLINE, previous, historical=True,
    )


def test_snapshot_selection_ignores_after_deadline_and_wrong_schedule(monkeypatch, tmp_path):
    references = [
        (datetime(2025, 8, 22, 9, 2, tzinfo=timezone.utc), "valid"),
        (datetime(2025, 8, 22, 15, 3, tzinfo=timezone.utc), "wrong schedule"),
        (datetime(2025, 8, 22, 17, 31, tzinfo=timezone.utc), "late"),
    ]
    calls = []

    def capture(_, path):
        calls.append(path)
        return _bootstrap(DEADLINE + pd.Timedelta(hours=1)) if path == "wrong schedule" else _bootstrap()

    monkeypatch.setattr(snapshots, "_capture", capture)
    rows = snapshots._one_round(tmp_path, references, [stamp for stamp, _ in references], SEASON, 2, DEADLINE)
    assert calls == ["wrong schedule", "valid"]
    assert [row["player"] for row in rows] == [42, 43]
    first, second = rows
    assert first["flag_probability"] == 0.75 and second["flag_probability"] == 1.0
    assert first["penalty_order"] == 1.0 and first["direct_fk_order"] == 0.0
    assert pd.isna(first["corner_fk_order"]) and pd.isna(second["flag_age_days"])
    assert first["context_asof"] < first["context_deadline"]
    assert first["provider_ep_next"] == 4.5


def test_network_failure_keeps_live_context(monkeypatch, tmp_path):
    def failure():
        raise snapshots.DataError("Unavailable")

    monkeypatch.setattr(snapshots, "_archive_index", failure)
    fixtures = pd.DataFrame({"season": [2024]})
    stamp = DEADLINE - pd.Timedelta(minutes=20)
    live = SimpleNamespace(season=SEASON, gameweek=2, bootstrap=_bootstrap(), deadline=DEADLINE, refreshed_at=stamp)
    context = snapshots.load_player_context(tmp_path, fixtures, live)
    assert list(context["season"]) == [SEASON, SEASON]
    assert list(context["context_asof"]) == [stamp, stamp]


def test_minutes_input_requires_a_strict_predeadline_timestamp(tmp_path):
    path = tmp_path / "minutes.csv"
    path.write_text(
        "season,gameweek,player,context_asof,expected_minutes,predicted_start_probability\n"
        "2025,2,42,2025-08-22T12:00:00Z,50,0.6\n"
        "2025,2,42,2025-08-22T17:29:00+00:00,77,0.8\n"
        "2025,2,42,2025-08-22T17:30:00Z,90,1\n"
        "2025,3,42,2025-08-22T17:20:00Z,80,0.9\n"
        "2025,2,43,2025-08-22T18:20:00+01:00,,0.7\n",
        encoding="utf-8",
    )
    schedule = pd.DataFrame({"season": [SEASON, SEASON], "gameweek": [2, 2], "deadline": [DEADLINE, DEADLINE]})
    result = snapshots.load_minutes_context(path, schedule).set_index("player")
    assert result.index.sort_values().tolist() == [42, 43]
    assert result.at[42, "expected_minutes"] == 77
    assert result.at[42, "predicted_start_probability"] == 0.8
    assert result.at[43, "predicted_start_probability"] == 0.7
    assert pd.isna(result.at[43, "expected_minutes"])


def test_minutes_input_rejects_naive_timestamps(tmp_path):
    path = tmp_path / "minutes.csv"
    path.write_text(
        "season,gameweek,player,context_asof,expected_minutes\n"
        "2025,2,42,2025-08-22T12:00:00,40\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit UTC offset"):
        snapshots.load_minutes_context(path, pd.DataFrame())
