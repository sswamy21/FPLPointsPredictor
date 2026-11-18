import pytest

from fpl.gameweek_value import evaluate_gameweek


def player(position, points, appearance=1.0, role=None):
    return {"position": position, "points": points, "appearance_chance": appearance, "role": role}


def test_no_show_cover_respects_minimum_defenders_and_vice_captain():
    lineup = {
        "starters": [player("GK", 3, 0.75), player("DEF", 6, 0.5, "C"),
                     *[player("DEF", 2) for _ in range(2)],
                     player("MID", 5, role="VC"), *[player("MID", 2) for _ in range(3)],
                     *[player("FWD", 2) for _ in range(3)]],
        "bench": [player("MID", 4), player("DEF", 2.4, 0.8), player("MID", 1)],
        "goalkeeper": player("GK", 2, 0.5),
        "projected_points": 36,
    }
    result = evaluate_gameweek(lineup)
    assert result["autosub_points"] == pytest.approx(0.25 * 2 + 0.5 * 2.4)
    assert result["vice_captain_points"] == pytest.approx(0.5 * 5)
    assert result["points"] == pytest.approx(40.2)


def test_first_eligible_bench_player_fills_open_outfield_slot():
    lineup = {
        "starters": [player("GK", 3), *[player("DEF", 2) for _ in range(4)],
                     player("MID", 0, 0), player("MID", 5, role="VC"),
                     *[player("MID", 2) for _ in range(2)],
                     player("FWD", 8, role="C"), player("FWD", 2)],
        "bench": [player("DEF", 2, 0.5), player("MID", 3), player("FWD", 1)],
        "goalkeeper": player("GK", 2),
        "projected_points": 38,
    }
    result = evaluate_gameweek(lineup)
    assert result["autosub_points"] == pytest.approx(2 + 0.5 * 3)
    assert result["vice_captain_points"] == 0


def test_two_absences_can_use_a_midfielder_and_keep_a_defender_slot_open():
    lineup = {
        "starters": [player("GK", 3), player("DEF", 0, 0), player("DEF", 2), player("DEF", 2),
                     player("MID", 0, 0), player("MID", 5, role="VC"), player("MID", 2), player("MID", 2),
                     player("FWD", 8, role="C"), player("FWD", 2), player("FWD", 2)],
        "bench": [player("MID", 5), player("MID", 4), player("DEF", 3)],
        "goalkeeper": player("GK", 2),
        "projected_points": 36,
    }
    assert evaluate_gameweek(lineup)["autosub_points"] == 8
