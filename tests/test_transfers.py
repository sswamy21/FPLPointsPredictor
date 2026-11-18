import pytest

from fpl.lineup import POSITION_IDS, SquadError
from fpl.transfers import recommend_transfers


def market():
    positions = (1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4)
    squad = list(range(1, 16))
    elements = [
        {"id": player_id, "element_type": position, "team": (player_id - 1) // 3 + 1, "now_cost": 50}
        for player_id, position in zip(squad, positions)
    ]
    return squad, elements


def add_player(elements, player_id, position=3, team=6, price=50, **flags):
    elements.append({"id": player_id, "element_type": position, "team": team, "now_cost": price, **flags})


def projections(elements, *changes):
    return {
        week: {
            element["id"]: {
                "id": element["id"], "name": f"P{element['id']}", "team": f"T{element['team']}",
                "position": POSITION_IDS[element["element_type"]], "points": overrides.get(element["id"], 2),
                "fixtures": [{"opponent": "FOE", "home": True, "points": overrides.get(element["id"], 2)}],
            }
            for element in elements
        }
        for week, overrides in enumerate(changes, start=11)
    }


def test_rankings_use_full_horizon_and_reoptimized_captaincy():
    squad, elements = market()
    add_player(elements, 16)
    add_player(elements, 17)
    points = projections(elements, {16: 7, 17: 3}, *([{16: 1, 17: 3}] * 3), {16: 1, 17: 5})
    four_weeks = {week: values for week, values in points.items() if week < 15}
    shorter = recommend_transfers(squad, four_weeks, elements, bank=0, free_transfers=0, outgoing_id=8)
    assert shorter["candidates"][0]["incoming"]["id"] == 16

    result = recommend_transfers(squad, points, elements, bank=0, free_transfers=0, outgoing_id=8)
    assert result["gameweeks"] == [11, 12, 13, 14, 15]
    assert result["baseline_points"] == 120
    assert result["hit"] == 4
    assert [candidate["incoming"]["id"] for candidate in result["candidates"]] == [17, 16]
    winner = result["candidates"][0]
    assert winner["gross_gain"] == 14
    assert winner["net_gain"] == 10
    assert [week["gain"] for week in winner["weeks"]] == [2, 2, 2, 2, 6]
    assert all(week["starts"] and week["captain"] for week in winner["weeks"])
    assert [fixture["gameweek"] for fixture in winner["incoming"]["fixtures"]] == [11, 12, 13, 14, 15]
    assert [week["starts"] for week in result["candidates"][1]["weeks"]] == [True, False, False, False, False]


def test_budget_uses_confirmed_selling_price_or_marks_estimate():
    squad, elements = market()
    add_player(elements, 16, price=51)
    points = projections(elements, {16: 8})

    def run(bank, selling_prices=None):
        return recommend_transfers(squad, points, elements, bank=bank, free_transfers=1,
                                   selling_prices=selling_prices, outgoing_id=8)["candidates"]

    assert run(0) == []
    assert run(1)[0]["estimated_sale_price"] is True
    assert run(1, {8: 49}) == []
    assert run(2, {8: 49})[0]["estimated_sale_price"] is False
    assert run(2, {8: 49})[0]["sale_price"] == 49
    assert run(2, {8: 49})[0]["bank_after"] == 0


def test_single_transfer_must_resolve_existing_club_excess():
    squad, elements = market()
    elements[7]["team"] = 1
    add_player(elements, 16, team=1)
    add_player(elements, 17, team=6)
    add_player(elements, 18, position=4, team=6)
    points = projections(elements, {16: 16, 17: 8, 18: 20})
    candidates = recommend_transfers(squad, points, elements, bank=0, free_transfers=1)["candidates"]
    assert [(candidate["outgoing"]["id"], candidate["incoming"]["id"]) for candidate in candidates] == [(8, 17)]


def test_unavailable_for_purchase_and_over_quota_are_excluded():
    squad, elements = market()
    for player_id, flags in enumerate((
        {"removed": True}, {"can_select": False}, {"can_transact": False}, {"status": "u"}, {"status": "n"},
    ), start=16):
        add_player(elements, player_id, **flags)
    add_player(elements, 21, team=1)
    add_player(elements, 22, status="d")
    points = projections(elements, {player_id: 8 for player_id in range(16, 23)})
    candidates = recommend_transfers(squad, points, elements, bank=0, free_transfers=1)["candidates"]
    assert [candidate["incoming"]["id"] for candidate in candidates] == [22]


def test_dedupe_and_ties_prefer_bank_remaining_then_ids():
    squad, elements = market()
    elements[7]["now_cost"] = 55
    elements[8]["now_cost"] = 55
    add_player(elements, 16, price=50)
    add_player(elements, 17, price=51)
    add_player(elements, 18, price=50)
    points = projections(elements, {16: 6, 17: 6, 18: 6})
    candidates = recommend_transfers(squad, points, elements, bank=0, free_transfers=1)["candidates"]
    assert [candidate["incoming"]["id"] for candidate in candidates] == [16, 18, 17]
    assert all(candidate["outgoing"]["id"] == 8 for candidate in candidates)


def test_hold_when_gross_gain_only_covers_the_hit():
    squad, elements = market()
    add_player(elements, 16)
    points = projections(elements, {16: 4})
    result = recommend_transfers(squad, points, elements, bank=0, free_transfers=0)
    assert result["hit"] == 4
    assert result["candidates"] == []


def test_ranks_a_bench_upgrade_by_its_expected_autosub_points():
    squad, elements = market()
    add_player(elements, 16, position=1)
    points = projections(elements, {1: 3, 2: 2, 16: 2.8})
    points[11][1]["appearance_chance"] = 0.5
    result = recommend_transfers(squad, points, elements, bank=0, free_transfers=1, outgoing_id=2)
    target = result["candidates"][0]
    assert target["incoming"]["id"] == 16
    assert not target["weeks"][0]["starts"]
    assert target["net_gain"] == pytest.approx(0.4)


@pytest.mark.parametrize("override, message", [
    ({"bank": -1}, "bank"),
    ({"free_transfers": 6}, "free transfers"),
    ({"outgoing_id": 16}, "squad"),
    ({"selling_prices": {8: 51}}, "current price"),
    ({"selling_prices": {16: 40}}, "squad"),
])
def test_rejects_invalid_user_inputs(override, message):
    squad, elements = market()
    with pytest.raises(SquadError, match=message):
        recommend_transfers(squad, projections(elements, {}), elements, **({"bank": 0, "free_transfers": 1} | override))
