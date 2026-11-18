"""Rank affordable single transfers by the change to the optimized lineup."""

from collections import Counter, defaultdict
from math import fsum

from . import TRANSFER_HORIZON
from .gameweek_value import evaluate_gameweek
from .lineup import POSITION_IDS, SquadError, optimize


def _valid_amount(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _can_buy(element):
    return (
        not element.get("removed") and element.get("can_select") is not False
        and element.get("can_transact") is not False and element.get("status") not in {"u", "n"}
    )


def _player_summary(element, weeks):
    player_id = element["id"]
    player = weeks[0][1][player_id]
    fixtures = [
        dict(fixture, gameweek=week)
        for week, projections in weeks
        for fixture in projections[player_id].get("fixtures", ())
    ]
    return {
        "id": player_id, "name": player["name"], "team": player["team"],
        "position": player["position"], "price": element["now_cost"],
        "status": element.get("status"), "fixtures": fixtures,
    }


def _rank(candidate):
    return (-candidate["net_gain"], -candidate["bank_after"],
            candidate["incoming"]["id"], candidate["outgoing"]["id"])


def recommend_transfers(squad_ids, projections_by_week, elements, *, bank, free_transfers,
                        selling_prices=None, outgoing_id=None, limit=8):
    """Evaluate each legal swap independently, in integer tenths of a million pounds."""
    if not _valid_amount(bank):
        raise SquadError("Enter a non-negative amount in the bank.")
    if not _valid_amount(free_transfers) or free_transfers > 5:
        raise SquadError("Enter between zero and five free transfers.")
    if not 1 <= len(projections_by_week) <= TRANSFER_HORIZON:
        raise SquadError(f"Choose between 1 and {TRANSFER_HORIZON} Gameweeks for transfer projections.")

    weeks = sorted(projections_by_week.items())
    baseline = {week: evaluate_gameweek(optimize(squad_ids, projections))["points"] for week, projections in weeks}
    owned_ids = set(squad_ids)
    if outgoing_id is not None and outgoing_id not in owned_ids:
        raise SquadError("Choose a player from your squad to transfer out.")

    catalog = {
        element["id"]: element for element in elements
        if element["element_type"] in POSITION_IDS and _valid_amount(element.get("now_cost"))
        and element["now_cost"] > 0
    }
    if not owned_ids.issubset(catalog):
        raise SquadError("The official price list is missing one or more squad players.")
    selling_prices = {} if selling_prices is None else selling_prices
    for player_id, price in selling_prices.items():
        if player_id not in owned_ids:
            raise SquadError("Enter selling prices for players in your squad.")
        if not _valid_amount(price) or not 0 < price <= catalog[player_id]["now_cost"]:
            raise SquadError("Each selling price must be positive and at most the player's current price.")

    owned = [catalog[player_id] for player_id in squad_ids]
    clubs = Counter(player["team"] for player in owned)
    available = defaultdict(list)
    for player_id, player in catalog.items():
        if player_id not in owned_ids and _can_buy(player) and all(player_id in pool for _, pool in weeks):
            available[player["element_type"]].append(player)

    hit = 0 if free_transfers else 4
    targets = {}
    for outgoing in owned:
        sell_id = outgoing["id"]
        if outgoing_id is not None and sell_id != outgoing_id:
            continue
        remaining_clubs = clubs.copy()
        remaining_clubs[outgoing["team"]] -= 1
        if max(remaining_clubs.values()) > 3:
            continue
        remaining_ids = [player_id for player_id in squad_ids if player_id != sell_id]
        sale_price = selling_prices.get(sell_id, outgoing["now_cost"])
        budget = bank + sale_price
        for incoming in available[outgoing["element_type"]]:
            buy_id = incoming["id"]
            price = incoming["now_cost"]
            if price > budget or remaining_clubs[incoming["team"]] >= 3:
                continue
            swapped = [*remaining_ids, buy_id]
            lineups = [(week, optimize(swapped, projections)) for week, projections in weeks]
            results = []
            for week, lineup in lineups:
                total = evaluate_gameweek(lineup)["points"]
                starter = next((player for player in lineup["starters"] if player["id"] == buy_id), None)
                results.append({
                    "gameweek": week, "baseline_points": baseline[week], "projected_points": total,
                    "gain": total - baseline[week], "starts": starter is not None,
                    "captain": starter is not None and starter["role"] == "C",
                })
            gross_gain = fsum(week["gain"] for week in results)
            net_gain = gross_gain - hit
            if net_gain <= 1e-9:
                continue
            candidate = {
                "outgoing": _player_summary(outgoing, weeks), "incoming": _player_summary(incoming, weeks),
                "sale_price": sale_price, "estimated_sale_price": sell_id not in selling_prices,
                "bank_after": budget - price, "gross_gain": gross_gain, "net_gain": net_gain, "weeks": results,
            }
            if buy_id not in targets or _rank(candidate) < _rank(targets[buy_id]):
                targets[buy_id] = candidate

    return {
        "gameweeks": list(baseline), "baseline_points": fsum(baseline.values()), "hit": hit,
        "candidates": sorted(targets.values(), key=_rank)[:limit],
    }
