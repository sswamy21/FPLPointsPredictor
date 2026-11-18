"""Opt-in smoke test against a warmed local server using official FPL data."""

import csv
import io
import os
from collections import Counter
from datetime import datetime, timezone
from math import fsum

import pytest
import requests

SERVER = os.environ.get("FPL_TEST_URL")
pytestmark = pytest.mark.skipif(not SERVER, reason="Set FPL_TEST_URL to a running dashboard.")


def _check_lineup(result, status):
    assert result["gameweek"] == status["gameweek"]
    assert result["season"] == status["season"]
    starters = result["starters"]
    assert len(starters) == 11 and len(result["bench"]) == 3
    assert result["goalkeeper"]["position"] == "GK"
    players = starters + result["bench"] + [result["goalkeeper"]]
    assert len({player["id"] for player in players}) == 15
    counts = Counter(player["position"] for player in starters)
    assert counts["GK"] == 1 and 3 <= counts["DEF"] <= 5
    assert 2 <= counts["MID"] <= 5 and 1 <= counts["FWD"] <= 3
    assert result["formation"] == '-'.join(str(counts[position]) for position in ("DEF", "MID", "FWD"))
    ranking = sorted(starters, key=lambda player: (-player["points"], player["id"]))
    assert [player["role"] for player in ranking] == ["C", "VC"] + [None] * 9
    assert result["projected_points"] == pytest.approx(
        fsum(player["points"] for player in starters) + ranking[0]["points"]
    )
    bench_scores = [player["points"] for player in result["bench"]]
    assert bench_scores == sorted(bench_scores, reverse=True)
    for player in players:
        assert player["points"] == pytest.approx(fsum(game["points"] for game in player["fixtures"]))


def test_upload_and_manager_against_live_fpl():
    status_response = requests.get(f"{SERVER}/api/status", timeout=10)
    status_response.raise_for_status()
    status = status_response.json()
    assert status["state"] == "ready" and status["warning"] is None
    assert datetime.fromisoformat(status["deadline"]) > datetime.now(timezone.utc)
    assert status["coverage"]["seasons"] >= 10
    assert len(status["metrics"]["folds"]) >= 3

    sample = requests.get(f"{SERVER}/sample.csv", timeout=10)
    sample.raise_for_status()
    ids = [int(row["id"]) for row in csv.DictReader(io.StringIO(sample.text))]
    assert len(ids) == 15
    csv_input = {"filename": "sample.csv", "contents": sample.text}
    json_input = {"filename": "sample.json", "contents": str(ids)}
    manager_input = {"manager_id": os.environ.get("FPL_TEST_MANAGER", "1")}
    results = []
    for inputs in (csv_input, json_input, manager_input):
        response = requests.post(f"{SERVER}/api/optimize", json=inputs, timeout=30)
        response.raise_for_status()
        result = response.json()
        _check_lineup(result, status)
        results.append(result)
    assert results[0]["starters"] == results[1]["starters"]
    assert results[0]["bench"] == results[1]["bench"]
    assert results[2]["source"]["type"] == "manager"
    assert results[2]["source"]["gameweek"] < status["gameweek"]


def test_transfer_search_against_live_fpl():
    response = requests.post(f"{SERVER}/api/optimize", json={"demo": True}, timeout=15)
    response.raise_for_status()
    lineup = response.json()
    players = lineup["starters"] + lineup["bench"] + [lineup["goalkeeper"]]
    player_ids = {player["id"] for player in players}
    payload = {
        "season": lineup["season"], "gameweek": lineup["gameweek"], "squad_ids": list(player_ids),
        "bank": "5.0", "free_transfers": 1, "horizon": len(lineup["transfer_gameweeks"]),
        "selling_prices": {str(player["id"]): f'{player["price"] / 10:.1f}' for player in players},
    }
    for allowance, hit in ((1, 0), (0, 4)):
        response = requests.post(f"{SERVER}/api/transfers", json=dict(payload, free_transfers=allowance), timeout=30)
        response.raise_for_status()
        targets = response.json()
        assert targets["gameweeks"] == lineup["transfer_gameweeks"]
        assert targets["hit"] == hit
        gains = [target["net_gain"] for target in targets["candidates"]]
        assert gains == sorted(gains, reverse=True)
        for target in targets["candidates"]:
            assert target["outgoing"]["id"] in player_ids and target["incoming"]["id"] not in player_ids
            assert target["incoming"]["position"] == target["outgoing"]["position"]
            assert target["bank_after"] == 50 + target["sale_price"] - target["incoming"]["price"] >= 0
            assert not target["estimated_sale_price"]
            assert target["weeks"][0]["baseline_points"] == pytest.approx(lineup["projected_points"])
            assert target["net_gain"] == pytest.approx(fsum(week["gain"] for week in target["weeks"]) - hit)
    stale = requests.post(f"{SERVER}/api/transfers", json=dict(payload, gameweek=0), timeout=10)
    assert stale.status_code == 409
    invalid = requests.post(f"{SERVER}/api/transfers", json=dict(payload, bank="0.05"), timeout=10)
    assert invalid.status_code == 400
