from collections import Counter
from itertools import combinations
from math import fsum
from random import Random

import pytest

from fpl.lineup import SQUAD_COUNTS, SquadError, optimize, resolve_upload


def fixture_squad(seed):
    random = Random(seed)
    positions = [position for position, count in SQUAD_COUNTS.items() for _ in range(count)]
    return {
        number: {"id": number, "position": position, "points": random.randrange(-4, 13) / 2}
        for number, position in enumerate(positions, 1)
    }


def legal(players):
    positions = Counter(player["position"] for player in players)
    return (
        positions["GK"] == 1 and 3 <= positions["DEF"] <= 5
        and 2 <= positions["MID"] <= 5 and 1 <= positions["FWD"] <= 3
    )


@pytest.mark.parametrize("seed", range(10))
def test_formation_and_captain_are_globally_optimal(seed):
    players = fixture_squad(seed)
    result = optimize(list(reversed(players)), players)
    starters = result["starters"]
    bench = result["bench"]
    assert len(starters) == 11 and legal(starters)
    assert len(bench) == 3 and all(player["position"] != "GK" for player in bench)
    assert result["goalkeeper"]["position"] == "GK"
    assert set(player["id"] for player in starters + bench + [result["goalkeeper"]]) == set(players)
    assert [player["points"] for player in bench] == sorted((player["points"] for player in bench), reverse=True)
    ranked = sorted(starters, key=lambda player: (-player["points"], player["id"]))
    assert [player["role"] for player in ranked] == ["C", "VC"] + [None] * 9
    brute_force = max(
        fsum(player["points"] for player in eleven) + max(player["points"] for player in eleven)
        for eleven in combinations(players.values(), 11) if legal(eleven)
    )
    assert result["projected_points"] == brute_force


def test_squad_composition_and_duplicates_are_rejected():
    players = fixture_squad(1)
    with pytest.raises(SquadError, match="distinct"):
        optimize([1, 1, *range(3, 16)], players)
    players[1]["position"] = "FWD"
    with pytest.raises(SquadError, match="2 goalkeepers"):
        optimize(list(players), players)


def test_upload_resolves_current_ids_and_names():
    elements = [
        {"id": i, "element_type": 1, "first_name": "João", "second_name": str(i), "web_name": f"J{i}"}
        for i in range(1, 16)
    ]
    contents = '\ufeffid,name\n,Joao 1\n' + '\n'.join(f"{i}," for i in range(2, 16))
    assert resolve_upload("squad.csv", contents, elements) == list(range(1, 16))
    contents = '{"players":["J1",' + ','.join(map(str, range(2, 16))) + ']}'
    assert resolve_upload("squad.json", contents, elements) == list(range(1, 16))
    elements[1]["web_name"] = "J1"
    with pytest.raises(SquadError, match="ambiguous"):
        resolve_upload("squad.json", contents, elements)
