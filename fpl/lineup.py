"""Squad input and exact lineup optimization for the projected-points objective."""

import csv
import io
import json
import math
import unicodedata
from collections import Counter, defaultdict
from itertools import product

SQUAD_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
POSITION_IDS = dict(enumerate(SQUAD_COUNTS, 1))
STARTER_RANGES = (range(3, 6), range(2, 6), range(1, 4))
FORMATIONS = tuple(shape for shape in product(*STARTER_RANGES) if sum(shape) == 10)


class SquadError(ValueError):
    """A squad input that the user can correct."""


def _normalized_name(name):
    return " ".join(
        "".join(
            char for char in unicodedata.normalize("NFKD", name.casefold())
            if not unicodedata.combining(char)
        ).split()
    )


def _references(filename, contents):
    contents = contents.lstrip("\ufeff").strip()
    if filename.lower().endswith(".json") or contents.startswith(("[", "{")):
        try:
            payload = json.loads(contents)
        except json.JSONDecodeError as error:
            raise SquadError(f"Invalid JSON on line {error.lineno}.") from error
        references = payload.get("players") if isinstance(payload, dict) else payload
        if not isinstance(references, list):
            raise SquadError('JSON must be a list or an object with a "players" list.')
        return references

    try:
        reader = csv.DictReader(io.StringIO(contents), strict=True)
        columns = reader.fieldnames or []
        if not {column.strip().lower() for column in columns} & {"id", "element", "name"}:
            raise SquadError('Use a CSV with an "id" or "name" header.')
        return [
            {key.strip().lower(): value for key, value in row.items() if key is not None}
            for row in reader
        ]
    except csv.Error as error:
        raise SquadError(f"Invalid CSV. {error}.") from error


def resolve_upload(filename, contents, elements):
    references = _references(filename, contents)
    if len(references) != 15:
        raise SquadError(f"Supply exactly 15 players. This file contains {len(references)}.")

    labels = {}
    names = defaultdict(set)
    for player in elements:
        if player["element_type"] not in POSITION_IDS:
            continue
        element = player["id"]
        full_name = f'{player["first_name"]} {player["second_name"]}'
        labels[element] = f"{full_name} (ID {element})"
        for name in (full_name, player["web_name"]):
            names[_normalized_name(name)].add(element)

    resolved = []
    for number, ref in enumerate(references, start=1):
        if isinstance(ref, dict):
            ref = ref.get("id") or ref.get("element") or ref.get("name")
        if isinstance(ref, bool) or not isinstance(ref, (str, int)) or not str(ref).strip():
            raise SquadError(f"Player {number} needs an FPL player ID or name.")
        ref = str(ref).strip()
        if ref.isascii() and ref.isdecimal() and len(ref) <= 10:
            element = int(ref)
            if element not in labels:
                raise SquadError(f"Player ID {element} is outside this season’s player list.")
        else:
            matches = sorted(names.get(_normalized_name(ref), ()))
            if not matches:
                raise SquadError(f'Player {number}, "{ref}", could not be matched. Use their FPL ID.')
            if len(matches) > 1:
                choices = ", ".join(labels[element] for element in matches)
                raise SquadError(f'"{ref}" is ambiguous. Use the FPL ID for one of these players: {choices}.')
            element = matches[0]
        resolved.append(element)
    return resolved


def _points_order(player):
    return -float(player["points"]), player["id"]


def demo_squad(projections):
    """Pick an illustrative current-season squad with the standard club limit."""
    slots = Counter(SQUAD_COUNTS)
    clubs = Counter()
    squad = []
    for player in sorted(projections.values(), key=_points_order):
        position, team = player["position"], player["team"]
        if slots[position] > 0 and clubs[team] < 3:
            slots[position] -= 1
            clubs[team] += 1
            squad.append(player["id"])
    if len(squad) != 15:
        raise SquadError("There are too few active players to build a sample squad.")
    return squad


def optimize(squad_ids, projections):
    """Maximize XI projection plus the highest starter's captain bonus.

    For each legal formation, take the highest scoring players in its position
    groups. Replacing a player by a better one in the same group cannot lower
    either the XI sum or its maximum, so one lineup per formation is sufficient.
    """
    if len(squad_ids) != 15 or len(set(squad_ids)) != 15:
        raise SquadError("Supply exactly 15 distinct FPL players.")
    if missing := sorted(set(squad_ids) - projections.keys()):
        raise SquadError(f"No current projection for player IDs {', '.join(map(str, missing))}.")
    squad = [projections[element] for element in squad_ids]
    if Counter(player["position"] for player in squad) != SQUAD_COUNTS:
        raise SquadError("A squad needs 2 goalkeepers, 5 defenders, 5 midfielders and 3 forwards.")
    if not all(math.isfinite(float(player["points"])) for player in squad):
        raise RuntimeError("The predictor returned a non-finite points value.")

    by_position = defaultdict(list)
    for player in sorted(squad, key=_points_order):
        by_position[player["position"]].append(player)

    def build_formation(shape):
        lineup = [by_position["GK"][0]]
        for position, count in zip(("DEF", "MID", "FWD"), shape):
            lineup.extend(by_position[position][:count])
        starting_points = math.fsum(float(player["points"]) for player in lineup)
        captain, vice = sorted(lineup, key=_points_order)[:2]
        score = math.fsum((starting_points, float(captain["points"])))
        return score, starting_points, lineup, captain, vice

    shape, result = max(
        ((shape, build_formation(shape)) for shape in FORMATIONS), key=lambda item: item[1][0]
    )
    score, starting_points, lineup, captain, vice = result
    roles = {captain["id"]: "C", vice["id"]: "VC"}
    used = {player["id"] for player in lineup}
    return {
        "formation": "-".join(map(str, shape)),
        "starters": [dict(player, role=roles.get(player["id"])) for player in lineup],
        "goalkeeper": dict(by_position["GK"][1], role=None),
        "bench": [
            dict(player, role=None) for player in sorted(squad, key=_points_order)
            if player["position"] != "GK" and player["id"] not in used
        ],
        "projected_points": score,
        "starting_points": starting_points,
        "captain_bonus": float(captain["points"]),
    }
