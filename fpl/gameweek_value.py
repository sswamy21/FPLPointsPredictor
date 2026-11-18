"""Expected automatic-substitution and vice-captain points for a selected XI."""

from functools import lru_cache
from itertools import product
from math import fsum

import numpy as np

from .lineup import STARTER_RANGES

OUTFIELD = ("DEF", "MID", "FWD")
MIN_STARTERS = tuple(allowed.start for allowed in STARTER_RANGES)
BENCH_APPEARANCES = np.array(list(product((False, True), repeat=3)))


def _appearance(player):
    chance = player.get("appearance_chance")
    return 1.0 if chance is None else min(1.0, max(0.0, float(chance)))


def _points_when_used(player, appearance):
    return float(player["points"]) / appearance if appearance else 0.0


@lru_cache(maxsize=None)
def _substitution_table(formation, bench_positions):
    """Map outfield absences and bench appearances to the slots that score."""
    states = list(product(*(range(count + 1) for count in formation)))
    taken = np.zeros((len(states), len(BENCH_APPEARANCES), 3))
    for row, absences in enumerate(states):
        vacant = sum(absences)
        if not vacant:
            continue
        playing = [count - absent for count, absent in zip(formation, absences)]
        for column, appears in enumerate(BENCH_APPEARANCES):
            remaining = vacant
            counts = playing.copy()
            for slot, (available, position) in enumerate(zip(appears, bench_positions)):
                if not available or not remaining:
                    continue
                counts[position] += 1
                # Unfilled places can keep their missing starter; enough places must
                # remain to meet all position minimums in the nominal XI.
                required = sum(max(0, minimum - count) for minimum, count in zip(MIN_STARTERS, counts))
                if required < remaining:
                    taken[row, column, slot] = 1.0
                    remaining -= 1
                else:
                    counts[position] -= 1
    return taken


def _outfield_sub_points(starters, bench):
    groups = [[_appearance(player) for player in starters if player["position"] == pos] for pos in OUTFIELD]
    formation = tuple(map(len, groups))
    bench_positions = tuple(OUTFIELD.index(player["position"]) for player in bench)
    appearance = np.array([_appearance(player) for player in bench])
    bench_states = np.where(BENCH_APPEARANCES, appearance, 1 - appearance).prod(axis=1)
    starter_states = np.array([1.0])
    for group in groups:
        distribution = np.array([1.0])
        for chance in group:
            distribution = np.convolve(distribution, [chance, 1 - chance])
        starter_states = np.outer(starter_states, distribution).ravel()
    table = _substitution_table(formation, bench_positions)
    active = np.einsum("i,j,ijk->k", starter_states, bench_states, table)
    return float(active @ [_points_when_used(player, chance) for player, chance in zip(bench, appearance)])


def evaluate_gameweek(lineup):
    """Score the given XI and bench order under independent appearance probabilities."""
    starters = lineup["starters"]
    outfield = [player for player in starters if player["position"] != "GK"]
    keeper = next(player for player in starters if player["position"] == "GK")
    captain = next(player for player in starters if player["role"] == "C")
    vice = next(player for player in starters if player["role"] == "VC")
    autosubs = (1 - _appearance(keeper)) * float(lineup["goalkeeper"]["points"])
    if not all(_appearance(player) == 1 for player in outfield):
        autosubs += _outfield_sub_points(outfield, lineup["bench"])
    vice_bonus = (1 - _appearance(captain)) * float(vice["points"])
    return {
        "points": fsum((lineup["projected_points"], autosubs, vice_bonus)),
        "autosub_points": autosubs,
        "vice_captain_points": vice_bonus,
    }
