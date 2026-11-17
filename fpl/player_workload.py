"""Player appearances and approximate minutes in completed cup and European games."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import gzip
import json
import logging
from pathlib import Path
import unicodedata

import pandas as pd

from .congestion import API, CLUBS, COMPETITIONS
from .data import DataError, _download, _write, player_identity, season_label

LOG = logging.getLogger(__name__)
RESULT_LAG = timedelta(hours=4)
RECENT_WINDOW = timedelta(days=7)
FEATURES = {
    "minutes": "other_player_minutes_7d",
    "starts": "other_player_starts_7d",
    "appearances": "other_player_appearances_7d",
}
COLUMNS = ("event_id", "competition", "kickoff", "observed_at", "team_code", "player", *FEATURES)
PERIOD_BOUNDS = {1: (0, 2700), 2: (2700, 5400), 3: (5400, 6300), 4: (6300, 7200)}


def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def _normalized_name(value):
    if not isinstance(value, str):
        return ""
    parts = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in parts if char.isalnum())


def _season_lookup(cache, live, season):
    if season == live.season:
        elements = live.bootstrap["elements"]
    else:
        path = cache / "historical" / season_label(season) / "players_raw.csv"
        if not path.is_file():
            return {}
        columns = {"id", "code", "first_name", "second_name", "web_name"}
        try:
            roster = pd.read_csv(path, usecols=lambda column: column in columns)
        except UnicodeDecodeError:
            roster = pd.read_csv(path, usecols=lambda column: column in columns, encoding="latin1")
        elements = roster.to_dict("records")

    aliases = defaultdict(set)
    for element in elements:
        code = player_identity(element.get("code"), season, element["id"])
        full_name = " ".join(part for part in (element.get("first_name"), element.get("second_name")) if isinstance(part, str))
        for value in (full_name, element.get("web_name")):
            if name := _normalized_name(value):
                aliases[name].add(code)
    return {name: next(iter(codes)) for name, codes in aliases.items() if len(codes) == 1}


def _clock(event):
    period = event.get("period", {}).get("number")
    if period not in PERIOD_BOUNDS:
        return None
    first, last = PERIOD_BOUNDS[period]
    seconds = _number(event.get("clock", {}).get("value"))
    if not first <= seconds <= last + 900:
        return None
    return min(seconds, last)


def _match_duration(events, status):
    extra_time = any(
        event.get("period", {}).get("number") in (3, 4) and not event.get("shootout")
        and "shootout" not in event.get("type", {}).get("type", "")
        for event in events
    )
    if extra_time or status.get("shortDetail") == "AET":
        return 7200
    if status.get("shortDetail") in {"FT", "FT-Pens"}:
        return 5400
    return None


def _substitutions(events):
    on, off, dismissed = {}, {}, {}
    for event in events:
        kind = event.get("type", {}).get("type", "")
        if kind not in {"substitution", "red-card", "yellow-red-card", "second-yellow-card"}:
            continue
        seconds = _clock(event)
        if seconds is None:
            continue
        athletes = [str(participant.get("athlete", {}).get("id", "")) for participant in event.get("participants", ())]
        if kind == "substitution" and len(athletes) == 2 and all(athletes):
            on[athletes[0]] = seconds
            off[athletes[1]] = seconds
        elif kind != "substitution" and athletes and athletes[0]:
            dismissed[athletes[0]] = seconds
    return on, off, dismissed


def _athlete_minutes(entry, athlete_id, times, duration):
    started = bool(entry.get("starter"))
    appeared = started or bool(entry.get("subbedIn"))
    if not appeared:
        return 0.0
    if duration is None:
        return None
    on, off, dismissed = times
    red_cards = next((stat.get("value") for stat in entry.get("stats", ()) if stat.get("name") == "redCards"), 0)
    first = 0 if started else on.get(athlete_id)
    last = off.get(athlete_id) if entry.get("subbedOut") else duration
    if athlete_id in dismissed:
        last = min(last, dismissed[athlete_id]) if last is not None else dismissed[athlete_id]
    elif _number(red_cards) > 0:
        return None
    if first is None or last is None or not 0 <= first <= last <= duration:
        return None
    return (last - first) / 60


def _event_workload(payload):
    competitions = payload.get("header", {}).get("competitions", [])
    status = competitions[0].get("status", {}).get("type", {}) if competitions else {}
    if not status.get("completed"):
        raise DataError("ESPN summary has no completed match status.")
    events = payload.get("keyEvents") or []
    duration = _match_duration(events, status)
    times = _substitutions(events)
    rows = []
    for side in payload.get("rosters") or []:
        club = CLUBS.get(side.get("team", {}).get("displayName"))
        if not club:
            continue
        for entry in side.get("roster", ()):
            athlete = entry.get("athlete", {})
            athlete_id = athlete.get("id")
            if not athlete_id or "starter" not in entry or "subbedIn" not in entry:
                continue
            starts = int(bool(entry["starter"]))
            appearances = int(bool(starts or entry["subbedIn"]))
            names = [_normalized_name(athlete.get(field)) for field in ("fullName", "displayName")]
            rows.append({
                "team_code": club, "names": names,
                "starts": starts, "appearances": appearances,
                "minutes": _athlete_minutes(entry, str(athlete_id), times, duration),
            })
    return rows


def load_player_workload(cache, other_fixtures, live, *, download=False):
    """Map available ESPN match rosters to stable FPL player codes.

    By default this reads local summaries only. Pass download=True to warm the
    optional cache from completed games. Unknown identity or minutes stay missing.
    """
    cache = Path(cache)
    root = cache / "player-workload-v1"
    if other_fixtures is None or other_fixtures.empty or (not download and not root.is_dir()):
        return pd.DataFrame(columns=COLUMNS)

    schedule = other_fixtures.dropna(subset=["kickoff"]).copy()
    schedule["season"] = schedule["kickoff"].dt.year - (schedule["kickoff"].dt.month < 7)
    schedule = schedule.drop_duplicates(["competition", "event_id", "team_code"])
    aliases = {season: _season_lookup(cache, live, season) for season in schedule["season"].unique()}
    fixtures = list(schedule.groupby(["competition", "event_id"], sort=False))

    def fetch(match):
        (competition, event_id), fixture = match
        event_id = str(event_id)
        if competition not in COMPETITIONS or not event_id.isascii() or not event_id.isdecimal():
            return []
        first = fixture.iloc[0]
        path = root / competition / f"{event_id}.json.gz"
        clubs = set(fixture["team_code"])
        lookup = aliases[first["season"]]
        try:
            if path.is_file():
                rows = json.loads(gzip.decompress(path.read_bytes()))
            elif download:
                body = _download(f"{API}/{competition}/summary?event={event_id}")
                rows = _event_workload(json.loads(body))
                _write(path, gzip.compress(json.dumps(rows, separators=(",", ":")).encode(), mtime=0))
            else:
                return []
        except (DataError, ValueError, TypeError, OSError) as error:
            LOG.debug("ESPN player workload unavailable for %s/%s: %s", competition, event_id, error)
            return []

        result = []
        for row in rows:
            if row["team_code"] not in clubs:
                continue
            codes = {lookup[name] for name in row["names"] if name in lookup}
            if len(codes) != 1:
                continue
            result.append({
                "event_id": event_id, "competition": competition, "player": codes.pop(),
                "kickoff": first["kickoff"], "observed_at": first["kickoff"] + RESULT_LAG,
                "team_code": row["team_code"], **{field: row[field] for field in FEATURES},
            })
        return result

    with ThreadPoolExecutor(max_workers=12 if download else 4) as pool:
        records = [row for rows in pool.map(fetch, fixtures) for row in rows]
    frame = pd.DataFrame(records, columns=COLUMNS)
    return frame.drop_duplicates(["competition", "event_id", "player"], keep=False)


def workload_features(workload, players, cutoff):
    if workload is None or workload.empty:
        return pd.DataFrame(index=players.index, columns=list(FEATURES.values()), dtype=float)
    recent = workload.loc[
        (workload["kickoff"] >= cutoff - RECENT_WINDOW) & (workload["observed_at"] <= cutoff)
    ]
    grouped = recent.groupby("player")
    totals = grouped[list(FEATURES)].sum(min_count=1)
    totals["minutes"] = totals["minutes"].where(grouped["minutes"].count().eq(grouped.size()))
    return pd.DataFrame({feature: players.map(totals[field]) for field, feature in FEATURES.items()})
