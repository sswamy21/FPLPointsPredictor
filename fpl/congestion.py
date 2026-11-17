"""Pre-deadline club workload from completed domestic and European cup matches."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from time import time

import pandas as pd

from .data import DataError, _download, _write

API = "https://site.api.espn.com/apis/site/v2/sports/soccer"
COMPETITIONS = ("eng.fa", "eng.league_cup", "uefa.champions", "uefa.europa", "uefa.europa.conf")
FIRST_YEAR = 2022
MAX_AGE_SECONDS = 6 * 60 * 60
CLUBS = {
    "Arsenal": 3, "Aston Villa": 7, "AFC Bournemouth": 91, "Brentford": 94,
    "Brighton & Hove Albion": 36, "Burnley": 90, "Chelsea": 8, "Coventry City": 9,
    "Crystal Palace": 31, "Everton": 11, "Fulham": 54, "Hull City": 88,
    "Ipswich Town": 40, "Leeds United": 2, "Leicester City": 13, "Liverpool": 14,
    "Luton Town": 102, "Manchester City": 43, "Manchester United": 1,
    "Newcastle United": 4, "Norwich City": 45, "Nottingham Forest": 17,
    "Sheffield United": 49, "Southampton": 20, "Sunderland": 56, "Tottenham Hotspur": 6,
    "Watford": 57, "West Ham United": 21, "Wolverhampton Wanderers": 39,
}


def _completed_events(payload):
    events = payload["events"]
    if len(events) >= 500:
        raise DataError("ESPN returned a truncated calendar.")
    results = []
    for event in events:
        if not event["status"]["type"]["completed"]:
            continue
        for participant in event["competitions"][0]["competitors"]:
            code = CLUBS.get(participant["team"]["displayName"])
            if code:
                results.append({"id": event["id"], "kickoff": event["date"], "team_code": code})
    return results


def load_other_fixtures(cache, *, now=None, refresh=False):
    """Cache complete ESPN calendars and return known club appearances in them."""
    now = now or datetime.now(timezone.utc)
    root = Path(cache) / "congestion"

    def load(request):
        year, competition = request
        path = root / f"{year}-{competition}.json"
        try:
            if path.is_file() and (year < now.year or (
                not refresh and time() - path.stat().st_mtime < MAX_AGE_SECONDS
            )):
                return year, competition, json.loads(path.read_text(encoding="utf-8")), None
            url = f"{API}/{competition}/scoreboard?dates={year}&limit=500"
            events = _completed_events(json.loads(_download(url)))
            _write(path, json.dumps(events, separators=(",", ":")).encode())
            return year, competition, events, None
        except (DataError, KeyError, TypeError, ValueError, OSError) as error:
            return year, competition, [], str(error)

    requests = [(year, competition) for year in range(FIRST_YEAR, now.year + 1) for competition in COMPETITIONS]
    with ThreadPoolExecutor(max_workers=5) as pool:
        calendars = list(pool.map(load, requests))
    missing = {year for year, _, _, failure in calendars if failure is not None}
    valid = {year for year, _, _, _ in calendars} - missing
    records = [
        {"event_id": row["id"], "kickoff": row["kickoff"], "team_code": row["team_code"], "competition": competition}
        for year, competition, events, _ in calendars if year in valid for row in events
    ]
    matches = pd.DataFrame(records, columns=["event_id", "kickoff", "team_code", "competition"])
    matches = matches.drop_duplicates(["competition", "event_id", "team_code"])
    matches["kickoff"] = pd.to_datetime(matches["kickoff"], utc=True, errors="coerce")
    matches = matches.loc[matches["kickoff"] < now].sort_values(["kickoff", "team_code"]).reset_index(drop=True)
    matches.attrs.update(
        covered_years=tuple(sorted(valid)), club_codes=tuple(sorted(CLUBS.values())),
        unavailable_calendars=tuple(f"{year} {competition}" for year, competition, _, error in calendars if error),
    )
    return matches
