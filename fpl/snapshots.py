"""Point-in-time FPL availability and set-piece duties from bootstrap captures."""

from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import gzip
import io
import json
import logging
import lzma
from pathlib import Path
import re

import pandas as pd

from .data import DataError, EndpointNotFound, POSITIONS, _download, _write, player_identity

LOG = logging.getLogger(__name__)
ARCHIVE = "Randdalf/fplcache"
ARCHIVE_TREE = f"https://api.github.com/repos/{ARCHIVE}/git/trees/main?recursive=1"
ARCHIVE_RAW = f"https://raw.githubusercontent.com/{ARCHIVE}/main"
ARCHIVE_PATH = re.compile(r"cache/(\d{4})/(\d{1,2})/(\d{1,2})/(\d{2})(\d{2})\.json\.xz")
ARCHIVE_FIRST_SEASON = 2020
SNAPSHOT_MAX_AGE = timedelta(hours=24)
ROLE_FIELDS = {
    "penalty_order": "penalties_order",
    "direct_fk_order": "direct_freekicks_order",
    "corner_fk_order": "corners_and_indirect_freekicks_order",
}
AVAILABILITY_FIELDS = ("flag_probability", "flag_age_days")
CONTEXT_KEYS = ("season", "gameweek", "player")
CONTEXT_DATES = ("context_asof", "context_deadline")
CONTEXT_COLUMNS = (
    *CONTEXT_KEYS, *CONTEXT_DATES, *AVAILABILITY_FIELDS, *ROLE_FIELDS,
    "context_team", "context_position", "provider_ep_next",
)
FLAG_FALLBACKS = {"d": 0.75, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}
MINUTES_FIELDS = ("expected_minutes", "predicted_start_probability")
MINUTES_COLUMNS = (*CONTEXT_KEYS, "context_asof", *MINUTES_FIELDS)


def _utc(value):
    return pd.to_datetime(value, utc=True, errors="coerce", format="mixed")


def _archive_index():
    """Fetch the paths once and use the UTC filenames written by the capture job."""
    tree = json.loads(_download(ARCHIVE_TREE))
    if tree.get("truncated") or not tree.get("tree"):
        raise DataError("The historical FPL bootstrap index is incomplete.")
    references = []
    for entry in tree["tree"]:
        path = entry["path"]
        if match := ARCHIVE_PATH.fullmatch(path):
            stamp = datetime(*map(int, match.groups()), tzinfo=timezone.utc)
            references.append((stamp, path))
    return sorted(references)


def _capture(cache, path):
    stored = cache / "raw" / Path(path).relative_to("cache")
    if stored.is_file():
        payload = stored.read_bytes()
    else:
        payload = _download(f"{ARCHIVE_RAW}/{path}")
    data = json.loads(lzma.decompress(payload))
    if not stored.is_file():
        _write(stored, payload)
    return data


def _events(bootstrap, season):
    events = {int(event["id"]): event for event in bootstrap.get("events", ())}
    first = _utc(events.get(1, {}).get("deadline_time"))
    if first is None or pd.isna(first) or first.year - (first.month < 7) != season:
        return {}
    return events


def _accept_capture(bootstrap, season, gameweek, deadline, stamp, *, historical):
    """Require an explicitly matching official deadline before reading any player fields."""
    if pd.isna(deadline) or pd.isna(stamp) or stamp >= deadline:
        return False
    event = _events(bootstrap, season).get(gameweek, {})
    if event.get("finished") or _utc(event.get("deadline_time")) != deadline:
        return False
    return not historical or bool(event.get("is_next"))


def _float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def _player_rows(bootstrap, season, week, deadline, stamp):
    rows = []
    for element in bootstrap["elements"]:
        position = POSITIONS.get(element.get("element_type"))
        if position is None:
            continue
        chance = _float(element.get("chance_of_playing_next_round"))
        if pd.isna(chance):
            chance = FLAG_FALLBACKS.get(element.get("status"), 1.0)
        else:
            chance = min(1.0, max(0.0, chance / 100))
        added = _utc(element.get("news_added")) if element.get("news") else None
        age = (stamp - added).total_seconds() / 86400 if pd.notna(added) and added <= stamp else float("nan")
        roles = {
            key: _float((element[field] or 0) if field in element else None)
            for key, field in ROLE_FIELDS.items()
        }
        rows.append({
            "season": season, "gameweek": week,
            "player": player_identity(element.get("code"), season, element["id"]),
            "context_asof": stamp, "context_deadline": deadline,
            "flag_probability": chance, "flag_age_days": age, **roles,
            "context_team": element.get("team"), "context_position": position,
            "provider_ep_next": _float(element.get("ep_next")),
        })
    return rows


def _season_deadlines(cache, references, dates, season, matches):
    """Read the final season calendar after every match has been recorded."""
    last_kickoff = _utc(matches["kickoff"]).max()
    if pd.isna(last_kickoff):
        return {}
    start = bisect_left(dates, last_kickoff + timedelta(hours=4))
    end = bisect_left(dates, last_kickoff + timedelta(days=3))
    wanted = set(matches["gameweek"].dropna().astype(int))
    for stamp, path in reversed(references[start:end]):
        try:
            bootstrap = _capture(cache, path)
        except EndpointNotFound:
            continue
        events = _events(bootstrap, season)
        if wanted and wanted.issubset(events) and all(events[week]["finished"] for week in wanted):
            deadlines = {week: _utc(events[week]["deadline_time"]) for week in wanted}
            if all(pd.notna(deadline) and deadline < stamp for deadline in deadlines.values()):
                return deadlines
    return {}


def _one_round(cache, references, dates, season, week, deadline):
    start = bisect_left(dates, deadline - SNAPSHOT_MAX_AGE)
    end = bisect_left(dates, deadline)
    for stamp, path in reversed(references[start:end]):
        # Paths have minute precision. Treat the last second of that minute as
        # the observation bound so a rounding coincidence cannot cross a deadline.
        bound = pd.Timestamp(stamp + timedelta(seconds=59, microseconds=999999))
        if bound >= deadline:
            continue
        try:
            bootstrap = _capture(cache, path)
        except EndpointNotFound:
            continue
        if _accept_capture(bootstrap, season, week, deadline, bound, historical=True):
            return _player_rows(bootstrap, season, week, deadline, bound)
    return []


def _read_context(path):
    frame = pd.read_csv(path, compression="gzip")
    for field in CONTEXT_DATES:
        frame[field] = _utc(frame[field])
    return frame.reindex(columns=CONTEXT_COLUMNS)


def _write_context(path, rows):
    buffer = io.StringIO()
    pd.DataFrame(rows, columns=CONTEXT_COLUMNS).to_csv(buffer, index=False)
    _write(path, gzip.compress(buffer.getvalue().encode("utf-8"), mtime=0))


def load_player_context(cache, fixtures, live):
    """Return one verified pre-deadline row per player and round where available.

    Finished seasons have a cache partition per season. Earlier rounds of the
    current season use per-round partitions so refreshing live data is incremental.
    A missing provider leaves those rows missing and preserves the core forecast.
    """
    cache = Path(cache) / "player-context-v1"
    frames, archived_seasons, live_rounds = [], {}, {}
    seasons = fixtures.loc[fixtures["season"].between(ARCHIVE_FIRST_SEASON, live.season - 1)]
    for season, matches in seasons.groupby("season", sort=True):
        season = int(season)
        path = cache / f"{season}.csv.gz"
        if path.is_file():
            frames.append(_read_context(path))
        else:
            archived_seasons[season] = (matches, path)

    events = _events(live.bootstrap, live.season)
    for week, event in events.items():
        if week >= live.gameweek or not event.get("finished"):
            continue
        path = cache / str(live.season) / f"{week}.csv.gz"
        if path.is_file():
            frames.append(_read_context(path))
        else:
            live_rounds[week] = (_utc(event["deadline_time"]), path)

    references = []
    if archived_seasons or live_rounds:
        try:
            references = _archive_index()
        except (DataError, ValueError, KeyError) as error:
            LOG.warning("Historical FPL availability index unavailable: %s", error)
    if references:
        dates = [stamp for stamp, _ in references]

        def collect(season, week, deadline):
            return _one_round(cache, references, dates, season, week, deadline)

        with ThreadPoolExecutor(max_workers=12) as pool:
            for season, (matches, path) in archived_seasons.items():
                try:
                    deadlines = _season_deadlines(cache, references, dates, season, matches)
                    if not deadlines:
                        continue
                    jobs = [pool.submit(collect, season, week, deadline) for week, deadline in sorted(deadlines.items())]
                    rows = [row for job in jobs for row in job.result()]
                    _write_context(path, rows)
                    frames.append(_read_context(path))
                except (DataError, OSError, ValueError, KeyError, lzma.LZMAError) as error:
                    LOG.warning("Historical FPL availability unavailable for %s: %s", season, error)
            jobs = {
                week: pool.submit(collect, live.season, week, deadline)
                for week, (deadline, _) in live_rounds.items() if pd.notna(deadline)
            }
            for week, job in jobs.items():
                try:
                    rows = job.result()
                    if rows:
                        path = live_rounds[week][1]
                        _write_context(path, rows)
                        frames.append(_read_context(path))
                except (DataError, OSError, ValueError, KeyError, lzma.LZMAError) as error:
                    LOG.warning("Current-season FPL availability unavailable for Gameweek %s: %s", week, error)

    deadline, stamp = _utc(live.deadline), _utc(live.refreshed_at)
    if _accept_capture(live.bootstrap, live.season, live.gameweek, deadline, stamp, historical=False):
        frames.append(pd.DataFrame.from_records(
            _player_rows(live.bootstrap, live.season, live.gameweek, deadline, stamp), columns=CONTEXT_COLUMNS,
        ))
    if not frames:
        return pd.DataFrame(columns=CONTEXT_COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(list(CONTEXT_KEYS), keep="last")


def load_minutes_context(path, fixtures):
    """Read independent pre-match minutes or start probabilities from a CSV.

    The input uses stable FPL player codes and a timestamp with an explicit UTC
    offset. Keep the latest row strictly before the matching Gameweek deadline.
    Expected minutes refer to one fixture and range from 0 to 90.
    """
    if path is None:
        return pd.DataFrame({
            **{column: pd.Series(dtype="int64") for column in CONTEXT_KEYS},
            "context_asof": pd.Series(dtype="datetime64[ns, UTC]"),
            **{column: pd.Series(dtype="float64") for column in MINUTES_FIELDS},
        })

    frame = pd.read_csv(path)
    required = {*CONTEXT_KEYS, "context_asof"}
    if not required.issubset(frame.columns) or not set(MINUTES_FIELDS).intersection(frame.columns):
        raise ValueError("Minutes CSV needs season, gameweek, player, context_asof and at least one prediction column.")
    frame = frame.reindex(columns=MINUTES_COLUMNS)
    for column in CONTEXT_KEYS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not (values.gt(0) & values.mod(1).eq(0)).all():
            raise ValueError(f"Minutes CSV {column} must contain positive whole numbers.")
        frame[column] = values.astype("int64")

    timestamps = frame["context_asof"].astype("string").str.strip()
    frame["context_asof"] = _utc(timestamps)
    with_offset = timestamps.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False)
    if frame["context_asof"].isna().any() or not with_offset.all():
        raise ValueError("Minutes CSV context_asof must be an ISO timestamp with an explicit UTC offset.")

    for column, limit in zip(MINUTES_FIELDS, (90, 1)):
        source = frame[column]
        values = pd.to_numeric(source, errors="coerce")
        if (source.notna() & ~values.between(0, limit)).any():
            raise ValueError(f"Minutes CSV {column} must be between 0 and {limit}.")
        frame[column] = values
    frame = frame.dropna(subset=list(MINUTES_FIELDS), how="all")

    round_keys = ["season", "gameweek"]
    schedule = fixtures[round_keys + ["deadline"]].copy()
    schedule["deadline"] = _utc(schedule["deadline"])
    deadlines = schedule.groupby(round_keys, as_index=False)["deadline"].min()
    frame = frame.merge(deadlines, on=round_keys, how="inner", validate="many_to_one")
    timely = frame.loc[frame["context_asof"] < frame["deadline"], list(MINUTES_COLUMNS)]
    return timely.sort_values("context_asof", kind="stable").drop_duplicates(list(CONTEXT_KEYS), keep="last")
