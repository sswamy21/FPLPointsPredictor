"""Download, cache, and normalize archived and current FPL match data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock, local
from time import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ARCHIVE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
API_URL = "https://fantasy.premierleague.com/api"
ARCHIVE_START = 2016
LIVE_MAX_AGE_SECONDS = 900
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
OPTIONAL_STATS = ("starts", "expected_goals", "expected_assists", "expected_goals_conceded")
STATS = ("points", "minutes", "goals_scored", "assists", "bonus", "bps", *OPTIONAL_STATS)
PLAYER_COLUMNS = [
    "season", "gameweek", "fixture", "kickoff", "deadline", "player", "element",
    "position", "team", "opponent", "home", *STATS,
]
FIXTURE_COLUMNS = [
    "season", "gameweek", "fixture", "kickoff", "deadline", "home_team", "away_team",
    "home_team_code", "away_team_code", "home_goals", "away_goals", "finished",
    "home_difficulty", "away_difficulty",
]
FIXTURE_NAMES = {
    "id": "fixture", "event": "gameweek", "kickoff_time": "kickoff",
    "team_h": "home_team", "team_a": "away_team",
    "team_h_score": "home_goals", "team_a_score": "away_goals",
    "team_h_difficulty": "home_difficulty", "team_a_difficulty": "away_difficulty",
    "deadline_time": "deadline",
}
PLAYER_NAMES = {
    "round": "gameweek", "kickoff_time": "kickoff", "opponent_team": "opponent",
    "was_home": "home", "total_points": "points",
}
HTTP_STATE = local()


class DataError(RuntimeError):
    """A required archive or official API response is unavailable."""


class PublicPicksUnavailable(DataError):
    """FPL has no accessible published squad for this manager."""


class EndpointNotFound(DataError):
    """An official endpoint returned HTTP 404."""


class EndpointForbidden(DataError):
    """An official endpoint returned HTTP 403."""


@dataclass(frozen=True)
class LiveData:
    season: int
    bootstrap: dict
    history: pd.DataFrame
    fixtures: pd.DataFrame
    gameweek: int
    deadline: pd.Timestamp
    refreshed_at: datetime
    warning: str | None = None


def season_label(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def player_identity(code: int | None, season: int, element: int) -> int:
    """Official player code spans seasons. A missing code gets a season-local key."""
    return int(code) if pd.notna(code) and int(code) > 0 else -(season * 10000 + element)


def _session() -> requests.Session:
    if not hasattr(HTTP_STATE, "session"):
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; FPL Squad Optimizer/1.0)",
            "Accept": "*/*",
        })
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        session.mount("https://", HTTPAdapter(max_retries=retry))
        HTTP_STATE.session = session
    return HTTP_STATE.session


def _download(url: str) -> bytes:
    try:
        response = _session().get(url, timeout=(5, 20))
        if response.status_code == 404:
            raise EndpointNotFound(f"No data at {url}.")
        if response.status_code == 403:
            raise EndpointForbidden(f"Access to {url} was denied.")
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        raise DataError(f"Could not download {url}: {exc}") from exc


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        file.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json(payload: bytes, source: str):
    try:
        return json.loads(payload)
    except (UnicodeError, ValueError) as exc:
        raise DataError(f"Invalid JSON from {source}.") from exc


def _normalize_fixtures(
    raw: pd.DataFrame,
    season: int,
    team_codes: dict[int, int],
    *,
    deadlines: dict[int, str] | None = None,
) -> pd.DataFrame:
    fixtures = raw.rename(columns=FIXTURE_NAMES).copy()
    fixtures["season"] = season
    fixtures["gameweek"] = pd.to_numeric(fixtures["gameweek"], errors="coerce").astype("Int64")
    fixtures["kickoff"] = pd.to_datetime(fixtures["kickoff"], utc=True, errors="coerce")
    fixtures["finished"] = fixtures["finished"].isin([True, "True", "true", 1])
    for side in ("home", "away"):
        fixtures[f"{side}_team_code"] = fixtures[f"{side}_team"].map(team_codes).astype("Int64")
    if deadlines is not None:
        fixtures["deadline"] = pd.to_datetime(fixtures["gameweek"].map(deadlines), utc=True)
    else:
        # Archived FDR is the final snapshot of the season. Use only archived results
        # as targets and for lagged form, then derive fixture difficulty as of each week.
        fixtures["home_difficulty"] = float("nan")
        fixtures["away_difficulty"] = float("nan")
        offset = timedelta(minutes=90 if season >= 2020 else 60)
        first_kickoff = fixtures.groupby("gameweek")["kickoff"].transform("min")
        inferred = first_kickoff - offset
        recorded = pd.to_datetime(fixtures.get("deadline", inferred), utc=True, errors="coerce")
        fixtures["deadline"] = recorded.fillna(inferred)
    return fixtures[FIXTURE_COLUMNS].sort_values(["kickoff", "fixture"]).reset_index(drop=True)


def _normalize_players(
    performances: pd.DataFrame,
    roster: pd.DataFrame,
    fixtures: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    columns = ["fixture", "element", "round", "was_home", "opponent_team", "total_points"]
    columns.extend(stat for stat in STATS if stat != "points")
    players = performances.reindex(columns=columns).rename(columns=PLAYER_NAMES)
    players["home"] = players["home"].isin([True, "True", "true", 1])
    players = players.merge(
        fixtures[["fixture", "gameweek", "kickoff", "deadline", "home_team", "away_team", "finished"]],
        on=["fixture", "gameweek"],
        validate="many_to_one",
    )
    players = players.loc[players["finished"]].copy()
    players["team"] = players["home_team"].where(players["home"], players["away_team"])
    # The club from the match venue and opponent is valid even when a player transferred
    # between two Premier League teams during the season.
    opponent = players["away_team"].where(players["home"], players["home_team"])
    if not players["opponent"].eq(opponent).all():
        raise DataError(f"Inconsistent fixture opposition in {season_label(season)}.")
    players = players.merge(
        roster.rename(columns={"id": "element"})[["element", "code", "element_type"]],
        on="element",
        how="left",
        validate="many_to_one",
    )
    players["position"] = players["element_type"].map(POSITIONS)
    players = players.loc[players["position"].notna()].copy()
    players["player"] = [
        player_identity(code, season, element)
        for code, element in zip(players["code"], players["element"])
    ]
    for stat in STATS:
        players[stat] = pd.to_numeric(players[stat], errors="coerce")
    if season == 2022:
        # The archive backfills these early-week fields with zeroes before collection began.
        players.loc[players["gameweek"] < 16, list(OPTIONAL_STATS)] = float("nan")
    players["season"] = season
    return (
        players[PLAYER_COLUMNS]
        .drop_duplicates(["element", "fixture"], keep="last")
        .sort_values(["kickoff", "fixture", "element"])
        .reset_index(drop=True)
    )


def _reconstruct_fixtures(performances: pd.DataFrame) -> pd.DataFrame:
    # In the two earliest seasons the repository lacks a standalone fixture table. For
    # each match, the away players give the home team ID and vice versa.
    performances = performances.dropna(subset=["team_h_score", "team_a_score"])
    is_home = performances["was_home"].isin([True, "True", "true", 1])
    away_team = performances.loc[is_home].groupby("fixture")["opponent_team"].first()
    home_team = performances.loc[~is_home].groupby("fixture")["opponent_team"].first()
    fixtures = (
        performances.drop_duplicates("fixture")[[
            "fixture", "round", "kickoff_time", "team_h_score", "team_a_score",
        ]]
        .set_index("fixture")
        .join([home_team.rename("team_h"), away_team.rename("team_a")])
        .reset_index()
        .rename(columns={"round": "gameweek"})
    )
    fixtures["finished"] = True
    return fixtures


def _live_target(bootstrap: dict, now: datetime) -> tuple[int, int, pd.Timestamp]:
    events = sorted(bootstrap.get("events", []), key=lambda event: event["id"])
    if not events:
        raise DataError("FPL did not publish a Gameweek schedule.")
    first_deadline = pd.Timestamp(events[0]["deadline_time"])
    season = first_deadline.year - int(first_deadline.month < 7)
    for event in events:
        deadline = pd.Timestamp(event["deadline_time"])
        if deadline > now and not event.get("finished"):
            return season, event["id"], deadline
    raise DataError("The official FPL schedule has no upcoming Gameweek deadline.")


class DataStore:
    """Raw CSVs are immutable. Official endpoints have a separate refreshable cache."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self._archive_lock = RLock()
        self._live_lock = RLock()
        self._archive: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._live: LiveData | None = None

    def _archive_csv(self, season: int, source: str, columns: set[str] | None = None) -> pd.DataFrame:
        label = season_label(season)
        path = self.cache_dir / "historical" / label / Path(source).name
        if not path.is_file():
            _write(path, _download(f"{ARCHIVE_URL}/{label}/{source}"))
        options = {"usecols": lambda name: columns is None or name in columns}
        try:
            return pd.read_csv(path, encoding="utf-8-sig", **options)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1", **options)

    def _historical_season(self, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        roster = self._archive_csv(season, "players_raw.csv", {
            "id", "code", "element_type", "team", "team_code",
        })
        performances = self._archive_csv(season, "gws/merged_gw.csv", {
            "fixture", "element", "round", "was_home", "opponent_team", "total_points",
            "kickoff_time", "team_h_score", "team_a_score", *STATS,
        })
        if season >= 2018:
            raw_fixtures = self._archive_csv(season, "fixtures.csv", {*FIXTURE_NAMES, "finished"})
        else:
            raw_fixtures = _reconstruct_fixtures(performances)
        team_codes = roster.drop_duplicates("team").set_index("team")["team_code"].to_dict()
        fixtures = _normalize_fixtures(raw_fixtures, season, team_codes)
        history = _normalize_players(performances, roster, fixtures, season)
        if len(fixtures) != 380 or not fixtures["finished"].all() or history.empty:
            raise DataError(f"The {season_label(season)} archive is incomplete.")
        return history, fixtures

    def history(self, through: int = 2025) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load completed seasons through the inclusive starting year (default 2025/26)."""
        years = range(ARCHIVE_START, through + 1)
        if not years:
            raise ValueError("The FPL archive starts in 2016/17.")
        with self._archive_lock:
            missing = [season for season in years if season not in self._archive]
            if missing:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    loaded = executor.map(self._historical_season, missing)
                    self._archive.update(zip(missing, loaded))
            player_tables, fixture_tables = zip(*(self._archive[season] for season in years))
        return pd.concat(player_tables, ignore_index=True), pd.concat(fixture_tables, ignore_index=True)

    def _api_json(self, endpoint: str):
        url = f"{API_URL}/{endpoint}/"
        return _json(_download(url), url)

    def _cached_api(self, endpoint: str, path: Path, *, refresh: bool, cache_only: bool):
        if path.is_file() and (cache_only or not refresh):
            try:
                return _json(path.read_bytes(), str(path))
            except DataError:
                if cache_only:
                    raise
        if cache_only:
            raise DataError(f"No cached copy of {endpoint}.")
        data = self._api_json(endpoint)
        _write(path, json.dumps(data, separators=(",", ":")).encode("utf-8"))
        return data

    def _fetch_live(self, now: datetime, *, refresh: bool, cache_only: bool = False) -> LiveData:
        base = self.cache_dir / "live"
        bootstrap_path = base / "bootstrap-static.json"
        fixtures_path = base / "fixtures.json"
        missing_or_old = any(
            not path.is_file() or time() - path.stat().st_mtime >= LIVE_MAX_AGE_SECONDS
            for path in (bootstrap_path, fixtures_path)
        )
        refresh_base = refresh or missing_or_old
        bootstrap = self._cached_api("bootstrap-static", bootstrap_path, refresh=refresh_base, cache_only=cache_only)
        raw_fixtures = self._cached_api("fixtures", fixtures_path, refresh=refresh_base, cache_only=cache_only)
        season, week, deadline = _live_target(bootstrap, now)
        roster = [element for element in bootstrap["elements"] if element["element_type"] in POSITIONS]
        season_cache = base / str(season)
        revision_path = season_cache / "summary-revision.txt"
        # A new completed fixture can change form even while the official current GW
        # stays the same. Reuse unchanged summaries and still query any new elements.
        results = [
            (match["id"], match.get("team_h_score"), match.get("team_a_score"))
            for match in raw_fixtures if match["finished"]
        ]
        checked = [event["id"] for event in bootstrap["events"] if event.get("data_checked")]
        revision = sha256(repr((sorted(results), sorted(checked))).encode("ascii")).hexdigest()
        if revision_path.is_file():
            has_new_results = revision_path.read_text(encoding="ascii") != revision
        else:
            has_new_results = refresh_base

        def summary(element: dict) -> list:
            element_id = element["id"]
            path = season_cache / "element-summary" / f"{element_id}.json"
            try:
                data = self._cached_api(
                    f"element-summary/{element_id}", path,
                    refresh=refresh or has_new_results, cache_only=cache_only,
                )
            except EndpointNotFound:
                if not element.get("removed"):
                    raise
                return []
            return data["history"]

        with ThreadPoolExecutor(max_workers=10) as executor:
            player_results = [match for history in executor.map(summary, roster) for match in history]
        if not cache_only:
            _write(revision_path, revision.encode("ascii"))

        team_codes = {team["id"]: team["code"] for team in bootstrap["teams"]}
        deadlines = {event["id"]: event["deadline_time"] for event in bootstrap["events"]}
        fixtures = _normalize_fixtures(pd.DataFrame(raw_fixtures), season, team_codes, deadlines=deadlines)
        history = (
            _normalize_players(pd.DataFrame(player_results), pd.DataFrame(roster), fixtures, season)
            if player_results else pd.DataFrame(columns=PLAYER_COLUMNS)
        )
        refreshed_at = datetime.fromtimestamp(
            min(bootstrap_path.stat().st_mtime, fixtures_path.stat().st_mtime), timezone.utc,
        )
        return LiveData(season, bootstrap, history, fixtures, week, deadline, refreshed_at)

    def live(self, refresh: bool = False) -> LiveData:
        with self._live_lock:
            now = datetime.now(timezone.utc)
            cached = self._live
            if (
                cached is not None and not refresh and cached.warning is None
                and now < cached.deadline
                and (now - cached.refreshed_at).total_seconds() < LIVE_MAX_AGE_SECONDS
            ):
                return cached
            try:
                current = self._fetch_live(now, refresh=refresh)
            except DataError as error:
                if cached is None or now >= cached.deadline:
                    try:
                        cached = self._fetch_live(now, refresh=False, cache_only=True)
                    except DataError:
                        raise error from None
                current = replace(
                    cached, warning=f"Official FPL refresh failed. Showing cached data. {error}",
                )
            self._live = current
            return current

    def public_picks(self, manager_id: int, live: LiveData) -> tuple[list[int], int]:
        """Return the latest published permanent squad for the upcoming Gameweek."""
        if isinstance(manager_id, bool) or not isinstance(manager_id, int) or manager_id <= 0:
            raise ValueError("Enter a positive numeric FPL Manager ID.")
        now = datetime.now(timezone.utc)
        public_weeks = sorted(
            (
                event["id"] for event in live.bootstrap["events"]
                if pd.Timestamp(event["deadline_time"]) <= now
            ),
            reverse=True,
        )
        if not public_weeks:
            raise PublicPicksUnavailable("FPL publishes manager picks after the first Gameweek deadline.")
        try:
            entry = self._api_json(f"entry/{manager_id}")
            first_week = entry.get("started_event") or 1
            for week in public_weeks:
                if week < first_week:
                    break
                try:
                    data = self._api_json(f"entry/{manager_id}/event/{week}/picks")
                except EndpointNotFound:
                    continue
                picks = data.get("picks", [])
                # The permanent squad returns when a one-week Free Hit expires.
                if len(picks) == 15 and data.get("active_chip") not in {"freehit", "free_hit"}:
                    return [int(pick["element"]) for pick in picks], week
        except EndpointNotFound as exc:
            raise PublicPicksUnavailable("No public FPL entry matched that Manager ID.") from exc
        except EndpointForbidden as exc:
            raise PublicPicksUnavailable("FPL has restricted access to this manager's picks.") from exc
        raise PublicPicksUnavailable("FPL has no published 15-player squad for that Manager ID.")
