"""Read public EPL market prices and prepare locally cached fixture inputs."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import logging
from math import exp
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from .data import DataError, DataStore, EndpointNotFound, _download, _write
from .markets import FIXTURE_KEY, MARKET_COLUMNS, implied_goals, select_market_context


LOGGER = logging.getLogger(__name__)
GAMMA = "https://gamma-api.polymarket.com/events/slug"
PRICE_HISTORY = "https://clob.polymarket.com/prices-history"
# FPL uses these stable club codes, including across promotion and relegation.
CLUB_SLUGS = {
    1: "mun", 2: "lee", 3: "ars", 4: "new", 6: "tot", 7: "ast", 8: "che", 9: "cov",
    11: "eve", 13: "lei", 14: "liv", 17: "not", 20: "sou", 21: "wes", 31: "cry", 36: "bri",
    39: "wol", 40: "ips", 43: "mac", 54: "ful", 56: "sun", 88: "hul", 90: "bur", 91: "bou", 94: "bre",
}


def _array(value):
    return json.loads(value) if isinstance(value, str) else value or []


def _markets(slug, kickoff, home, away):
    match = json.loads(_download(f"{GAMMA}/{slug}"))
    if pd.to_datetime(match.get("startTime"), utc=True, errors="coerce") != kickoff:
        return None
    moneyline = {market["slug"]: market for market in match["markets"]}
    if not all(f"{slug}-{key}" in moneyline for key in (home, "draw", away)):
        return None
    contracts = [(moneyline[f"{slug}-{key}"], "Yes") for key in (home, "draw", away)]
    try:
        more = json.loads(_download(f"{GAMMA}/{slug}-more-markets"))
    except EndpointNotFound:
        return 2.5, contracts
    if pd.to_datetime(more.get("startTime"), utc=True, errors="coerce") != kickoff:
        return 2.5, contracts
    totals = [
        (float(market["line"]), market)
        for market in more["markets"]
        if market.get("sportsMarketType") == "totals" and market.get("line") in (2.5, 3.5, 1.5)
    ]
    if not totals:
        return 2.5, contracts
    line, total = min(totals, key=lambda item: abs(item[0] - 2.5))
    return line, contracts + [(total, "Over")]


def _contract_index(market, outcome):
    return _array(market["outcomes"]).index(outcome)


def _historical_price(market, outcome, deadline):
    token = _array(market["clobTokenIds"])[_contract_index(market, outcome)]
    # Leave a full aggregation bin before the deadline in case the provider labels
    # hourly values with the start of their time bucket.
    end = int((deadline - timedelta(hours=1)).timestamp())
    start = int((deadline - timedelta(days=2)).timestamp())
    params = urlencode({"market": token, "startTs": start, "endTs": end, "fidelity": 60})
    page = json.loads(_download(f"{PRICE_HISTORY}?{params}"))
    prices = [(point["t"], float(point["p"])) for point in page["history"] if start <= point["t"] < end]
    return max(prices, default=None)


def _live_price(market, outcome):
    if not market.get("acceptingOrders") or market.get("closed"):
        return None
    prices = _array(market.get("outcomePrices"))
    return float(prices[_contract_index(market, outcome)]) if prices else None


def _fixture_market(fixture, cache, now):
    home = CLUB_SLUGS.get(fixture.home_team_code)
    away = CLUB_SLUGS.get(fixture.away_team_code)
    deadline, kickoff = fixture.deadline, fixture.kickoff
    if not home or not away or pd.isna(deadline) or pd.isna(kickoff) or deadline >= kickoff:
        return None
    slug = f"epl-{home}-{away}-{kickoff.tz_convert('Europe/London').date()}"
    historical = deadline <= now
    history_cache = cache / str(fixture.season) / f"{fixture.fixture}-{int(deadline.timestamp())}.json"
    if historical and history_cache.is_file():
        cached = json.loads(history_cache.read_bytes())
        if cached is not None:
            return cached
    try:
        market_group = _markets(slug, kickoff, home, away)
        result = None
        if market_group is not None:
            line, contracts = market_group
            if historical:
                quotes = [_historical_price(market, outcome, deadline) for market, outcome in contracts]
                if any(quote is None for quote in quotes[:3]):
                    return None
                observed = pd.to_datetime(max(quote[0] for quote in quotes if quote is not None), unit="s", utc=True)
                values = [quote[1] if quote is not None else None for quote in quotes]
            else:
                values = [_live_price(market, outcome) for market, outcome in contracts]
                observed = pd.Timestamp.now(tz="UTC")
            if observed < deadline and all(value is not None and 0 < value < 1 for value in values[:3]):
                over = values[3] if len(values) > 3 and values[3] is not None and 0 < values[3] < 1 else None
                home_rate, away_rate = implied_goals(*values[:3], over=over, line=line)
                result = {
                    "season": int(fixture.season), "fixture": int(fixture.fixture),
                    "as_of": observed.isoformat(), "home_expected_goals": home_rate,
                    "away_expected_goals": away_rate, "home_clean_sheet_probability": exp(-away_rate),
                    "away_clean_sheet_probability": exp(-home_rate),
                }
    except EndpointNotFound:
        result = None
    except (DataError, KeyError, IndexError, TypeError, ValueError) as error:
        LOGGER.debug("Could not load market for %s: %s", slug, error)
        return None
    if historical:
        _write(history_cache, json.dumps(result).encode())
    return result


def collect_polymarket_context(fixtures, cache_dir, *, workers=10):
    """Collect historical prices for supplied fixtures and capture live prices as of fetch time."""
    cache = Path(cache_dir) / "polymarket"
    now = pd.Timestamp.now(tz="UTC")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda match: _fixture_market(match, cache, now), fixtures.itertuples(index=False)))
    columns = FIXTURE_KEY + ["as_of", *MARKET_COLUMNS]
    quotes = pd.DataFrame.from_records([row for row in records if row is not None], columns=columns)
    quotes.rename(columns={"as_of": "market_asof"}, inplace=True)
    quotes["market_asof"] = pd.to_datetime(quotes["market_asof"], utc=True, format="mixed")
    return select_market_context(quotes, fixtures)


def main():
    from .app import _official_deadlines, default_cache
    from .snapshots import load_player_context

    parser = argparse.ArgumentParser(description="Capture deadline-safe Polymarket odds without an account.")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--season", type=int, choices=range(2024, 2100), action="append", metavar="START_YEAR")
    parser.add_argument("--live", action="store_true", help="Include the upcoming official Gameweek.")
    parser.add_argument("--output", type=Path, help="Destination market CSV. Defaults to cache/market.csv.")
    args = parser.parse_args()
    if not args.live and args.season is None:
        parser.error("Choose --season or --live.")
    if args.season is None and args.live and args.output is None:
        parser.error("Specify --output for a live-only capture so the historical market file is preserved.")

    cache = Path(args.cache or os.environ.get("FPL_CACHE_DIR") or default_cache()).expanduser()
    store = DataStore(cache)
    scopes = []
    for season in args.season or []:
        # Avoid loading the preceding nine archives for a single market capture.
        _, archive = store._historical_season(season)
        scopes.append(archive)
    if args.live:
        live = store.live()
        scopes.append(live.fixtures.loc[live.fixtures["gameweek"] == live.gameweek])
    fixtures = pd.concat(scopes, ignore_index=True).drop_duplicates(FIXTURE_KEY, keep="last")
    if args.live:
        fixtures = _official_deadlines(fixtures, load_player_context(cache, fixtures, live))
    rows = collect_polymarket_context(fixtures, cache)
    if rows.empty:
        parser.exit(1, "No eligible price snapshots were available.\n")
    output = args.output or cache / "market.csv"
    _write(output, rows.rename(columns={"market_asof": "as_of"}).to_csv(index=False).encode())
    for season, count in rows.groupby("season")["fixture"].count().items():
        print(f"{season}: {count} of {(fixtures['season'] == season).sum()} fixtures")
    print(output)


if __name__ == "__main__":
    main()
