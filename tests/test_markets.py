import math
from datetime import timedelta

import pandas as pd
import pytest
from scipy.stats import poisson, skellam

from fpl import polymarket
from fpl.markets import MARKET_COLUMNS, MarketDataError, implied_goals, load_market_context


@pytest.fixture
def fixtures():
    return pd.DataFrame([
        {"season": 2025, "fixture": 40, "gameweek": 5, "home_team_code": 3, "away_team_code": 8,
         "deadline": pd.Timestamp("2025-09-20T11:00:00Z"), "kickoff": pd.Timestamp("2025-09-20T12:30:00Z")},
        {"season": 2025, "fixture": 41, "gameweek": 5, "home_team_code": 4, "away_team_code": 7,
         "deadline": pd.Timestamp("2025-09-20T11:00:00Z"), "kickoff": pd.Timestamp("2025-09-21T14:00:00Z")},
    ])


def test_market_deadline_is_shared_across_the_gameweek(fixtures, tmp_path):
    csv = tmp_path / "market.csv"
    csv.write_text(
        "season,fixture,as_of,home_expected_goals,away_expected_goals\n"
        "2025,40,2025-09-19T06:30:00-04:00,1.7,0.9\n"
        "2025,40,2025-09-20T09:30:00Z,2.1,1.0\n"
        "2025,40,2025-09-20T11:00:00Z,19,9\n"
        "2025,41,2025-09-21T08:00:00Z,19,9\n"
        "2026,40,2025-09-20T09:30:00Z,19,9\n",
        encoding="utf-8",
    )
    market = load_market_context(csv, fixtures)
    assert list(market["fixture"]) == [40]
    assert market.loc[0, "home_expected_goals"] == 2.1
    assert market.loc[0, "home_clean_sheet_probability"] == pytest.approx(math.exp(-1.0))
    assert market.loc[0, "away_clean_sheet_probability"] == pytest.approx(math.exp(-2.1))
    replay = load_market_context(csv, fixtures, as_of="2025-09-19T11:00:00Z")
    assert replay.loc[0, "home_expected_goals"] == 1.7


def test_market_rejects_missing_timezone_and_conflicting_duplicates(fixtures, tmp_path):
    csv = tmp_path / "market.csv"
    header = "season,fixture,as_of,home_expected_goals\n"
    csv.write_text(header + "2025,40,2025-09-20T09:30:00,1.6\n", encoding="utf-8")
    with pytest.raises(MarketDataError, match="explicit time zone"):
        load_market_context(csv, fixtures)
    csv.write_text(
        header + "2025,40,2025-09-20T09:30:00Z,1.6\n" + "2025,40,2025-09-20T09:30:00Z,1.7\n",
        encoding="utf-8",
    )
    with pytest.raises(MarketDataError, match="Conflicting"):
        load_market_context(csv, fixtures)


def test_decimal_odds_recovers_goal_rates(fixtures, tmp_path):
    home, away = 1.9, 0.7
    h = skellam.sf(0, home, away)
    d = skellam.pmf(0, home, away)
    a = skellam.cdf(-1, home, away)
    over = poisson.sf(2, home + away)
    assert implied_goals(h, d, a, over) == pytest.approx((home, away))
    assert implied_goals(h, d, a) == pytest.approx((home, away))
    csv = tmp_path / "market.csv"
    double_margin = [1 / (p * 1.05) for p in (h, d, a)] + [1 / (p * 1.03) for p in (over, 1 - over)]
    csv.write_text(
        "season,fixture,as_of,home_win_odds,draw_odds,away_win_odds,over_25_odds,under_25_odds\n"
        + "2025,40,2025-09-19T09:00:00Z," + ",".join(map(str, double_margin)),
        encoding="utf-8",
    )
    market = load_market_context(csv, fixtures)
    assert tuple(market.loc[0, list(MARKET_COLUMNS[:2])]) == pytest.approx((home, away))


def test_public_historical_source_never_uses_returned_post_cutoff_prices(monkeypatch):
    deadline = pd.Timestamp("2025-09-20T11:00:00Z")
    end = int((deadline - timedelta(hours=1)).timestamp())
    query = []

    def download(url):
        query.append(url)
        return ('{"history":[{"t":%s,"p":0.5},{"t":%s,"p":0.999}]}' % (end - 1, end + 30)).encode()

    monkeypatch.setattr(polymarket, "_download", download)
    assert polymarket._historical_price({"outcomes": '["Yes", "No"]', "clobTokenIds": '["1", "2"]'}, "Yes", deadline) == (end - 1, 0.5)
    assert f"market=1&" in query[0] and f"endTs={end}&" in query[0]
