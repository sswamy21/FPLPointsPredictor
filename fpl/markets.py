"""Deadline-safe, optional fixture market inputs."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import poisson, skellam


FIXTURE_KEY = ["season", "fixture"]
MARKET_COLUMNS = (
    "home_expected_goals", "away_expected_goals",
    "home_clean_sheet_probability", "away_clean_sheet_probability",
)
DECIMAL_ODDS = ("home_win_odds", "draw_odds", "away_win_odds", "over_25_odds", "under_25_odds")
MARKET_SCHEMA = {
    "season": "int64", "fixture": "int64", "market_asof": "datetime64[ns, UTC]",
    **{column: "float64" for column in MARKET_COLUMNS},
}


class MarketDataError(ValueError):
    """A market CSV is unusable or contains conflicting snapshots."""


def _empty_market():
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in MARKET_SCHEMA.items()})


def implied_goals(home_win, draw, away_win, over=None, line=2.5):
    """Fit independent Poisson rates to the total and normalized home-away win balance."""
    win_prices = np.array([home_win, draw, away_win], dtype=float)
    if not np.isfinite(win_prices).all() or (win_prices <= 0).any():
        raise MarketDataError("Market price probabilities must be finite and strictly positive.")
    win_prices /= win_prices.sum()
    if over is None:
        target = win_prices[[0, 2]]

        def win_residual(rates):
            home_rate, away_rate = rates
            return [skellam.sf(0, home_rate, away_rate) - target[0], skellam.cdf(-1, home_rate, away_rate) - target[1]]

        fit = least_squares(win_residual, [1.5, 1.2], bounds=(0.01, 8))
        if not fit.success or np.max(np.abs(fit.fun)) > 0.02:
            raise MarketDataError("Match result prices are incompatible with plausible goal rates.")
        return tuple(fit.x)
    if not 0 < over < 1:
        raise MarketDataError("The over-goals probability must be between zero and one.")
    total = brentq(lambda rate: poisson.sf(int(line), rate) - over, 0, 32)
    win_balance = win_prices[0] - win_prices[2]

    def balance(home_rate):
        away_rate = total - home_rate
        return 1 - skellam.cdf(0, home_rate, away_rate) - skellam.cdf(-1, home_rate, away_rate)

    try:
        home_rate = brentq(lambda rate: balance(rate) - win_balance, 1e-8, total - 1e-8)
    except ValueError as error:
        raise MarketDataError("Match result and total prices imply incompatible goal rates.") from error
    return home_rate, total - home_rate


def _number(raw, column):
    values = raw[column]
    parsed = pd.to_numeric(values, errors="coerce")
    if (values.notna() & ~np.isfinite(parsed)).any():
        raise MarketDataError(f"Market column {column!r} contains a non-numeric or non-finite value.")
    return parsed


def _fill_market_values(rows):
    values = pd.DataFrame(index=rows.index)
    for i, column in enumerate(MARKET_COLUMNS):
        values[column] = _number(rows, column) if column in rows else np.nan
        invalid = values[column].lt(0) | (values[column].gt(1) if i >= 2 else False)
        if invalid.any():
            raise MarketDataError(f"Market column {column!r} contains a value outside its allowed range.")

    if all(column in rows for column in DECIMAL_ODDS):
        odds = pd.DataFrame({column: _number(rows, column) for column in DECIMAL_ODDS})
        if odds.le(1).any(axis=None):
            raise MarketDataError("Decimal market odds must be greater than one.")
        for index, row in odds.dropna().iterrows():
            home, draw, away, over, under = 1 / row.to_numpy(dtype=float)
            if values.loc[index, list(MARKET_COLUMNS[:2])].isna().any():
                try:
                    inferred = implied_goals(home, draw, away, over / (over + under))
                except ValueError as error:
                    raise MarketDataError(f"Cannot convert decimal odds near CSV line {index + 2}: {error}") from error
                for column, rate in zip(MARKET_COLUMNS[:2], inferred):
                    if pd.isna(values.at[index, column]):
                        values.at[index, column] = rate

    for index in (0, 1):
        clean_sheet = MARKET_COLUMNS[2 + index]
        opposition_goals = MARKET_COLUMNS[1 - index]
        values[clean_sheet] = values[clean_sheet].fillna(np.exp(-values[opposition_goals]))
    return values


def select_market_context(quotes, fixtures, *, as_of=None):
    """Keep the latest usable quote before kickoff, the round deadline and the global cutoff."""
    if quotes.empty:
        return _empty_market()
    cutoffs = fixtures[FIXTURE_KEY + ["deadline", "kickoff"]]
    eligible = quotes.merge(cutoffs, on=FIXTURE_KEY, how="inner", validate="many_to_one")
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if pd.isna(now) or now.tzinfo is None:
        raise MarketDataError("The market cutoff must include a time zone.")
    deadline = pd.to_datetime(eligible["deadline"], utc=True, errors="coerce")
    kickoff = pd.to_datetime(eligible["kickoff"], utc=True, errors="coerce")
    observed = eligible["market_asof"]
    eligible = eligible.loc[(observed <= now) & (observed < deadline) & (observed < kickoff)]
    eligible = eligible[list(MARKET_SCHEMA)].dropna(subset=list(MARKET_COLUMNS), how="all").drop_duplicates()
    if eligible.duplicated(FIXTURE_KEY + ["market_asof"]).any():
        raise MarketDataError("Conflicting market values for the same fixture and capture time.")
    return eligible.sort_values("market_asof").drop_duplicates(FIXTURE_KEY, keep="last").reset_index(drop=True)


def load_market_context(path: Path | None, fixtures: pd.DataFrame, *, as_of=None) -> pd.DataFrame:
    """Load a local CSV of normalized rates or decimal match-result and O/U 2.5 odds."""
    if path is None:
        return _empty_market()
    try:
        raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise MarketDataError(f"Could not read market CSV {path}: {error}") from error

    required = set(FIXTURE_KEY + ["as_of"])
    if not required.issubset(raw.columns):
        raise MarketDataError("Market CSV must contain season, fixture and as_of columns.")
    if not set(MARKET_COLUMNS).intersection(raw.columns) and not set(DECIMAL_ODDS).issubset(raw.columns):
        raise MarketDataError("Market CSV must contain goal estimates, clean-sheet probabilities or the five decimal odds columns.")
    quotes = pd.DataFrame(index=raw.index)
    for column in FIXTURE_KEY:
        values = _number(raw, column)
        if values.isna().any() or not values.ge(1).all() or not values.le(2**31).all() or not values.mod(1).eq(0).all():
            raise MarketDataError(f"Market column {column!r} must contain positive integer IDs.")
        quotes[column] = values.astype("int64")

    captures = raw["as_of"].str.strip()
    utc = pd.to_datetime(captures, format="mixed", utc=True, errors="coerce")
    if utc.isna().any() or not captures.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
        raise MarketDataError("Each market as_of timestamp must be valid ISO 8601 with an explicit time zone.")
    quotes["market_asof"] = utc
    return select_market_context(quotes.join(_fill_market_values(raw)), fixtures, as_of=as_of)
