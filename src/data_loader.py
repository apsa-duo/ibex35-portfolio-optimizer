"""
Data loading and preprocessing module.

Handles downloading historical stock data from Yahoo Finance
and computing the daily total returns (capital gains + dividends).
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def downloadStockData(
    tickers: list[str],
    startDate: str,
    endDate: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted close prices and dividend data for the given tickers.

    Uses *yfinance* with ``actions=True`` so that both price and
    corporate-action data (dividends, splits) are fetched in a
    single network call.

    Args:
        tickers: List of Yahoo Finance ticker symbols (e.g. ``["AENA.MC"]``).
        startDate: Start date in ``YYYY-MM-DD`` format (inclusive).
        endDate: End date in ``YYYY-MM-DD`` format (inclusive).

    Returns:
        A tuple ``(closePrices, dividends)`` where each element is a
        ``pd.DataFrame`` indexed by date with one column per ticker.
    """
    rawData = yf.download(
        tickers,
        start=startDate,
        end=endDate,
        actions=True,
    )

    closePrices: pd.DataFrame = rawData["Adj Close"]
    dividends: pd.DataFrame = rawData["Dividends"]

    return closePrices, dividends


def calculateDailyReturns(
    closePrices: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily total returns for each stock.

    The total return on day *t* accounts for both the change in
    adjusted close price **and** any dividends received::

        r_t = (D_t + P_t − P_{t−1}) / P_{t−1}

    The first row (which would be ``NaN`` due to the lag) is dropped.

    Args:
        closePrices: Adjusted close prices indexed by date.
        dividends: Dividend payments indexed by date.

    Returns:
        A ``pd.DataFrame`` of daily total returns, one column per ticker.
    """
    dailyReturns: pd.DataFrame = (
        (dividends + closePrices - closePrices.shift(1)) / closePrices.shift(1)
    ).dropna()

    return dailyReturns


def calculateExpectedReturns(
    dailyReturns: pd.DataFrame,
    tickers: list[str],
) -> dict[str, float]:
    """
    Calculate the expected (mean) daily return for each stock.

    Args:
        dailyReturns: DataFrame of daily total returns.
        tickers: List of ticker symbols to compute means for.

    Returns:
        A dictionary mapping each ticker to its mean daily return.
    """
    expectedReturns: dict[str, float] = {}

    for ticker in tickers:
        expectedReturns[ticker] = dailyReturns[ticker].mean()

    return expectedReturns
