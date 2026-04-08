"""
Portfolio analytics module.

Provides statistical measures (covariance, correlation) that
form the foundation of Markowitz mean-variance optimization.
"""

from __future__ import annotations

import pandas as pd


def calculateCovarianceMatrix(dailyReturns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the covariance matrix of daily returns.

    The covariance matrix quantifies how pairs of asset returns
    move together, which is essential for portfolio risk estimation.

    Args:
        dailyReturns: DataFrame of daily total returns (one column per ticker).

    Returns:
        A symmetric ``pd.DataFrame`` covariance matrix.
    """
    return dailyReturns.cov()


def calculateCorrelationMatrix(dailyReturns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix of daily returns.

    Correlation normalises covariance to the [−1, 1] range,
    making it easier to interpret the strength of linear
    relationships between asset returns.

    Args:
        dailyReturns: DataFrame of daily total returns (one column per ticker).

    Returns:
        A symmetric ``pd.DataFrame`` correlation matrix.
    """
    return dailyReturns.corr()
