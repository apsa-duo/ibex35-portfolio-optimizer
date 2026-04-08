"""
Visualization module for the IBEX-35 Portfolio Optimizer.

Provides functions to generate publication-quality charts including
return-distribution histograms and the efficient frontier curve.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plotReturnHistograms(dailyReturns: pd.DataFrame) -> None:
    """
    Plot normalised histograms with KDE overlays for each stock's daily returns.

    Each stock gets its own subplot, stacked vertically, so that the
    shape of each return distribution can be compared at a glance.

    Args:
        dailyReturns: DataFrame of daily total returns (one column per ticker).
    """
    numStocks = len(dailyReturns.columns)
    fig, axes = plt.subplots(
        numStocks, 1,
        figsize=(8, 4 * numStocks),
    )
    fig.tight_layout(pad=5.0)

    for index, ticker in enumerate(dailyReturns.columns):
        ax = axes[index]
        ax.hist(
            dailyReturns[ticker],
            bins=30,
            alpha=0.4,
            density=True,
        )
        sns.kdeplot(
            dailyReturns[ticker],
            label="Density curve",
            ax=ax,
        )
        ax.set_title(f"Histogram & Density — {ticker}")
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.legend()

    plt.show()


def plotEfficientFrontier(
    frontierCoordinates: list[list[float]],
    minimumVariancePoint: tuple[float, float] | None = None,
) -> None:
    """
    Plot the efficient frontier curve (and optionally the MVP).

    The efficient frontier is the set of portfolios offering the
    highest expected return for each level of risk.  If the global
    Minimum-Variance Portfolio (MVP) coordinates are supplied, it is
    highlighted as a red dot on the chart.

    Args:
        frontierCoordinates: List of ``[volatility, return]`` pairs
            tracing out the frontier.  When the MVP is included its
            coordinate should **not** be in this list — pass it via
            *minimumVariancePoint* instead.
        minimumVariancePoint: Optional ``(volatility, return)`` tuple
            for the global minimum-variance portfolio.
    """
    # If MVP is provided, merge and sort for a smooth curve
    if minimumVariancePoint is not None:
        allCoordinates = frontierCoordinates + [list(minimumVariancePoint)]
        allCoordinates.sort(key=lambda c: c[1])  # sort by return
        mvpCoord = minimumVariancePoint
    else:
        allCoordinates = list(frontierCoordinates)
        allCoordinates.sort(key=lambda c: c[1])
        mvpCoord = None

    volatilities = [coord[0] for coord in allCoordinates]
    returns = [coord[1] for coord in allCoordinates]

    plt.figure(figsize=(10, 6))
    plt.plot(
        volatilities,
        returns,
        marker="o",
        linestyle="-",
        color="blue",
        label="Efficient Frontier",
    )

    if mvpCoord is not None:
        plt.scatter(
            mvpCoord[0],
            mvpCoord[1],
            color="red",
            s=100,
            zorder=5,
            label="Minimum-Variance Portfolio",
        )

    plt.xlabel("Portfolio Volatility", fontsize=12)
    plt.ylabel("Portfolio Expected Return (E[r])", fontsize=12)
    plt.title("Efficient Frontier", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
