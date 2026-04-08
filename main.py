"""
IBEX-35 Portfolio Optimizer - Main Entry Point

Orchestrates data fetching, statistical analysis, and Markowitz
mean-variance optimization for the configured IBEX-35 stocks.
Generates an efficient frontier and visualises the results.
"""

from src import (
    config,
    data_loader,
    portfolio_analytics,
    portfolio_optimizer,
    visualization,
)


def main() -> None:
    # 1. Apply display settings
    config.applyDisplaySettings()

    # 2. Download adjusted close prices & dividends
    print(f"Downloading data for {config.IBEX35_TICKERS}...")
    closePrices, dividends = data_loader.downloadStockData(
        config.IBEX35_TICKERS,
        config.START_DATE,
        config.END_DATE,
    )
    print("\n--- Adjusted Close Prices ---")
    print(closePrices.head())
    print("\n--- Dividends ---")
    print(dividends.head())

    # 3. Calculate daily returns & expected means
    print("\nCalculating daily returns...")
    dailyReturns = data_loader.calculateDailyReturns(closePrices, dividends)
    print(dailyReturns.head())

    expectedReturns = data_loader.calculateExpectedReturns(
        dailyReturns, config.IBEX35_TICKERS
    )
    print("\n--- Expected Daily Returns ---")
    for ticker, meanReturn in expectedReturns.items():
        print(f"{ticker}: {meanReturn:.6f}")

    # 4. Plot return histograms
    print("\nGenerating return distribution histograms...")
    visualization.plotReturnHistograms(dailyReturns)

    # 5. Compute Covariance & Correlation matrices
    print("\n--- Covariance Matrix ---")
    covarianceMatrix = portfolio_analytics.calculateCovarianceMatrix(
        dailyReturns
    )
    print(covarianceMatrix)

    print("\n--- Correlation Matrix ---")
    correlationMatrix = portfolio_analytics.calculateCorrelationMatrix(
        dailyReturns
    )
    print(correlationMatrix)

    # 6. Generate the efficient frontier (multiple portfolios)
    print(f"\nGenerating {config.NUM_PORTFOLIOS} portfolios for the Efficient Frontier...")
    frontierCoordinates = portfolio_optimizer.generateEfficientFrontier(
        expectedReturns,
        covarianceMatrix,
        config.NUM_PORTFOLIOS,
    )

    # 7. Compute the global Minimum-Variance Portfolio (MVP)
    mvpCoordinates = portfolio_optimizer.calculateMinimumVariancePortfolio(
        covarianceMatrix,
        expectedReturns,
    )

    # 8. Plot the Efficient Frontier with the MVP
    print("\nPlotting the Efficient Frontier...")
    visualization.plotEfficientFrontier(frontierCoordinates, mvpCoordinates)


if __name__ == "__main__":
    main()
