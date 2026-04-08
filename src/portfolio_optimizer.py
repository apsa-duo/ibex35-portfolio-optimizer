"""
Portfolio optimization module using the Lagrange multiplier method.

Implements the analytical (closed-form) Markowitz mean-variance
optimization.  Instead of relying on a numerical solver, the optimal
weights are obtained by constructing an augmented system of linear
equations derived from the Lagrangian and solving via matrix inversion.

Mathematical background
-----------------------
For *n* assets the Lagrangian of the minimum-variance problem with
a target expected return *E* is:

    L = w'Σw − λ₁(w'μ − E) − λ₂(Σwᵢ − 1)

Setting the partial derivatives to zero produces the linear system
``A · x = b`` where:

* **A** is the ``(n+2) × (n+2)`` augmented matrix built from 2Σ, −μ and −1.
* **b** is the ``(n+2) × 1`` vector ``[0 … 0, −E, −1]ᵀ``.
* **x** contains the optimal weights followed by the two Lagrange
  multipliers λ₁ and λ₂.

For the *global* minimum-variance portfolio (no target-return
constraint), the system reduces to ``(n+1) × (n+1)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Matrix construction helpers
# ---------------------------------------------------------------------------


def buildConstraintMatrixA(
    covarianceMatrix: pd.DataFrame,
    expectedReturns: dict[str, float],
) -> pd.DataFrame:
    """
    Build the augmented Lagrangian constraint matrix **A**.

    The matrix has shape ``(n+2, n+2)`` where *n* is the number
    of assets.  The upper-left block is ``2Σ`` (twice the covariance
    matrix) — the factor of 2 arises from differentiating the
    quadratic variance objective.  Two extra rows and columns encode
    the equality constraints for the target return and the budget
    (weights summing to one).

    Args:
        covarianceMatrix: ``n × n`` covariance matrix of daily returns.
        expectedReturns: Dictionary mapping each ticker to its mean return.

    Returns:
        An ``(n+2) × (n+2)`` augmented ``pd.DataFrame``.
    """
    numAssets = covarianceMatrix.shape[0]

    # Multiply covariance by 2 (Lagrangian first-order condition)
    matrixA = covarianceMatrix * 2

    # Column of negative expected returns
    negExpectedReturns = [
        -1 * expectedReturns[ticker] for ticker in matrixA.index
    ]
    matrixA["extraX1"] = negExpectedReturns

    # Column of -1 (budget constraint)
    matrixA["extraX2"] = [-1] * numAssets

    # Row for the return constraint
    returnConstraintRow = negExpectedReturns + [0, 0]
    matrixA.loc["extraY1"] = returnConstraintRow

    # Row for the budget constraint
    budgetConstraintRow = [-1] * numAssets + [0, 0]
    matrixA.loc["extraY2"] = budgetConstraintRow

    return matrixA


def invertMatrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the inverse of a square matrix preserving DataFrame labels.

    Args:
        matrix: A square ``pd.DataFrame``.

    Returns:
        The inverse as a ``pd.DataFrame`` with the same index and columns.
    """
    return pd.DataFrame(
        np.linalg.inv(matrix),
        columns=matrix.columns,
        index=matrix.index,
    )


def buildTargetVectorB(
    targetReturn: float,
    covarianceMatrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the right-hand-side target vector **b**.

    The first *n* entries are zero; the last two are ``−E`` (target
    return) and ``−1`` (budget constraint).

    Args:
        targetReturn: Desired portfolio expected return.
        covarianceMatrix: Used only to infer the number of assets.

    Returns:
        An ``(n+2) × 1`` ``pd.DataFrame``.
    """
    numAssets = len(covarianceMatrix.columns)
    zeros = np.zeros((numAssets, 1))
    tail = np.array([[-targetReturn], [-1]])
    data = np.vstack([zeros, tail])

    return pd.DataFrame(data, columns=["Values"])


# ---------------------------------------------------------------------------
# Single-portfolio solvers
# ---------------------------------------------------------------------------


def solvePortfolioWeights(
    targetReturn: float,
    covarianceMatrix: pd.DataFrame,
    expectedReturns: dict[str, float],
    *,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Find the minimum-variance portfolio for a given target return.

    Solves the system ``x = A⁻¹ · b`` and prints the resulting
    weights, portfolio return, variance, volatility, and Lagrange
    multipliers.

    Args:
        targetReturn: Desired expected return *E*.
        covarianceMatrix: Covariance matrix of daily returns.
        expectedReturns: Mean daily return per ticker.
        verbose: If ``True``, print intermediate matrices and results.

    Returns:
        A tuple ``(volatility, portfolioReturn)`` representing the
        coordinates of this portfolio on the risk–return plane.
    """
    matrixA = buildConstraintMatrixA(covarianceMatrix, expectedReturns)
    inverseA = invertMatrix(matrixA)
    vectorB = buildTargetVectorB(targetReturn, covarianceMatrix)
    solutionX = np.dot(inverseA, vectorB)

    if verbose:
        print(f"\nFor E = {targetReturn}, the matrices are:")
        print(f"Matrix A:\n{matrixA}")
        print(f"Inverse of A:\n{inverseA}")
        print(f"Vector B:\n{vectorB}")
        print(f"Solution X:\n{solutionX}")

    # Extract optimal weights
    numAssets = len(covarianceMatrix)
    weights = pd.Series(
        solutionX[:numAssets, 0], index=covarianceMatrix.columns
    )
    lagrangeReturn = solutionX[numAssets, 0]
    lagrangeBudget = solutionX[numAssets + 1, 0]

    # Portfolio metrics
    portfolioReturn = sum(
        weights[ticker] * expectedReturns[ticker]
        for ticker in expectedReturns
    )
    portfolioVariance = float(weights.T @ covarianceMatrix @ weights)
    portfolioVolatility = portfolioVariance ** 0.5

    if verbose:
        print("Optimal asset weights:")
        print(weights)
        print(f"Expected portfolio return: {portfolioReturn}")
        print(f"Portfolio variance: {portfolioVariance}")
        print(f"Portfolio volatility (std. dev.): {portfolioVolatility}")
        print(f"Lagrange multiplier (return constraint): {lagrangeReturn}")
        print(f"Lagrange multiplier (budget constraint): {lagrangeBudget}")
        print(f"Sum of weights: {weights.sum()}")

    return portfolioVolatility, portfolioReturn


# ---------------------------------------------------------------------------
# Minimum-variance portfolio (single constraint: weights sum to 1)
# ---------------------------------------------------------------------------


def _buildMinVarMatrixA(covarianceMatrix: pd.DataFrame) -> pd.DataFrame:
    """
    Build the ``(n+1) × (n+1)`` system for the *global* minimum-variance
    portfolio (no target-return constraint).
    """
    numAssets = covarianceMatrix.shape[0]
    matrixA = covarianceMatrix * 2

    matrixA["extraX1"] = [-1] * numAssets
    matrixA.loc["extraY1"] = [-1] * numAssets + [0]

    return matrixA


def _buildMinVarVectorB(covarianceMatrix: pd.DataFrame) -> pd.DataFrame:
    """Build the ``(n+1) × 1`` target vector for the global MVP."""
    numAssets = len(covarianceMatrix.columns)
    data = np.zeros((numAssets + 1, 1))
    data[-1, 0] = -1
    return pd.DataFrame(data, columns=["Values"])


def calculateMinimumVariancePortfolio(
    covarianceMatrix: pd.DataFrame,
    expectedReturns: dict[str, float],
    *,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Calculate the *global* minimum-variance portfolio (MVP).

    This portfolio minimises variance **without** constraining the
    expected return.  It represents the leftmost point of the
    efficient frontier.

    Args:
        covarianceMatrix: Covariance matrix of daily returns.
        expectedReturns: Mean daily return per ticker.
        verbose: If ``True``, print the resulting weights and metrics.

    Returns:
        A tuple ``(volatility, portfolioReturn)`` for the MVP.
    """
    matrixA = _buildMinVarMatrixA(covarianceMatrix)
    inverseA = invertMatrix(matrixA)
    vectorB = _buildMinVarVectorB(covarianceMatrix)
    solutionX = np.dot(inverseA, vectorB)

    numAssets = len(covarianceMatrix)
    weights = pd.Series(
        solutionX[:numAssets, 0], index=covarianceMatrix.columns
    )
    lagrangeBudget = solutionX[numAssets, 0]

    portfolioReturn = sum(
        weights[ticker] * expectedReturns[ticker]
        for ticker in expectedReturns
    )
    portfolioVariance = float(weights.T @ covarianceMatrix @ weights)
    portfolioVolatility = portfolioVariance ** 0.5

    if verbose:
        print("\n--- Global Minimum-Variance Portfolio ---")
        print("Optimal asset weights:")
        print(weights)
        print(f"Expected portfolio return: {portfolioReturn}")
        print(f"Portfolio variance: {portfolioVariance}")
        print(f"Portfolio volatility (std. dev.): {portfolioVolatility}")
        print(f"Lagrange multiplier (budget constraint): {lagrangeBudget}")
        print(f"Sum of weights: {weights.sum()}")

    return portfolioVolatility, portfolioReturn


# ---------------------------------------------------------------------------
# Efficient frontier generation
# ---------------------------------------------------------------------------


def _computeTargetReturns(
    expectedReturns: dict[str, float],
    numPortfolios: int,
) -> list[float]:
    """
    Generate evenly spaced target returns between the lowest and
    highest individual stock expected returns, doubled (as used in
    the original academic model).

    The doubling factor is an artefact of the course assignment's
    convention for annualising / scaling the target return range.

    Args:
        expectedReturns: Mean daily return per ticker.
        numPortfolios: How many target-return levels to create.

    Returns:
        A list of *numPortfolios* target-return values.
    """
    minReturn = min(expectedReturns.values())
    maxReturn = max(expectedReturns.values())
    spread = maxReturn - minReturn
    interval = spread / (numPortfolios - 1)

    minTicker = min(expectedReturns, key=expectedReturns.get)
    maxTicker = max(expectedReturns, key=expectedReturns.get)
    print(
        f"Stock with the lowest expected return: {minTicker} "
        f"({expectedReturns[minTicker]:.4f})"
    )
    print(
        f"Stock with the highest expected return: {maxTicker} "
        f"({expectedReturns[maxTicker]:.4f})"
    )
    print(f"Spread between highest and lowest: {spread:.4f}")

    targetReturns = [
        (minReturn + interval * i) * 2 for i in range(numPortfolios)
    ]
    return targetReturns


def generateEfficientFrontier(
    expectedReturns: dict[str, float],
    covarianceMatrix: pd.DataFrame,
    numPortfolios: int,
) -> list[list[float]]:
    """
    Compute the efficient frontier by solving for multiple target returns.

    For each target return the minimum-variance portfolio is found
    analytically.  The resulting ``(volatility, return)`` pairs trace
    out the efficient frontier in risk–return space.

    Args:
        expectedReturns: Mean daily return per ticker.
        covarianceMatrix: Covariance matrix of daily returns.
        numPortfolios: Number of points along the frontier.

    Returns:
        A list of ``[volatility, return]`` pairs.
    """
    targetReturns = _computeTargetReturns(expectedReturns, numPortfolios)
    print(f"Target returns for each portfolio: {targetReturns}")

    coordinates: list[list[float]] = []
    for target in targetReturns:
        volatility, portfolioReturn = solvePortfolioWeights(
            target, covarianceMatrix, expectedReturns
        )
        coordinates.append([volatility, portfolioReturn])

    return coordinates
