"""
Configuration module for the IBEX-35 Portfolio Optimizer.

Centralizes all global constants and display settings used across
the application. Modifying values here allows easy experimentation
with different stocks, date ranges, or portfolio counts without
touching business logic.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Stock Universe
# ---------------------------------------------------------------------------
IBEX35_TICKERS: list[str] = [
    "AENA.MC",   # Aena S.M.E.
    "BBVA.MC",   # Banco Bilbao Vizcaya Argentaria
    "SAN.MC",    # Banco Santander
    "IBE.MC",    # Iberdrola
    "ANA.MC",    # Acciona
    "TEF.MC",    # Telefónica
]

# ---------------------------------------------------------------------------
# Analysis Period
# ---------------------------------------------------------------------------
START_DATE: str = "2024-01-02"
END_DATE: str = "2024-10-31"

# ---------------------------------------------------------------------------
# Optimization Parameters
# ---------------------------------------------------------------------------
NUM_PORTFOLIOS: int = 9  # Number of portfolios on the efficient frontier

# ---------------------------------------------------------------------------
# Display Settings
# ---------------------------------------------------------------------------


def applyDisplaySettings() -> None:
    """
    Configure pandas and numpy display options for clean, readable output.

    Sets high float precision, removes scientific notation,
    and removes row/column display limits so that full matrices
    and data frames are printed without truncation.
    """
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.float_format", "{:.6f}".format)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1000)
