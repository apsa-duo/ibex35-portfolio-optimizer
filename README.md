# IBEX-35 Portfolio Optimizer 📈

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

A professional, closed-form Markowitz Mean-Variance Portfolio Optimization engine developed in Python. This tool automatically fetches historical stock data from Yahoo Finance, computes the necessary risk/return matrices, and applies constrained Lagrangian optimisation to calculate the Efficient Frontier for selected IBEX-35 stocks.

## 🎯 The Problem It Solves

Modern Portfolio Theory asserts that an investor can construct a set of optimal portfolios that offer the maximum possible expected return for a given level of risk. 

While many libraries solve this numerically (using gradient descent or iterative solvers), this repository implements the **analytical, closed-form mathematical solution** using Lagrange multipliers. It provides a transparent, trackable, and mathematically rigorous calculation of the Efficient Frontier and the Global Minimum-Variance Portfolio.

## 🌟 Key Features

- **Automated Data Pipeline:** Fetches adjusted closing prices and dividend histories directly via the `yfinance` API.
- **Total Return Calculation:** Accurately calculates daily total returns incorporating both capital gains and dividend payouts.
- **Analytical Optimization:** Uses Lagrangian matrix inversion for exact constraint solving rather than computational approximations.
- **Risk Analytics:** Computes covariance and correlation matrices natively.
- **Advanced Visualization:** Generates density histograms, scatter plots, and the continuous Efficient Frontier curve using `matplotlib` and `seaborn`.
- **Modular Architecture:** Cleanly separated concerns (Data, Analytics, Optimization, Visualization) following SOLID principles.

## 📁 Project Structure

```text
ibex35-portfolio-optimizer/
│
├── src/                        # Core application modules
│   ├── config.py               # Constants, tickers, date ranges
│   ├── data_loader.py          # yfinance API integration
│   ├── portfolio_analytics.py  # Covariance & Correlation math
│   ├── portfolio_optimizer.py  # Lagrangian optimization engine
│   └── visualization.py        # Chart generation (Histograms, Frontier)
│
├── docs/                       # Original academic research and notebooks
│   ├── Primera_práctica_Dirección_Financiera_I_Grupo_11.ipynb
│   └── Primera_práctica_Dirección_Financiera_I_Grupo_11.pdf
│
├── main.py                     # Primary entry point & workflow orchestrator
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🚀 Setup & Installation

Follow these steps to run the portfolio optimization engine locally.

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ibex35-portfolio-optimizer.git
   cd ibex35-portfolio-optimizer
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Optimizer**
   ```bash
   python main.py
   ```
   *The script will download the data, calculate matrices, print metrics to the console, and launch interactive Matplotlib windows showing the return distributions and the Efficient Frontier.*

## 🛣️ Future Roadmap

- [ ] **Dynamic Ticker Selection:** Allow passing stock tickers and date ranges via command-line arguments (CLI).
- [ ] **Risk-Free Rate Integration:** Add the ability to specify a risk-free rate to calculate the optimal Sharpe Ratio portfolio (Capital Allocation Line).
- [ ] **Web Dashboard:** Migrate the visualizations from Matplotlib to a Streamlit or Dash web application for interactive exploration.

## 👥 Contributors

This tool was researched, designed, and implemented as a collaborative Financial Management project by:

* **Sergio Alonso Zarcero** 
* **Andrea Pascual Aguilera**
* **Angel Meda Lopez**
* **Marta Pulido Alobera**
* **Sandra Zorrilla Gutierrez**

---
*Built with ❤️ for quantitative finance and clean code.*