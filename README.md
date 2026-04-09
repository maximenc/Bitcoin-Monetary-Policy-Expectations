# Is Bitcoin A Hedge Against Central Banking?

Replication code for **"Is Bitcoin A Hedge Against Central Banking?"** by Maxime L. D. Nicolas, François Sicard, Marion Laboure, Zixin Sun, and Anahí Rodríguez-Martínez.

This repository contains the Jupyter notebooks required to replicate all figures and tables in the paper. Each notebook corresponds to a self-contained step of the analysis, with step-by-step instructions embedded in the cells.

---

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `01_MPE_Scrape_Stocktwits.ipynb` | Scrapes StockTwits messages tagged `$FED` and `$MACRO` (2014–2025) to build the raw corpus used to construct the Monetary Policy Expectations (MPE) index |
| 2 | `02_MPE_Classifier_Mistral.ipynb` | Classifies messages into five sentiment categories (Very Hawkish to Very Dovish) using a Mistral-7B LLM; constructs the daily MPE index — reproduces Tables 1–2 |
| 3 | `03_MPE_Analysis.ipynb` | Analyzes the MPE index: plots Bitcoin prices, MPE, and the Federal Funds Rate; computes return distributions by monetary regime — reproduces Figure 1, Figure 3, and Table 5 |
| 4 | `04_Google_Trends_Variables.ipynb` | Downloads Google Trends series (inflation, recession, climate change) via `pytrends` for use as control variables |
| 5 | `05_Data_Transform_Stationary.ipynb` | Harmonizes all series to a common frequency, applies stationarity transformations (log returns, first differences) as described in Table 11 |
| 6 | `06_Descriptive_Statistics.ipynb` | Computes and formats descriptive statistics for all variables — reproduces Tables 3–4 |
| 7 | `07_FOMC_Event_Study.ipynb` | Conducts an event study around FOMC announcement dates to measure Bitcoin's short-term response to monetary policy surprises — reproduces Figure 2 |
| 8 | `08_Granger_Causality.ipynb` | Runs standard Granger causality tests between the MPE index and Bitcoin returns/volatility — reproduces Table 6 |
| 9 | `09_VMD_Granger_Causality.ipynb` | Applies Variational Mode Decomposition (VMD) to decompose series by frequency and runs frequency-domain Granger causality tests — reproduces Table 7 |
| 10 | `10_ARIMA_GARCH.ipynb` | Fits ARIMA and GARCH models as baseline forecasters; reports RMSE and MAE benchmarks — reproduces baseline results in Table 8 |
| 11 | `11_LSTM.ipynb` | Trains an LSTM model with walk-forward validation and Optuna hyperparameter optimization; evaluates forecasting performance — reproduces Table 8, Figure 4, and Table 13 |
| 12 | `12_SHAP_Analysis.ipynb` | Computes SHAP values for LSTM predictions to interpret the contribution of each variable — reproduces Figures 5–9 and Tables 9–10 |

---

## Requirements

The notebooks run on Python 3.9+. Install dependencies with:

```bash
pip install numpy pandas scipy matplotlib statsmodels scikit-learn torch shap optuna vmdpy pytrends ollama
```

A GPU is recommended for notebooks 11 and 12 (LSTM training and SHAP analysis).

---

## Usage

Run the notebooks in order. StockTwits data (notebook 1) and Google Trends data (notebook 4) are collected automatically via their respective APIs. Notebook 2 requires a local Mistral-7B model served via [Ollama](https://ollama.com/).
