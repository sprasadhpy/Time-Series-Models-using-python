# Time Series Models using Python

Teaching notebooks prepared as part of a computational finance course for
undergraduate and graduate students. The material walks through three families
of time series models, from univariate forecasting to volatility modeling and
multivariate macro forecasting.

## Notebooks

| Notebook | Topic | Learning objectives |
|----------|-------|---------------------|
| `Time_series_analysis_1.ipynb` | Univariate ARIMA models | Decomposition, stationarity and the ADF test, differencing and log transforms, ACF/PACF correlograms, ARMA/ARIMA/SARIMAX, order selection with AIC/BIC and rolling out-of-sample RMSE |
| `Time_series_analysis_2.ipynb` | ARCH/GARCH volatility forecasting | Heteroskedasticity, volatility clustering, order selection via rolling out-of-sample forecasts, fitting a GARCH model and diagnosing residuals |
| `Time_series_analysis_3.ipynb` | VAR / VARMAX macro forecasting | Unit root testing across multiple series, VAR(p) and VARMAX estimation, residual diagnostics, impulse response analysis |

## Data

All data is pulled from the [FRED](https://fred.stlouisfed.org/) database via
`pandas-datareader`. The series used are:

- `IPGMFN` – Industrial production, manufacturing
- `NASDAQCOM` – NASDAQ Composite index
- `UMCSENT` – University of Michigan consumer sentiment

The shared `load_fred` helper in `utils.py` caches each pull to `data/*.csv` on
first run, so the notebooks reproduce the same results offline afterwards.

## Setup

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate time-series-models
```

The `arch` package (used in notebook 2) is included in both files, so no
separate install step is required.

## Running

```bash
jupyter notebook
```

Then open any of the three notebooks and run the cells top to bottom. The first
run fetches data from FRED (network required); later runs use the cached CSVs in
`data/`.

## Reproducibility

- A global random seed is set in `utils.py` (`SEED = 42`).
- Data is cached to CSV so results do not change between runs.
- Notebook outputs are stripped from version control. Install the git filter
  once with `nbstripout --install` so committed notebooks stay clean.

## License

Released under the [MIT License](LICENSE).
