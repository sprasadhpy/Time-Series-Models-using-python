"""Shared helpers for the time series analysis notebooks.

Keeping these in one module avoids copy-pasting the same code into every
notebook and prevents the three copies from drifting apart.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot, moment
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set a global seed so model fits and any sampling are reproducible for teaching.
SEED = 42
np.random.seed(SEED)

DATA_DIR = "data"


def load_fred(series, start, end, filename):
    """Load a FRED series, caching to CSV so notebooks run offline.

    On the first run this pulls from FRED and writes ``data/<filename>``.
    Subsequent runs read the cached CSV, giving stable, reproducible results
    without a network connection.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).squeeze()

    import pandas_datareader.data as web

    data = web.DataReader(series, "fred", start, end)
    data.to_csv(path)
    return data.squeeze()


def correlogram(x, lags, title):
    """Plot series, probability plot, ACF and PACF with summary statistics."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

    x.plot(ax=axes[0][0], grid=True, color="red")
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f"ADF: {adfuller(x)[1]:.2f}\nQ-Stat: {q_p:.2f}"
    axes[0][0].text(x=0.03, y=0.85, s=stats, transform=axes[0][0].transAxes)

    probplot(x, plot=axes[0][1])
    mean, variance, skewness, kurtosis = moment(x, moment=[1, 2, 3, 4])
    stats1 = (
        f"Mean: {mean:.2f}\nSD: {np.sqrt(variance):.2f}\n"
        f"Skew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}"
    )
    axes[0][1].text(x=0.02, y=0.75, s=stats1, transform=axes[0][1].transAxes)

    plot_acf(x, lags=lags, zero=False, ax=axes[1][0], alpha=0.05)
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel("Lag")
    axes[1][1].set_xlabel("Lag")

    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig


def unit_root_test(df):
    """Print the Augmented Dickey-Fuller p-value for a series."""
    result = adfuller(df.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
