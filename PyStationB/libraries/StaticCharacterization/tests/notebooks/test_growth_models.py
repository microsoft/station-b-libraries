# This test focuses on the
# development and exploration of growth models and their properties.
# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from psbutils.filecheck import Plottable, figure_found
from scipy.optimize import curve_fit


def plot_figure(name: str, ax: Optional[Plottable] = None) -> None:
    sns.despine()
    assert figure_found(ax, f"test_growth_models/{name}")
    plt.clf()


# Logistic
def logistic_model(ts: np.ndarray, mu: float, K: float, c0: float, lag: float) -> np.ndarray:
    return np.array([K / (1.0 + (K - c0) / c0 * np.exp(mu * (lag - t))) if t > lag else c0 for t in ts])


# Gompertz
def model(ts: np.ndarray, mu: float, K: float, c0: float, lag: float) -> np.ndarray:
    return np.array([K * np.exp(np.log(c0 / K) * np.exp(mu * (lag - t))) if t > lag else c0 for t in ts])


@pytest.mark.timeout(30)
def test_growth_models():
    r_true = 0.015
    K_true = 2
    c0_true = 0.002
    lag_true = 200
    sig = 0.05

    n = 101
    ts = np.linspace(0, 1200, n)
    xs = model(ts, r_true, K_true, c0_true, lag_true)
    np.random.seed(42)
    ys = xs * (1 + sig * np.random.randn(n))
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(ts, xs)
    plot_figure("plot1_scatter")

    mle = curve_fit(model, ts, ys, p0=[0.02, 2, 0.01, 100])[0]
    r, K, c0, lag = mle[:4]

    df = pd.DataFrame(mle, columns=["MLE"])
    df.insert(0, "Names", ["r", "K", "c0", "lag"])
    print(df)

    plt.figure(figsize=(6.4, 4.8))
    ax = plt.subplot()
    ax.scatter(ts, ys, c="k", s=2, label="Data")
    ax.plot(ts, model(ts, r, K, c0, lag), c="r", label="Model")
    ax.plot(ts, model(ts, r, K, c0 / 100, lag), c="b", label="Model (smaller c0)")
    ax.plot(ts, model(ts, r, K, c0 * 100, lag), c="g", label="Model (larger c0)")
    plt.legend()
    plot_figure("plot2_models")

    # Try fitting {r, K, c0} with lag fixed. As you vary lag, the optimizer just finds a different value for c0
    # that makes the model fit the data just as well.

    lag = 100.0

    def objective1(t, r, K, c0):
        return model(t, r, K, c0, lag)

    mle = curve_fit(objective1, ts, ys, p0=[0.02, 2, 0.01], bounds=([0, 0, 0], [0.1, 3, 0.1]))[0]
    r, K, c0 = mle[:3]

    df = pd.DataFrame(mle, columns=["MLE"])
    df.insert(0, "Names", ["r", "K", "c0"])
    print(df)

    ax = plt.subplot()
    ax.scatter(ts, ys, c="k", s=2, label="Data")
    ax.plot(ts, model(ts, r, K, c0, lag), c="r", label="Model")
    plt.legend()
    plot_figure("plot3_lag_fixed")
    # Save the data

    X = np.stack([ts, ys]).T
    np.savetxt("logistic_example_1.csv", X, delimiter=",")

    # Evaluate the objective function over a patch of values in the 2d subspace spanned by c0 and lag.

    n = 31
    c0s = np.logspace(-5, -1, n)
    lags = np.linspace(0, 300, n)
    err = np.zeros((n, n))
    for i, c0 in enumerate(c0s):
        for j, lag in enumerate(lags):
            err[i][j] = np.linalg.norm(ys - model(ts, r_true, K_true, c0, lag))

    # Plot the surface, and compare with the true optimum

    plt.contourf(c0s, lags, err, 20)
    plt.scatter([c0_true], [lag_true], c="r", label="True parameters")
    plt.xscale("log")
    plt.xlabel("c0")
    plt.ylabel("lag")
    plt.colorbar(label="Objective")
    plt.legend()
    plot_figure("plot4_surface")
