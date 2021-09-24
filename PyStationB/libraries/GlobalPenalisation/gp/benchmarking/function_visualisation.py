# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def plot_2d_function(func, param_space, res=100, is_vectorized: bool = True, mark_minimum=False, mark_maximum=False):
    x1grid = np.linspace(*param_space.parameters[0].bounds[0], res)
    x2grid = np.linspace(*param_space.parameters[1].bounds[0], res)
    xx1, xx2 = np.meshgrid(x1grid, x2grid)
    xx = np.stack((xx1.ravel(), xx2.ravel()), axis=1)

    if is_vectorized:
        yy = func(xx)
    else:
        yy = np.zeros([xx.shape[0]])
        for i in range(len(yy)):
            yy[i] = func(xx[i][None, :])
    y = yy.reshape([res, res])

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # ax.contour(x1grid, x2grid, y, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax.contourf(x1grid, x2grid, y, levels=20, cmap="viridis")
    plt.colorbar(cntr1)

    if mark_minimum:
        ymin_idx = yy.argmin()
        xmin = xx[ymin_idx]
        plt.scatter(*xmin, marker="x", label="Minimum", c="red")
    if mark_maximum:
        ymax_idx = yy.argmax()
        xmax = xx[ymax_idx]
        plt.scatter(*xmax, marker="x", label="Maximum", c="red")

    ax.set_xlabel(param_space.parameter_names[0])
    ax.set_ylabel(param_space.parameter_names[1])
    return fig, ax


def plot_2d_additive_function(func, param_space, res=100):
    x1grid = np.linspace(*param_space.parameters[0].bounds[0], res)
    x2grid = np.linspace(*param_space.parameters[1].bounds[0], res)
    xx1, xx2 = np.meshgrid(x1grid, x2grid)
    xx = np.stack((xx1.ravel(), xx2.ravel()), axis=1)
    yy = func(xx)
    y = yy.reshape([res, res])

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 5], height_ratios=[5, 1], figure=fig)
    ax = plt.subplot(gs[0, 1])
    axl = plt.subplot(gs[0, 0], sharey=ax)
    axb = plt.subplot(gs[1, 1], sharex=ax)

    ax.contour(x1grid, x2grid, y, levels=14, linewidths=0.5, colors="k")
    ax.contourf(x1grid, x2grid, y, levels=14, cmap="RdBu_r")

    axl.plot(y[:, 0], x2grid)
    axb.plot(x1grid, y[0])
    axb.set_xlabel("x1")
    axl.set_ylabel("x2")
    return fig, (ax, axl, axb)


def visualise_gp(gpy_model, X, Y, param_space, res=100, annotate=False, axes=None):
    if axes is None:
        fig, axes = plt.subfigures(ncols=2)
    ax, ax_var = axes

    assert param_space.dimensionality == 2
    x1grid = np.linspace(*param_space.parameters[0].bounds[0], res)
    x2grid = np.linspace(*param_space.parameters[1].bounds[0], res)
    xx1, xx2 = np.meshgrid(x1grid, x2grid)
    xx = np.stack((xx1.ravel(), xx2.ravel()), axis=1)
    y_mean, y_var = gpy_model.predict(xx)
    y_mean = y_mean.reshape([res, res])
    y_var = y_var.reshape([res, res])
    y_std = np.sqrt(y_var)

    # 2D contour plot
    ax.contour(x1grid, x2grid, y_mean, levels=14, linewidths=0.5, colors="k")
    ax.contourf(x1grid, x2grid, y_mean, levels=14, cmap="RdBu_r")
    #  2D plot scatter
    ax.scatter(X[:, 0], X[:, 1], c="black")
    ax.set_title("Mean Model Prediction")

    # 2D contour plot - variance
    ax_var.contourf(x1grid, x2grid, y_std, levels=14, cmap="Greys")
    #  2D plot scatter
    ax_var.scatter(X[:, 0], X[:, 1], c="black")
    ax_var.set_title("Standard Deviation of Model Prediction")

    if annotate:
        for i in range(X.shape[0]):
            ax.annotate(str(i + 1), (X[i, 0], X[i, 1]), fontsize=15, xytext=(3, 3), textcoords="offset points")
            ax_var.annotate(str(i + 1), (X[i, 0], X[i, 1]), fontsize=15, xytext=(3, 3), textcoords="offset points")

    return fig, (ax, ax_var)


def visualise_additive_gp(gpy_model, X, Y, param_space, res=100, annotate=False):
    x1grid = np.linspace(*param_space.parameters[0].bounds[0], res)
    x2grid = np.linspace(*param_space.parameters[1].bounds[0], res)
    xx1, xx2 = np.meshgrid(x1grid, x2grid)
    xx = np.stack((xx1.ravel(), xx2.ravel()), axis=1)
    y_mean, y_var = gpy_model.predict(xx)
    y_mean = y_mean.reshape([res, res])
    y_var = y_var.reshape([res, res])
    y_std = np.sqrt(y_var)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 6, 6], height_ratios=[6, 1], figure=fig)
    ax = plt.subplot(gs[0, 1])
    ax_var = plt.subplot(gs[0, 2], sharey=ax, sharex=ax)
    axl = plt.subplot(gs[0, 0], sharey=ax)
    axb = plt.subplot(gs[1, 1], sharex=ax)

    # 2D contour plot
    ax.contour(x1grid, x2grid, y_mean, levels=14, linewidths=0.5, colors="k")
    ax.contourf(x1grid, x2grid, y_mean, levels=14, cmap="RdBu_r")
    #  2D plot scatter
    ax.scatter(X[:, 0], X[:, 1], c="black")
    ax.set_title("Mean Model Prediction")

    # 2D contour plot - variance
    ax_var.contourf(x1grid, x2grid, y_std, levels=14, cmap="Greys")
    #  2D plot scatter
    ax_var.scatter(X[:, 0], X[:, 1], c="black")
    ax_var.set_title("Standard Deviation of Model Prediction")

    if annotate:
        for i in range(X.shape[0]):
            ax.annotate(str(i + 1), (X[i, 0], X[i, 1]), fontsize=15, xytext=(3, 3), textcoords="offset points")
            ax_var.annotate(str(i + 1), (X[i, 0], X[i, 1]), fontsize=15, xytext=(3, 3), textcoords="offset points")

    # Side plots
    # Average std:
    y_std_avg1 = y_std.min(axis=1)
    axl.fill_betweenx(x2grid, y_mean[:, 0] - y_std_avg1, y_mean[:, 0] + y_std_avg1, alpha=0.3)

    y_std_avg0 = y_std.min(axis=0)
    axb.fill_between(x1grid, y_mean[0] - y_std_avg0, y_mean[0] + y_std_avg0, alpha=0.3)

    axl.plot(y_mean[:, 0], x2grid)
    axb.plot(x1grid, y_mean[0])

    axb.set_xlabel("x1")
    axl.set_ylabel("x2")
    return fig, (ax, ax_var, axl, axb)
