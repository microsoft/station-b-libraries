#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------

# # Problems with initial cell density
#
# In this notebook, we will quickly infer the initial density for each well in one experimental plate.
# Then we will see whether our methods can recover that the initial
# density is similar in each well.
#
# Let's start by loading a data set.

# In[ ]:


from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
import staticchar as ch
from psbutils.filecheck import Plottable, figure_found
from psbutils.misc import find_subrepo_directory

SUBREPO_DIR = find_subrepo_directory()


def plot_figure(name: str, ax: Optional[Plottable] = None) -> None:
    sns.despine()
    assert figure_found(ax, f"test_problems_with_initial_density/{name}")
    plt.clf()


@pytest.mark.timeout(60)
def test_problems_with_initial_density():

    dataset = ch.datasets.Dataset(SUBREPO_DIR / "tests/test_data/SignificantDeath")  # type: ignore # auto

    # Now we will fit a growth model to each well and visualise them.
    #
    # Note that for some wells we won't be able to fit the growth model,
    # so we should catch the exceptions.

    # df = dataset.get_a_frame(0)
    df = ch.subtract_background(dataset.get_a_frame(0), columns=["OD"], strategy=ch.BackgroundChoices.FirstMeasurement)

    params = ch.GompertzModel.fit(df[ch.TIME], df["OD"])  # type: ignore # auto
    print(params)

    model = ch.GompertzModel(params)
    ch.plot_growth_model(df[ch.TIME], df["OD"], model=model)  # type: ignore # auto
    plot_figure("plot1_gompertz")

    # Get the figure to visualise the growth models. We choose 2 rows and 4 columns because there are 7 datasets.
    n_rows = 2
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6), sharex=False, sharey=False)

    log_initial_density = []

    for (name, df), ax in zip(sorted(dataset.items()), axs.ravel()):
        df = ch.subtract_background(df, columns=["OD"], strategy=ch.BackgroundChoices.FirstMeasurement)
        try:
            params = ch.LogisticModel.fit(df[ch.TIME], df["OD"])  # type: ignore # auto
            model = ch.LogisticModel(params)
            # params = ch.GompertzModel.fit(df[ch.TIME], df["OD"])
            # model = ch.GompertzModel(params)
            # Append inferred density to the list
            log_initial_density.append(model.initial_density(log=True))
        except RuntimeError:
            model = None

        ch.plot_growth_model(df[ch.TIME], df["OD"], model=model, ax=ax)  # type: ignore # auto

        if model is not None:
            ax.set_title(f"{name}")

    fig.tight_layout()

    plot_figure("plot2_growth_models")

    p = ch.models.gompertz._reparametrize_to_model(model.parameters)

    np.log(p.a) - p.b

    model.initial_density(log=True)

    # Now let's take a look at the initial density.

    plt.hist(log_initial_density)  # type: ignore # auto
    plt.title("Logarithm of inferred initial density")
    plot_figure("plot3_density")

    # This range is about $e^{17}$, that is over 7 orders of magnitude! This suggests that:
    #
    # 1. There is a mistake in calculation of initial cell density. (E.g. we can compare it
    #    with the numerical value).
    # 2. The three-parameter model with this parametrization does not properly capture the
    #    behaviour of this system.
