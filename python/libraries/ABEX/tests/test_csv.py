# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import Optional
from pathlib import Path

import pytest
import matplotlib.pyplot as plt

plt.switch_backend("Agg")
# flake8: noqa: E402
from abex.compute_optima import compute_optima
from abex import plotting, settings
from abex.bayesopt import BayesOptModel
from abex.compute_optima import compute_optima
from abex.dataset import Dataset
from abex.optimizers.bayes_optimizer import BayesOptimizer
from psbutils.misc import find_subrepo_directory


@pytest.fixture
def load_config(tmp_path_factory):
    """Returns a callable function that can be called to create a config pointing to a temporary results
    directory with a name given by the argument.
    """

    def _load_config(name: str):
        config_path = find_subrepo_directory() / "tests/data/Specs/tutorial-intro.yml"
        config: settings.OptimizerConfig = settings.simple_load_config(config_path)

        config.seed = 0
        config.results_dir = tmp_path_factory.mktemp(name)

        return config

    return _load_config


@pytest.mark.skip("Configuration needs streamlining")
@pytest.mark.timeout(120)
def test_run_traintest(load_config):  # noqa: 811
    config = load_config("test_run_traintest")
    BayesOptimizer(config).run()


@pytest.mark.skip("Configuration needs streamlining")
@pytest.mark.timeout(30)
def test_run_trainonly(load_config):  # noqa: 811
    config = load_config("test_run_trainonly")
    config.training.num_folds = 1
    BayesOptimizer(config).run()
