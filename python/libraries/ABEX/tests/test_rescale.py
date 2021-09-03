# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path

import abex.data_settings
import matplotlib.pyplot as plt
import pytest

plt.switch_backend("Agg")
# flake8: noqa: E402
from abex import plotting, settings
from abex.bayesopt import BayesOptModel
from abex.dataset import Dataset
from psbutils.misc import find_subrepo_directory


@pytest.mark.skip("Configuration file not available")
@pytest.mark.timeout(20)
def test_rescale_input():
    yml_path = find_subrepo_directory() / "tests/data/Specs/Rescaled.yml"
    config = settings.simple_load_config(yml_path)
    config.seed = 0
    dataset = Dataset.from_data_settings(config.data)
    model = BayesOptModel.from_config(config, dataset)
    model.run()
    for cat in config.data.categorical_inputs:
        # noinspection PyUnresolvedReferences
        ax = plt.subplot()
        plotting.plot_predictions_against_observed(
            ax=ax, models=[model], datasets=[model.train], category=cat, title="Train only"
        )
