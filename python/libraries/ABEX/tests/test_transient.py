# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path

import abex.data_settings
import pytest
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

# flake8: noqa: E402
from abex import plotting, settings
from abex.bayesopt import BayesOptModel
from abex.dataset import Dataset
from psbutils.misc import find_subrepo_directory


@pytest.mark.skip("Configuration needs streamlining")
@pytest.mark.timeout(30)
def test_train_test_scaleonly(tmp_path_factory):
    yml_path = find_subrepo_directory() / "tests/data/Specs/Transient_1to5.yml"
    config = settings.simple_load_config(yml_path)
    config.seed = 0
    # Create results in a temporary directory
    results_dir: Path = tmp_path_factory.mktemp("test_transient")
    config.results_dir = results_dir

    train_test(config)


def load_data(config):
    dataset = Dataset.from_data_settings(config.data)
    if config.data.zero == 0.0:
        n_data = {
            "190222_DoE01.csv": 65,
            "190222_DoE02.csv": 150,
            "190222_DoE03.csv": 25,
            "190222_DoE04_NoDuc.csv": 122,
            "190222_DoE04_LowDuc.csv": 120,
            "190222_DoE04_MidDuc.csv": 133,
            "190222_DoE04_HighDuc.csv": 132,
            "190222_DoE04_5.csv": 52,
            "190222_DoE05.csv": 150,
            "190222_DoE1to5.csv": 949,
            "181201_DoE67.csv": 69,
        }
    else:
        n_data = {
            "190222_DoE01.csv": 90,
            "190222_DoE02.csv": 150,
            "190222_DoE03.csv": 92,
            "190222_DoE04_NoDuc.csv": 136,
            "190222_DoE04_LowDuc.csv": 136,
            "190222_DoE04_MidDuc.csv": 136,
            "190222_DoE04_HighDuc.csv": 136,
            "190222_DoE04_5.csv": 54,
            "190222_DoE05.csv": 150,
            "190222_DoE1to5.csv": 1080,
            "181201_DoE67.csv": 69,
        }
    expected_n_data = 0
    for _, f in config.data.files.items():
        expected_n_data += n_data[f]
    assert len(dataset) == expected_n_data
    return dataset


def train_test(config, seed=0):
    """Test a train-test split"""
    dataset = load_data(config)
    config.seed = seed
    # Train against 80%, saving 20% for test
    test_chunks = dataset.split_folds(5, seed=seed)
    model = BayesOptModel.from_config(config, dataset, test_chunks[0])
    model.run()
    models = [model]
    # f = pp.figure()
    ax = plt.subplot()
    r = plotting.plot_predictions_against_observed(
        ax, models=models, datasets=[model.test for model in models], title="Cross-validation"  # type: ignore # auto
    )
    plt.close()
    assert r is not float("nan")
