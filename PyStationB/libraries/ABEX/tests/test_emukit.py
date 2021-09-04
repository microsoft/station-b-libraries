# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import mock
import numpy as np
import pytest
from abex import settings
from abex.bayesopt import BayesOptModel
from abex.dataset import Dataset
from psbutils.misc import find_subrepo_directory


@pytest.fixture
def test_config():
    yml_path = find_subrepo_directory() / "tests/data/Specs/tutorial-intro.yml"
    config = settings.simple_load_config(yml_path)
    config.seed = 0
    return config


@pytest.mark.skip("Configuration needs streamlining")
@pytest.mark.timeout(60)
def test_model_generate_batch(test_config):
    dataset = Dataset.from_data_settings(test_config.data)
    model = BayesOptModel.from_config(test_config, dataset)
    model.run()

    # Generate experiment
    test_config.bayesopt.batch = 1
    expt_batch = model.generate_batch(test_config.bayesopt.batch, num_samples=10, num_anchors=3)
    assert expt_batch.shape[0] == test_config.bayesopt.batch

    test_config.bayesopt.batch = 3
    expt_single = model.generate_batch(batch_size=test_config.bayesopt.batch, num_samples=10, num_anchors=3)
    assert expt_single.shape[0] == test_config.bayesopt.batch


def test_model_generate_batch__with_optima(test_config):
    dataset = Dataset.from_data_settings(test_config.data)
    model = BayesOptModel.from_config(test_config, dataset)
    model.run()

    optimum = 100.0
    # Generate experiment
    test_config.bayesopt.batch = 3
    with mock.patch("abex.bayesopt.LocalPenalizationPointCalculator") as local_pen:
        local_pen_instance = local_pen.return_value
        local_pen_instance.compute_next_points.side_effect = (np.ones([test_config.bayesopt.batch, dataset.ndims]),)
        model.generate_batch(test_config.bayesopt.batch, num_samples=10, num_anchors=3, optimum=optimum)
        local_pen.assert_called_once()
        print(local_pen.call_args)
        assert local_pen.call_args.kwargs["fixed_minimum"] == -optimum
