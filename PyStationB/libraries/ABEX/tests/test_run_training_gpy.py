# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import itertools
from typing import Dict, Union

import matplotlib
import pytest
from pandas import DataFrame

matplotlib.use("Agg")
# flake8: noqa: E402
from emukit.core.acquisition import IntegratedHyperParameterAcquisition
from emukit.model_wrappers import GPyModelWrapper

from abex.dataset import Dataset  # noqa: E402
from abex.data_settings import DataSettings, ParameterConfig, InputParameterConfig
from abex.bayesopt import BayesOptModel, HMCBayesOptModel
from abex.settings import (
    OptimizerConfig,
    TrainSettings,
    ModelSettings,
    BayesOptSettings,
    KernelFunction,
)


@pytest.fixture
def mock_dataframe() -> DataFrame:
    records = [
        (2500, 3, 2, 0.4, 0, 500000, 1, "96 Well", 1.40e04),
        (500, 10, 20, 0, 11, 500000, 1, "96 Well", 4.60e04),
        (2500, 3, 2, 0, 11, 500000, 1, "96 Well", 1.90e04),
        (1500, 6.5, 11, 0.2, 5.5, 1250000, 1, "96 Well", 0.00e00),
        (500, 10, 20, 0.4, 0, 2000000, 1, "96 Well", 1.20e04),
        (1500, 6.5, 11, 0.3, 5.5, 1250000, 1, "96 Well", 1.80e05),
        (1500, 6.5, 15.5, 0.2, 5.5, 1250000, 1, "96 Well", 1.50e05),
    ]
    cols = ["Gene", "Reagent1", "Reagent2", "Reagent3", "Reagent4", "Cell Density", "Gene ID", "Scale", "Titre"]
    return DataFrame.from_records(records, columns=cols)


@pytest.fixture(scope="module")
def mock_gpy_config(tmp_path_factory):
    results_dir = tmp_path_factory.mktemp("test_hmc_gpy")

    config = OptimizerConfig(
        data=mock_dataset(),
        training=mock_training(),
        model=mock_model(),
        bayesopt=mock_bayesopt(),
        results_dir=results_dir,
        seed=0,
    )

    return config


def mock_dataset() -> DataSettings:
    inputs_settings = {
        "Gene": InputParameterConfig(),
        "Reagent1": InputParameterConfig(default_condition=0.0),
        "Reagent2": InputParameterConfig(default_condition=10.0),
        "Reagent3": InputParameterConfig(default_condition=0.0),
        "Reagent4": InputParameterConfig(default_condition=0.0),
        "Cell Density": InputParameterConfig(default_condition=0.0),
    }

    params = ParameterConfig(log_transform=True, offset=1e-4)

    categorical_filters = {"Scale": ["96 Well", "Shake Flask"]}
    files: Dict[str, str] = {}
    return DataSettings(
        inputs=inputs_settings,
        output_column="Titre",
        categorical_inputs=["Gene ID", "Scale"],
        categorical_filters=categorical_filters,
        files=files,
        output_settings=params,
    )


def mock_training() -> TrainSettings:
    return TrainSettings(plot_slice1d=False, hmc=True, iterations=5)


def mock_model() -> ModelSettings:
    return ModelSettings()


def mock_bayesopt() -> BayesOptSettings:
    context: Dict[str, Union[float, str]] = {"Reagent3": 0.0, "Scale": "96 Well"}
    return BayesOptSettings(batch=1, context=context)


@pytest.mark.timeout(60)
def test_gpy(mock_gpy_config: OptimizerConfig, mock_dataframe: DataFrame) -> None:
    dataset = Dataset(mock_dataframe, mock_gpy_config.data)
    model = HMCBayesOptModel.from_config(mock_gpy_config, dataset)

    assert model.model_settings == mock_gpy_config.model
    assert model.train == dataset
    assert model.test is None  # by default, test_ids=None therefore model.test=None
    assert model.fold_id == 0

    # run training
    model.run()
    assert isinstance(model.emukit_model, GPyModelWrapper)
    assert isinstance(model.acquisition, IntegratedHyperParameterAcquisition)
    assert model.acquisition.n_samples == int(mock_gpy_config.training.iterations / 5)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("is_hmc", [True, False])
def test_bayesoptmodel_parameter_retrieval(
    is_hmc: bool, mock_gpy_config: OptimizerConfig, mock_dataframe: DataFrame
) -> None:
    # Ensure the model config specifies a single RBF kernel model with no bias
    mock_gpy_config.model.add_bias = False
    mock_gpy_config.model.kernel = KernelFunction.RBF.name
    mock_gpy_config.training.hmc = is_hmc

    dataset = Dataset(mock_dataframe, mock_gpy_config.data)
    if is_hmc:
        model = HMCBayesOptModel.from_config(mock_gpy_config, dataset)
    else:
        model = BayesOptModel.from_config(mock_gpy_config, dataset)

    model.run()

    # Try retrieving without log-likelihood
    param_df = model.get_model_parameters()
    if not is_hmc:
        # Only one row expected if is_hmc
        assert len(param_df) == 1
    else:
        # Otherwise number of rows equals number HMC samples
        assert len(param_df) == model.acquisition.n_samples  # type: ignore # auto
    assert len(param_df.columns) == 3
    assert "rbf.variance" in param_df.columns
    assert "rbf.lengthscale" in param_df.columns
    assert "Gaussian_noise.variance" in param_df.columns
    # Try retrieving with log-likelihood
    param_df = model.get_model_parameters_and_log_likelihoods()
    if not is_hmc:
        # Only one row expected if is_hmc
        assert len(param_df) == 1
    else:
        # Otherwise number of rows equals number HMC samples
        assert len(param_df) == model.acquisition.n_samples  # type: ignore # auto
    assert len(param_df.columns) == 4
    assert "rbf.variance" in param_df.columns
    assert "rbf.lengthscale" in param_df.columns
    assert "Gaussian_noise.variance" in param_df.columns


@pytest.mark.timeout(15)
@pytest.mark.parametrize("is_hmc, include_fixed", list(itertools.product([True, False], [True, False])))
def test_bayesoptmodel_parameter_retrieval_with_fixed_params(
    is_hmc: bool, include_fixed: bool, mock_gpy_config: OptimizerConfig, mock_dataframe: DataFrame
) -> None:
    # Ensure the model config specifies a single RBF kernel model with no bias
    mock_gpy_config.model.add_bias = False
    mock_gpy_config.model.kernel = KernelFunction.RBF.name
    mock_gpy_config.training.hmc = is_hmc
    # Constrain Gaussian variance
    mock_gpy_config.model.fixed_hyperparameters = {"Gaussian_noise.variance": 1e-5}

    dataset = Dataset(mock_dataframe, mock_gpy_config.data)
    if is_hmc:
        model = HMCBayesOptModel.from_config(mock_gpy_config, dataset)
    else:
        model = BayesOptModel.from_config(mock_gpy_config, dataset)

    model.run()

    # Try retrieving without log-likelihood
    param_df = model.get_model_parameters(include_fixed=include_fixed)
    if not is_hmc:
        # Only one row expected if is_hmc
        assert len(param_df) == 1
    else:
        # Otherwise number of rows equals number HMC samples
        assert len(param_df) == model.acquisition.n_samples  # type: ignore # auto
    assert len(param_df.columns) == 2 + int(include_fixed)
    assert "rbf.variance" in param_df.columns
    assert "rbf.lengthscale" in param_df.columns
    if include_fixed:
        assert "Gaussian_noise.variance" in param_df.columns
    # Try retrieving with log-likelihood
    param_df = model.get_model_parameters_and_log_likelihoods(include_fixed=include_fixed)
    if not is_hmc:
        # Only one row expected if is_hmc
        assert len(param_df) == 1
    else:
        # Otherwise number of rows equals number HMC samples
        assert len(param_df) == model.acquisition.n_samples  # type: ignore # auto
    assert len(param_df.columns) == 3 + int(include_fixed)
    assert "rbf.variance" in param_df.columns
    assert "rbf.lengthscale" in param_df.columns
    if include_fixed:
        assert "Gaussian_noise.variance" in param_df.columns
