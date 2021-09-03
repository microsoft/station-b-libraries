# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Module for determining the input values that maximise the mean function of a BayesOptModel.

Exports:
    compute_optima
"""

from typing import List

import pandas as pd
from abex import optim
from abex.bayesopt import BayesOptModel
from abex.constants import FOLD
from abex.dataset import Dataset
from abex.settings import OptimizerConfig
from abex.transforms import InvertibleTransform


def compute_optima(models: List[BayesOptModel], dataset: Dataset, config: OptimizerConfig) -> None:  # pragma: no cover
    """
    Evaluate the optima for a list of models

    Args:
        models: A list of BayesOptModels to evaluate optima for
        dataset: The Dataset used for determining the initial point for the optimizer (the optimal point within the
         dataset will be used as the initial point for optimization)
        config: Specifies the details of optimization (method and num. optimizations)
    """
    optx, opty = optim.evaluate_models_optima(models, dataset, config)
    optima = pd.DataFrame(
        columns=(
            [FOLD]
            + dataset.transformed_cont_input_names
            + (dataset.transformed_categ_input_names if dataset.n_categorical_inputs else [])
            + [dataset.transformed_output_name]
        )
    )
    for fold in range(len(models)):
        # Iterate over all contexts, or just iterate once with context=None if there are no categ. vars in dataset
        for i, context in enumerate(dataset.unique_context_generator_or_none()):
            # Make another row for the dataframe of optima
            optima_df_row = pd.DataFrame({FOLD: [fold]})
            optima_df_row[dataset.transformed_cont_input_names] = optx[fold, i]
            if context:
                optima_df_row[dataset.transformed_categ_input_names] = context
            optima_df_row[dataset.transformed_output_name] = opty[fold, i]
            optima = optima.append(optima_df_row)
    if isinstance(dataset.preprocessing_transform, InvertibleTransform):
        optima_original_space = dataset.preprocessing_transform.backward(optima)
        optim.plot_optima(optima_original_space, config.data.categorical_inputs)
