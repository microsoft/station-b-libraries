# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Locate the position of the optima of the output mean for a (single-output) GP model.

Usages:
    - The optima are used as a reference point in batch Bayesian optimization (local penalization).
    - The optima are useful for analysing differences across a multiplicity of models, and so the methods here are
      called by providing a list of models. Such lists usually originate from cross-validation. Function
      ``plot_optima`` provides a visual comparison.
"""
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns
from abex.bayesopt import BayesOptModel
from abex.constants import FOLD, SLSQP
from abex.dataset import Dataset
from abex.settings import OptimizerConfig
from scipy.optimize import Bounds, minimize


def model_output_mean(
    x: np.ndarray, model: BayesOptModel, onehot: Iterable[float] = None
) -> np.ndarray:  # pragma: no cover
    """Evaluate the model mean for given continuous inputs (``x``) and categorical inputs (``onehot``).

    Returns:
        the mean of the GP model at the requested point (``x``)
    """
    if onehot is None:
        onehot = []  # pragma: no cover
    xf = np.append(x, onehot)
    y, _ = model.minus_predict(xf[np.newaxis, :])
    return -y


def robust_optimum(
    model: BayesOptModel,
    bounds: Bounds,
    context_onehot: Optional[Iterable[float]] = None,
    N_sample: int = 100,
    manual_x0: Optional[np.ndarray] = None,
    method: str = SLSQP,
) -> Tuple[np.ndarray, float]:  # pragma: no cover
    """Find the optimum of the model mean prediction, i.e. the maximum value within bounds, conditioned on a given
    context. The function tries to find the minimum starting from N_sample different random initialisation (and
    possibly from manual_x0 inital point if given), and returns the best of the local optima found by the optimization
    algorithm.

    Returns:
        Tuple of:
            np.ndarray of values of continuous inputs at optimum with shape [n_continuous_inputs]
            objective at optimum (float)
    """
    if context_onehot is None:
        context_onehot = []  # pragma: no cover
    # noinspection PyUnresolvedReferences
    # Generate N_sample initial points for optimization
    x0s = np.array(
        [
            bounds.lb + (bounds.ub - bounds.lb) * np.random.random(len(model.continuous_parameters))  # type: ignore
            for _ in range(N_sample)
        ]
    )
    if manual_x0 is not None:
        x0s = np.append(x0s, manual_x0, axis=0)
        N_sample += 1
    optima_x = np.zeros((N_sample, len(model.continuous_parameters)))
    optima_y = np.zeros(N_sample)
    for i, x0 in enumerate(x0s):
        opt = minimize(lambda x: model_output_mean(x, model, context_onehot), x0, method=method, bounds=bounds)
        logging.debug(f"{-np.round(opt.fun, 3)}")
        optima_y[i] = -opt.fun
        optima_x[i] = opt.x
    # Return the best iteration found from (possibly) multiple random inits.
    best_loc = np.argmax(optima_y)
    # noinspection PyUnboundLocalVariable
    return optima_x[best_loc], optima_y[best_loc]


def evaluate_model_optima(
    model: BayesOptModel, dataset: Dataset, config: OptimizerConfig
) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """Find the maximum of the model's mean prediction for each possible categorical context (i.e. for each possible
    combination of the categorical inputs).

    Args:
        model: [description]
        dataset: Dataset, used for determining the initial point for the optimizer (the optimal point within the dataset
            will be used as the initial point for optimization)
        config: Specifies the details of optimization (method and num. optimizations)
        verbose (optional): Verbosity of logging. Defaults to False.

    Returns:
        Tuple of: 1) array of input locations of the optima for each categorical context, and 2) array of corresponding
            predicted outputs
    """

    # Compute optima for each possible categorical context, or just one set of optima if there are no contexts
    n_cases = max(dataset.n_unique_contexts, 1)
    optx = np.zeros((n_cases, dataset.n_continuous_inputs))
    opty = np.zeros((n_cases))

    bound_array = np.array(list(model.continuous_parameters.values()))
    bounds = Bounds(bound_array[:, 0], bound_array[:, 1])
    if dataset.n_unique_contexts > 0:
        for i, context in enumerate(dataset.unique_context_generator()):
            # Get the onehot-transformed representation of the context
            onehot = dataset.categ_context_to_onehot(context)
            # Get the idxs of examples for which categorical variables match the context
            dlocs = np.where((dataset.categorical_inputs_df == context).all(axis=1).values)[0]
            mx0 = _optimum_input_in_dataset(dataset, dlocs) if len(dlocs) > 0 else None
            # Find the optimum for this model, with this context
            optx[i, :], opty[i] = robust_optimum(
                model,
                bounds,
                context_onehot=onehot,
                manual_x0=mx0,
                method=config.training.optim_method,
                N_sample=config.training.optim_samples,
            )
    else:
        optx[0, :], opty[0] = robust_optimum(  # pragma: no cover
            model,
            bounds,
            context_onehot=None,
            manual_x0=_optimum_input_in_dataset(dataset),
            method=config.training.optim_method,
            N_sample=config.training.optim_samples,
        )
    return optx, opty


def evaluate_models_optima(
    models: List[BayesOptModel], dataset: Dataset, config: OptimizerConfig
) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """A for-loop wrapper around evaluate_model_optima to evaluate the optima for a list of models.

    Returns a tuple of arrays of shape [num_models, num_contexts, num_continuous_inputs] and [num_models, num_contexts].
    The former stores the optima locations for all the models, and for all possible categorical contexts
    (combinations of categ. variables), and the latter stores the predicted outputs for all models and all contexts.
    """
    optx_models = []
    opty_models = []
    for model in models:
        optx, opty = evaluate_model_optima(model, dataset, config)
        optx_models.append(optx)
        opty_models.append(opty)
    return np.stack(optx_models, axis=0), np.stack(opty_models, axis=0)


def _optimum_input_in_dataset(dataset: Dataset, dlocs: Optional[np.ndarray] = None):  # pragma: no cover
    """Find the optimum point (the one with highest objective) from among examples indexed by dlocs (or among all
    examples if dlocs not specified)
    """
    # If dlocs not specified, find the best input from entire dataset
    if dlocs is None:
        dlocs = np.arange(len(dataset))  # pragma: no cover
    x = dataset.inputs_array[dlocs, :]
    # Get the idx of the example with maximum objective values
    max_loc = np.argmax(dataset.output_array[dlocs])
    x_optim = np.atleast_2d(x[max_loc, : dataset.n_continuous_inputs])
    return x_optim


def plot_optima(
    optima_df: pd.DataFrame,
    categorical_input_cols: Optional[List[str]] = None,
    ncols: int = 3,
    height: float = 3.5,
    aspect: float = 1.2,
    fname: Optional[Path] = None,
) -> sns.FacetGrid:  # pragma: no cover
    """
    Plots for optimal values of input variables, separate the optima based on categorical variable context
    - optima_df is a pandas dataframe with rows being the optima found for each categorical context (opt: for each fold)
    - config is a OptimizerConfig
    """
    assert len(optima_df) > 0
    categorical_input_cols = categorical_input_cols if categorical_input_cols is not None else []
    id_vars = [FOLD] + categorical_input_cols
    feature = "Feature"
    optimal_value = "Optimal value"
    melted = optima_df.melt(id_vars=id_vars, var_name=feature, value_name=optimal_value)
    n_cat_inputs = len(categorical_input_cols)
    hue = categorical_input_cols[1] if n_cat_inputs > 1 else None
    # If any categorical inputs given, vary categ. var. along x-axis. Otherwise vary fold.
    x = FOLD if n_cat_inputs == 0 else categorical_input_cols[0]
    g = sns.catplot(
        x=x,
        y=optimal_value,
        hue=hue,
        col=feature,
        col_wrap=ncols,
        sharey=False,
        data=melted,
        height=height,
        aspect=aspect,
    )
    if fname is not None:  # pragma: no cover
        g.savefig(fname, bbox_inches="tight")
        # noinspection PyArgumentList
        plt.close()
    return g
