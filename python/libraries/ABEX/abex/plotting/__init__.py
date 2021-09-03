# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Exports:
    plot_prediction_slices2d - function for plotting 2d slices of model predictions through a given point
    plot_prediction_slices1d - function for plottin 1d slices of model prediction through a given point
    plot_pred_objective_for_batch - Plot the objective predicted by the model for a batch of inputs given
    plot_train_test_predictions - Plot the objective predicted by model against the actual objective on train/test data
    plot_predictions_against_observed - Plot the objective predicted by model against the actual objective on dataset
        given.
    experiment_distance - Make plot showing the distance between inputs in a batch of points
    hmc_samples - Make a plot visualising the HMC posterior samples for model parameters
    plot_convergence - Plot convergence of the cumulative best objective observations as a function of batches
        (for possibly multiple experiments).
    plot_multirun_convergence - Plot convergence of cumulative best observation for multiple experiments with multiple
        (randomly seeded) sub-runs each.
    plot_multirun_convergence_per_sample - Plot convergence of cumulative best, but as a function of samples observed.
    acquisition1d - Plot 1d slices of the acquistion function at a specified location.

"""
from abex.plotting.bayesopt_plotting import (
    plot_acquisition_slices,
    plot_prediction_slices2d,
    plot_prediction_slices1d,
    plot_pred_objective_for_batch,
    plot_train_test_predictions,
    plot_predictions_against_observed,
    experiment_distance,
    hmc_samples,
    acquisition1d,
    plot_gpy_priors,
    plot_calibration_curve,
)
from abex.plotting.convergence_plotting import (
    plot_convergence,
    plot_multirun_convergence,
    plot_multirun_convergence_per_sample,
)
from abex.plotting.composite_core import plot_multidimensional_function_slices, plot_slices1d_with_binned_data

__all__ = [
    "plot_acquisition_slices",
    "plot_prediction_slices2d",
    "plot_prediction_slices1d",
    "plot_pred_objective_for_batch",
    "plot_train_test_predictions",
    "plot_predictions_against_observed",
    "experiment_distance",
    "hmc_samples",
    "plot_convergence",
    "plot_multirun_convergence",
    "plot_multirun_convergence_per_sample",
    "plot_multidimensional_function_slices",
    "acquisition1d",
    "plot_gpy_priors",
    "plot_calibration_curve",
    "plot_slices1d_with_binned_data",
]
