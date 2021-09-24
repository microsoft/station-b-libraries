# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Deprecated.

This module implements an approximation to multi-point expected improvement using the moment-matching strategy
introduced in:

C.E. Clark, The Greatest of a Finite Set of Random Variables, Operations Research, Vol. 9, No. 2 (1961).

Note: The API in this module is out of date with the standard in gp.base.
"""
import numpy as np
from typing import Tuple, Dict, Optional

from emukit.core.acquisition import Acquisition
from emukit.core.loop import CandidatePointCalculator, LoopState
from emukit.core.optimization import AcquisitionOptimizerBase
from emukit.core import ParameterSpace
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions.expected_improvement import get_standard_normal_pdf_cdf
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement

from gp.moment_matching.numpy.moment_matching_minimum import (
    calculate_cumulative_min_moments,
    get_next_cumulative_min_moments,
)
from gp.numeric import batch_correlation_from_covariance


class MomentMatchingMultipointEI(Acquisition):
    """."""

    def __init__(self, model: GPyModelWrapper, prev_x: np.ndarray):  # Not necessary
        """
        :param model: Model
        :param prev_x: Inputs selected for the batch thus far
        """
        self.model = model
        self.prev_x = prev_x
        prev_y_mean, prev_y_cov = model.predict_with_full_covariance(self.prev_x)
        # Ensure right shape
        self.prev_y_mean = prev_y_mean.T  # Shape [1, prev_x.shape[0]]
        self.prev_y_cov = prev_y_cov[None, :]  # Shape [1, prev_x.shape[0], prev_x.shape[0]]
        prev_y_var = np.einsum("...ii->...i", self.prev_y_cov)  # Shape [1, prev_x.shape[0]]
        self.prev_y_std = np.sqrt(prev_y_var)
        self.prev_y_corr = batch_correlation_from_covariance(self.prev_y_cov, standard_deviations=self.prev_y_std)
        # Calculate the moments of the minima for previous batch points
        self.theta_means, self.theta_stds, self.alphas = calculate_cumulative_min_moments(
            means=self.prev_y_mean, stds=self.prev_y_std, corr_matrix=self.prev_y_corr
        )

    @property
    def _next_point_idx(self) -> int:
        return self.prev_y_mean.shape[1]

    @property
    def has_gradients(self) -> bool:
        return False

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the penalization function value. x is of shape [num_points, input_dim].
        """
        # Predict mean, variance, and correlation to previous batch points for the new point
        next_mean, next_variance = self.model.predict(x)
        next_std = np.sqrt(next_variance)
        covar_to_prev = self.model.get_covariance_between_points(x, self.prev_x)
        # covar_new is of shape [x.shape[0], prev_x.shape[0]]. Normalise each entry by the
        # std of corresponding observation at entries in x and prev_x
        corr_to_prev = covar_to_prev / (next_std * self.prev_y_std)
        corr_to_prev = np.clip(corr_to_prev, -1.0, 1.0)
        # TODO: Fill in 0. where covar_to_prev == 0

        next_theta_mean, next_theta_std, _, _ = get_next_cumulative_min_moments(
            next_output_idx=self._next_point_idx,
            mean=next_mean[:, 0],
            std=next_std[:, 0],
            prev_stds=self.prev_y_std,
            corr_to_next=corr_to_prev,
            theta_means=self.theta_means,
            theta_stds=self.theta_stds,
            alphas=self.alphas,  # TODO: This argument is not expected.
        )
        # Calculate EI of the total batch minimum with the suggested new points
        y_minimum = np.min(self.model.Y, axis=0)
        improvement = expected_improvement_from_mean_and_std(next_theta_mean, next_theta_std, y_minimum)
        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the penalization function value and gradients with respect to x
        """
        # TODO
        raise NotImplementedError()


def expected_improvement_from_mean_and_std(mean, standard_deviation, y_minimum):
    u, pdf, cdf = get_standard_normal_pdf_cdf(y_minimum, mean, standard_deviation)
    improvement = standard_deviation * (u * cdf + pdf)
    return improvement


class SequentialMomentMatchingEICalculator(CandidatePointCalculator):
    """
    Probability of Improvement insipred global penalization point calculator
    """

    def __init__(
        self,
        acquisition_optimizer: AcquisitionOptimizerBase,
        model: GPyModelWrapper,
        parameter_space: ParameterSpace,
        batch_size: int,
    ) -> None:
        """
        :param acquisition_optimizer: AcquisitionOptimizer object to optimize the penalized acquisition
        :param model: Model object, used to compute the parameters of the local penalization
        :param parameter_space: Parameter space describing input domain
        :param batch_size: Number of points to collect in each batch
        """
        self.acquisition_optimizer = acquisition_optimizer
        self.model = model
        self.parameter_space = parameter_space
        self.batch_size = batch_size

    def compute_next_points(self, loop_state: LoopState, context: Optional[Dict] = None) -> np.ndarray:
        """
        Computes a batch of points using local penalization.
        :param loop_state: Object containing the current state of the loop
        """
        # Compute first point:
        x1, _ = self.acquisition_optimizer.optimize(ExpectedImprovement(self.model))
        x_batch = [x1]

        # Compute the next points:
        for i in range(1, self.batch_size):
            mmei_acquisition = MomentMatchingMultipointEI(model=self.model, prev_x=np.concatenate(x_batch, axis=0))
            # Collect point
            x_next, _ = self.acquisition_optimizer.optimize(mmei_acquisition)
            x_batch.append(x_next)
        assert len(x_batch) == self.batch_size
        return np.concatenate(x_batch, axis=0)
