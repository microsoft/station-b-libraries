# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This module implements global penalization using the moment-matching strategy introduced in

C.E. Clark, The Greatest of a Finite Set of Random Variables, Operations Research, Vol. 9, No. 2 (1961).

TODO: The contents of this module should be moved into the emukit repository once complete.
TODO: There is also a more proper implementation with gradients worked on here:
https://dev.azure.com/msrcambridge/biocomputing/_git/global-penalisation
"""
from typing import Dict, Optional, Tuple

import numpy as np
from abex.emukit.integrated_hyperparam_from_samples import IntegratedHyperParameterAcquisitionFromSamples
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.acquisitions.expected_improvement import (
    MeanPluginExpectedImprovement,
    get_standard_normal_pdf_cdf,
)
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.loop import CandidatePointCalculator, LoopState
from emukit.core.optimization import AcquisitionOptimizerBase
from emukit.model_wrappers import GPyModelWrapper
from scipy.stats import norm  # type: ignore # auto


def calc_beta(std: np.ndarray, theta_std_prev: np.ndarray, output_to_prev_theta_corr: np.ndarray) -> np.ndarray:
    """Helper function to calculate :math:`\\beta` in the equations for moment approximation to :math:`\\max_i Y_i`.

    Args:
        std (np.ndarray): Standard deviations of Gaussian-distributed random variables Y_i
        theta_std_prev (np.ndarray): The standard deviations of theta_j=max_j Y_j computed so far.
        output_to_prev_theta_corr (np.ndarray): The vector of correlations between Gaussian random variable Y_i and
            theta_j=max_j Y_j for

    Returns:
        np.ndarray: The beta in equations for moment-matching.
    """
    beta = theta_std_prev ** 2 + std ** 2 - 2 * theta_std_prev * std * output_to_prev_theta_corr  # type: ignore
    # Ensure that beta >= 0. (sometimes it can become negative due to numerical errors)
    return np.clip(beta, 0.0, None)


def calc_alpha(mean, theta_mean_prev, beta):
    # Checks on shape
    assert theta_mean_prev.ndim == 1
    alpha_nominator = theta_mean_prev - mean
    # Account for the case when beta_i == 0
    alpha_limiting_case = np.where(alpha_nominator > 0.0, np.inf, -np.inf)
    alpha = np.where(beta > 0.0, alpha_nominator / np.sqrt(beta), alpha_limiting_case)
    return alpha


def calc_output_i_theta_j_corr(std, corr, theta_std, theta_prev_std, output_theta_prev_corr, alpha):
    """
    Computes the (approximate) correlation between output Y_i and theta_j (the cumulative minimum of outputs
    up to index j), where i > j.

    Args:
        std: Standard deviation of the Gaussian variable Y_j
        corr: Correlation between Gaussian variables Y_i and Y_j
        theta_std: Standard deviation of the cumulative minimum up to index j
        theta_prev_std: Standard deviation of the cumulative minimum up to index j - 1
        output_theta_prev_corr ([type]): Correlation between Gaussian variable Y_i and cumulative min. up
            to index j - 1
        alpha: Helper alpha variable for output with index j (see calc_alpha)

    Returns:
        np.ndarray: Approximate correlation between Gaussian variable Y_i and theta_j (the cumulative minimum up to
            index j)
    """
    nominator_term1 = theta_prev_std * output_theta_prev_corr * norm.cdf(-alpha)
    nominator_term2 = std * corr * norm.cdf(alpha)
    denom = theta_std
    output_i_theta_j_corr = (nominator_term1 + nominator_term2) / denom
    return np.clip(output_i_theta_j_corr, -1.0, 1.0)


def calc_theta_mean(mean, theta_mean_prev, alpha, beta):
    return theta_mean_prev * norm.cdf(-alpha) + mean * norm.cdf(alpha) - np.sqrt(beta) * norm.pdf(alpha)


def calc_theta_uncentered_2nd_moment(mean, std, theta_mean_prev, theta_std_prev, alpha, beta):
    term1 = (theta_mean_prev ** 2 + theta_std_prev ** 2) * norm.cdf(-alpha)
    term2 = (mean ** 2 + std ** 2) * norm.cdf(alpha)
    term3 = -(theta_mean_prev + mean) * np.sqrt(beta) * norm.pdf(alpha)
    theta_uncentered_2nd_moment = term1 + term2 + term3
    return theta_uncentered_2nd_moment


def calc_theta_std(mean, std, theta_mean, theta_mean_prev, theta_std_prev, alpha, beta):
    theta_uncentered_2nd_moment = calc_theta_uncentered_2nd_moment(
        mean=mean, std=std, theta_mean_prev=theta_mean_prev, theta_std_prev=theta_std_prev, alpha=alpha, beta=beta
    )
    theta_var = theta_uncentered_2nd_moment - theta_mean ** 2
    return np.sqrt(theta_var)


def get_next_cumulative_min_moments(
    next_output_idx: int,
    mean: np.ndarray,
    std: np.ndarray,
    prev_stds: np.ndarray,
    corr_to_next: np.ndarray,
    theta_means: np.ndarray,
    theta_stds: np.ndarray,
    alphas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For the next output Y_i, calculate the moments of cumulative minimum up to index i:
        theta_i = min_{j <= i} Y_j
    from the moments of cumulative minimum up to index i - 1.

    Can operate on a batch of new suggested outputs at index next_output_idx.

    Args:
        next_output_idx (int): Integer index of the next output to incorporate into the cumulative minimum. Equals
            the number of previous points (num_prev_points)
        mean (np.ndarray): Array of shape [N] with the means of the next output(s) Y_i
        std (np.ndarray): Array of shape [N] with the standard deviations of the next output(s) Y_i
        prev_stds (np.ndarray): Array of shape [N, num_prev_points] or [1, next_output_idx].
            If first dimension has size 1, then the batch of next outputs of size [N] is treated as N candidate point
            suggestions for the next point, and the cumulative minimum will be calculated with respect to the same
            single set of previous outputs.
            Otherwise, if first dimension has size N, the batch of next outputs are treated as independent outputs,
            and the cumulative minimum will be calculated with respect to a different set of previous outputs for each
            of the next outputs
        corr_to_next (np.ndarray): Array of shape [N, num_prev_points] where each element is the correlation of the
            next output Y_i to one of the previous ones.
        theta_means (np.ndarray): Sequence of arrays of shape [N] or [1] with previous cumulative minimum means
        theta_stds (np.ndarray): Sequence of arrays of shape [N] or [1] with previous cumulative standard deviations
        alphas (np.ndarray): Sequence of arrays of shape [N] or [1] with previous alpha parameters

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Returns four arrays
            - next_theta_mean: the mean of the cumulative minimum up to index next_output_idx of shape [num_points]
            - next_theta_std: the st. deviation of the cumulative min. up to index next_output_idx of shape [num_points]
            - new_output_theta_corr: Array of vectors representing the correlation between the next output
                and previous cumulative minima. Has shape [next_output_idx], and each element is a vector of
                correlations of shape [num_points]
            - alpha: Array of shape [num_points] of helper values alpha which are useful for later calculations
                of the moments.
    """
    assert mean.ndim == 1
    assert std.ndim == 1
    assert prev_stds.ndim == 2
    assert corr_to_next.ndim == 2
    # Assert the shapes are consistent
    assert len({mean.shape[0], std.shape[0], corr_to_next.shape[0]}) == 1

    if prev_stds.shape[0] == 1:
        # If only a single set of previous points given, and the cumulative minimum moments are to be calculated
        # for a batch of new candidate outputs:
        assert isinstance(theta_means[next_output_idx - 1], float) or theta_means[next_output_idx - 1].shape[0] == 1
        assert isinstance(theta_stds[next_output_idx - 1], float) or theta_stds[next_output_idx - 1].shape[0] == 1
    else:
        assert len({mean.shape[0], prev_stds.shape[0]}) == 1

    # Array for correlations between the next output and previous theta_j (cumulative min. up to index j)
    new_output_theta_corr = np.empty([next_output_idx], dtype=object)
    # First correlation entry is exact
    new_output_theta_corr[0] = corr_to_next[:, 0]
    for j in range(1, next_output_idx):  # Iterate over previous thetas
        new_output_theta_corr[j] = calc_output_i_theta_j_corr(
            std=prev_stds[:, j],
            corr=corr_to_next[:, j],
            theta_std=theta_stds[j],
            theta_prev_std=theta_stds[j - 1],
            output_theta_prev_corr=new_output_theta_corr[j - 1],
            alpha=alphas[j],
        )
    # Calculate alphas and betas
    beta = calc_beta(
        std=std,
        theta_std_prev=theta_stds[next_output_idx - 1],
        output_to_prev_theta_corr=new_output_theta_corr[next_output_idx - 1],
    )
    alpha = calc_alpha(mean=mean, theta_mean_prev=theta_means[next_output_idx - 1], beta=beta)
    # Calculate the moments
    next_theta_mean = calc_theta_mean(
        mean=mean, theta_mean_prev=theta_means[next_output_idx - 1], alpha=alpha, beta=beta
    )
    next_theta_std = calc_theta_std(
        mean=mean,
        std=std,
        theta_mean=next_theta_mean,
        theta_mean_prev=theta_means[next_output_idx - 1],
        theta_std_prev=theta_stds[next_output_idx - 1],
        alpha=alpha,
        beta=beta,
    )
    return next_theta_mean, next_theta_std, new_output_theta_corr, alpha


def calculate_cumulative_min_moments(means, stds, corr_matrix):
    """
    Calculate approximate moments of the cumulative minimum of multiple Gaussians:
        theta_j = min_{i <= j} Y_i
    where theta_j is the maximum of j Gaussian variables Y_i for i = 1, ..., j.
    """
    assert means.ndim == 2
    assert stds.ndim == 2
    assert corr_matrix.ndim == 3
    # Assert the shapes are consistent
    assert len({means.shape[1], stds.shape[1], corr_matrix.shape[1], corr_matrix.shape[2]}) == 1
    assert len({means.shape[0], stds.shape[0], corr_matrix.shape[0]}) == 1

    ndims = means.shape[1]
    # Define the arrays to iteratively compute (these are used as indexable variables)
    theta_means = np.empty([ndims], dtype=object)
    theta_stds = np.empty([ndims], dtype=object)
    alphas = np.empty([ndims], dtype=object)
    # First entries are exact
    theta_means[0] = means[:, 0]
    theta_stds[0] = stds[:, 0]
    for i in range(1, means.shape[1]):
        next_theta_mean, next_theta_std, next_output_theta_corr, next_alpha = get_next_cumulative_min_moments(
            next_output_idx=i,
            mean=means[:, i],
            std=stds[:, i],
            prev_stds=stds,
            corr_to_next=corr_matrix[:, :, i],
            theta_means=theta_means,
            theta_stds=theta_stds,
            alphas=alphas,
        )
        theta_means[i] = next_theta_mean
        theta_stds[i] = next_theta_std
        alphas[i] = next_alpha
    return theta_means, theta_stds, alphas


class MomentMatchingMultipointEI(Acquisition):  # pragma: no cover
    """."""

    def __init__(self, model: GPyModelWrapper, prev_x: np.ndarray, mean_plugin: bool = True):
        """
        :param model: Model
        :param prev_x: Inputs selected for the batch thus far
        """
        self.model = model
        self.prev_x = prev_x
        self.mean_plugin = mean_plugin

        prev_y_mean, prev_y_cov = model.model.predict(self.prev_x, full_cov=True, include_likelihood=not mean_plugin)
        # Ensure right shape
        self.prev_y_mean = prev_y_mean.T  # Shape [1, prev_x.shape[0]]
        self.prev_y_cov = prev_y_cov[None, :]  # Shape [1, prev_x.shape[0], prev_x.shape[0]]
        prev_y_var = np.einsum("...ii->...i", self.prev_y_cov)  # Shape [1, prev_x.shape[0]]
        self.prev_y_std = np.sqrt(prev_y_var)
        self.prev_y_corr = correlation_from_covariance(self.prev_y_cov, standard_deviations=self.prev_y_std)
        # Calculate the moments of the minima for previous batch points
        self.theta_means, self.theta_stds, self.alphas = calculate_cumulative_min_moments(
            means=self.prev_y_mean, stds=self.prev_y_std, corr_matrix=self.prev_y_corr
        )

    @property
    def _next_point_idx(self) -> int:  # pragma: no cover
        return self.prev_y_mean.shape[1]

    @property
    def has_gradients(self) -> bool:  # pragma: no cover
        return False

    def evaluate(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Evaluates the penalization function value. x is of shape [num_points, input_dim].
        """
        # Predict mean, variance, and correlation to previous batch points for the new point
        next_mean, next_variance = self.model.model.predict(x, include_likelihood=not self.mean_plugin)
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
            alphas=self.alphas,
        )
        # Calculate EI of the total batch minimum with the suggested new points
        y_minimum = self._get_y_minimum()
        improvement = expected_improvement_from_mean_and_std(next_theta_mean, next_theta_std, y_minimum)
        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        """
        Evaluates the penalization function value and gradients with respect to x
        """
        # TODO
        raise NotImplementedError()

    def _get_y_minimum(self) -> np.ndarray:
        if not self.mean_plugin:  # pragma: no cover
            return np.min(self.model.Y, axis=0)
        else:  # pragma: no cover
            means_at_prev, _ = self.model.model.predict(self.model.X, include_likelihood=False)
            return np.min(means_at_prev, axis=0)


def expected_improvement_from_mean_and_std(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    y_minimum: np.ndarray,
) -> np.ndarray:  # pragma: no cover
    """Return the expected improvement upon the current minimum (y_minimum) given a Gaussian-distributed random
    variables with means and standard deviations given by arrays mean and standard_deviation.
    """
    u, pdf, cdf = get_standard_normal_pdf_cdf(y_minimum, mean, standard_deviation)  # type: ignore # auto
    improvement = standard_deviation * (u * cdf + pdf)  # type: ignore # auto
    return improvement


class SequentialMomentMatchingEICalculator(CandidatePointCalculator):
    """
    Probability of Improvement inspired global penalization point calculator
    """

    def __init__(
        self,
        acquisition_optimizer: AcquisitionOptimizerBase,
        model: GPyModelWrapper,
        parameter_space: ParameterSpace,
        batch_size: int,
        mean_plugin: bool = True,
        hyperparam_samples: Optional[np.ndarray] = None,
    ) -> None:  # pragma: no cover
        """
        :param acquisition_optimizer: AcquisitionOptimizer object to optimize the penalized acquisition
        :param model: Model object, used to compute the parameters of the local penalization
        :param parameter_space: Parameter space describing input domain
        :param batch_size: Number of points to collect in each batch
        :param mean_plugin: Whether to use the mean plugin heuristic to deal with noisy functions
        :param hyperparam_samples: If given, those should be the HMC samples generated by
            model.generate_hyperparameters_samples(), and will be used to integrate out the acquisition
        """
        self.acquisition_optimizer = acquisition_optimizer
        self.model = model
        self.parameter_space = parameter_space
        self.batch_size = batch_size
        self.mean_plugin = mean_plugin
        self.hyperparam_samples = hyperparam_samples

    def compute_next_points(
        self, loop_state: LoopState, context: Optional[Dict] = None
    ) -> np.ndarray:  # pragma: no cover
        """
        Computes a batch of points using local penalization.
        :param loop_state: object containing the current state of the loop
        :param context: context variables to condition on
        """
        # Acquisition function for the first point is just standard single-point EI
        acquisition_generator = MeanPluginExpectedImprovement if self.mean_plugin else ExpectedImprovement
        x_batch = []

        # Compute the next points:
        for i in range(0, self.batch_size):
            if self.hyperparam_samples is not None:  # pragma: no cover
                acquisition = IntegratedHyperParameterAcquisitionFromSamples(
                    self.model, acquisition_generator, self.hyperparam_samples
                )
            else:
                acquisition = acquisition_generator(self.model)
            # Collect point
            x_next, _ = self.acquisition_optimizer.optimize(acquisition)
            x_batch.append(x_next)
            acquisition_generator = lambda model: MomentMatchingMultipointEI(  # noqa: E731
                model=model, prev_x=np.concatenate(x_batch, axis=0), mean_plugin=self.mean_plugin
            )
        assert len(x_batch) == self.batch_size
        return np.concatenate(x_batch, axis=0)  # pragma: no cover


def correlation_from_covariance(covariance: np.ndarray, standard_deviations: Optional[np.ndarray]) -> np.ndarray:
    """Compute the correlation matrices from an array of covariance matrices.

    Args:
        covariance (np.ndarray): Array of shape [num_cases, num_dim, num_dim] where (for every i)
        the covariance[i, :, :] matrix is a positive semi-definite covariance matrix for a multivariate Gaussian.
        standard_deviations: A [num_cases, num_dim] shaped array of standard deviations corresponding to the
            covariance matrices. This argument is optional if standard deviations have been precomputed before calling.

    Returns:
        np.ndarray: A [num_cases, num_dim, num_dim] shaped array of correlation matrices.
    """
    # stds_outer_product = standard_deviations[:, :, None] * standard_deviations[:, None, :]
    stds_outer_product = np.einsum("...i,...j->...ij", standard_deviations, standard_deviations)  # type: ignore
    correlation = covariance / stds_outer_product
    correlation[covariance == 0] = 0
    # Due to numerical errors, sometimes abs(correlation) > 1.0 by a narrow margin
    correlation = np.clip(correlation, -1.0, 1.0)
    return correlation
