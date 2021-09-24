# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import numpy as np
import torch
from typing import Optional, Tuple
import gp
from gp.moment_matching.pytorch.moment_matching_minimum_iterative import (
    get_next_cumulative_min_moments,
    approximate_minimum_with_prob_is_min,
    calc_output_to_cum_min_cov,
)
from gp.moment_matching.pytorch.moment_matching_minimum import approximate_minimum, stack_covariance_from_subcovariances
import gp.moment_matching.pytorch.numeric as numeric


class SequentialMomentMatchingEI(gp.base.SequentialMomentMatchingBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_y_mean: torch.Tensor = torch.tensor(self.selected_y_mean, dtype=torch.float64)
        self.selected_y_cov: torch.Tensor = torch.tensor(self.selected_y_cov, dtype=torch.float64)

        # Calculate and save the cumulative minima means and standard deviations
        (
            self.selected_cumulative_min_mean,
            self.selected_cumulative_min_var,
            self.selected_points_prob_is_min,
        ) = approximate_minimum_with_prob_is_min(
            means=self.selected_y_mean.T, covariance=self.selected_y_cov[None, :, :]
        )  # Shape (n_selected, 1) each

    def _evaluate_with_gradients(
        self, y_mean: np.ndarray, y_var: np.ndarray, cross_covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Get the minimum as observed so far
        y_min = self.model.Y.min()

        #  -- Transfer to Torch
        #  Map all arrays to torch tensors
        y_mean, y_var, cross_covariance = map(
            lambda x: torch.tensor(x, requires_grad=True, dtype=torch.float64), (y_mean, y_var, cross_covariance)
        )
        y_min = torch.tensor(y_min, dtype=torch.float64)

        # Compute the mean and variance of minimum
        min_mean, min_var = self._evaluate_cumulative_min_from_selected(
            y_mean=y_mean,
            y_var=y_var,
            cross_covariance=cross_covariance,
            selected_y_mean=self.selected_y_mean,
            selected_y_cov=self.selected_y_cov,
            selected_cumulative_min_mean=self.selected_cumulative_min_mean,
            selected_cumulative_min_var=self.selected_cumulative_min_var,
            selected_points_prob_is_min=self.selected_points_prob_is_min,
        )
        #  Compute standard deviation of min in a numerically safe way
        min_std = torch.sqrt(numeric.passthrough_clamp(min_var, 0.0, None))

        #  Calculate the Expected Improvement
        expected_improvement = numeric.expected_improvement(
            y_min=y_min,
            mean=min_mean,
            standard_deviation=min_std,
        )  #  Shape (num_candidates)

        # -- Get the gradients with respect to y_mean, y_std and covariance
        expected_improvement.sum().backward()
        # Partial derivative of expected improvement (ei) wrt. y_mean, y_std and covariance
        grad_y_mean = y_mean.grad.detach().numpy()
        grad_y_var = y_var.grad.detach().numpy()
        grad_cross_covariance = cross_covariance.grad.detach().numpy()
        expected_improvement = expected_improvement.detach().numpy()
        return expected_improvement[:, None], grad_y_mean, grad_y_var, grad_cross_covariance

    @staticmethod
    def _evaluate_cumulative_min_from_selected(
        y_mean: torch.Tensor,
        y_var: torch.Tensor,
        cross_covariance: torch.Tensor,
        selected_y_mean: torch.Tensor,
        selected_y_cov: torch.Tensor,
        selected_cumulative_min_mean: torch.Tensor,
        selected_cumulative_min_var: torch.Tensor,
        selected_points_prob_is_min: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        min_mean, min_var, *_ = get_next_cumulative_min_moments(
            mean=y_mean.flatten(),
            variance=y_var.flatten(),
            cov_to_next=cross_covariance,
            cum_min_mean=selected_cumulative_min_mean,
            cum_min_var=selected_cumulative_min_var,
            prob_is_min=selected_points_prob_is_min,
        )  # Shape (num_candidates) for both

        return min_mean, min_var


class SimultaneousMomentMatchingEI(gp.base.SimultaneousMomentMatchingBase):
    def _evaluate_with_gradients(
        self, y_mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """[summary]

        Args:
            y_mean: Array of means of shape (n_points, 1)
            covariance: Covariance of shape (n_points, n_points)

        Returns:
            Tuple[float, np.ndarray, np.ndarray]:
                float acquisition value for this batch of points.
                Gradient of the acquisition with respect to the passed means at the batch locations.
                Gradient of the acquisition with respect to the passed covariance between
                    points at the batch locations.
        """

        # Get the minimum as observed so far
        y_min: float = self.model.Y.min()

        #  -- Transfer to Torch
        #  Map all arrays to torch tensors
        y_mean, covariance = map(
            lambda x: torch.tensor(x, requires_grad=True, dtype=torch.float64), (y_mean, covariance)
        )
        y_min = torch.tensor(y_min, dtype=torch.float64)

        min_mean, min_var = approximate_minimum(means=y_mean.T, covariance=covariance[None, :, :])
        #  Compute standard deviation of min in a numerically safe way
        min_std = torch.sqrt(numeric.passthrough_clamp(min_var, 0.0, None))

        #  Calculate the Expected Improvement
        expected_improvement = numeric.expected_improvement(
            y_min=y_min,
            mean=min_mean,
            standard_deviation=min_std,
        )

        # -- Get the gradients with respect to y_mean and covariance
        expected_improvement.sum().backward()
        # Partial derivative of expected improvement (EI) wrt. y_mean and covariance
        grad_y_mean = y_mean.grad.detach().numpy()
        grad_covariance = covariance.grad.detach().numpy()
        expected_improvement = expected_improvement.detach().numpy()
        return float(expected_improvement), grad_y_mean, grad_covariance


class SimultaneousMomentMatchingExpectedMin(gp.base.SimultaneousMomentMatchingBase):
    def _evaluate_with_gradients(
        self, y_mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """An acquisition that minimises the expected minimum of the outputs. If used with a finite
        set of inputs on an infinite domain, this expectation will always be an upper bound
        on the true expected value of the function minimum.

        Since acquisitions are maximised, this returns -ve expected minimum.

        Args:
            y_mean: Array of means of shape (n_points, 1)
            covariance: Covariance of shape (n_points, n_points)

        Returns:
            Tuple[float, np.ndarray, np.ndarray]:
                float acquisition value for this batch of points.
                Gradient of the acquisition with respect to the passed means at the batch locations.
                Gradient of the acquisition with respect to the passed covariance between
                    points at the batch locations.
        """
        #  -- Transfer to Torch
        #  Map all arrays to torch tensors
        y_mean, covariance = map(
            lambda x: torch.tensor(x, requires_grad=True, dtype=torch.float64), (y_mean, covariance)
        )
        min_mean, min_var = approximate_minimum(means=y_mean.T, covariance=covariance[None, :, :])
        # The goal of this acquisition is to minimise the exp. minimum <-> maximise -ve exp. minimum
        neg_min_mean = -min_mean

        # -- Get the gradients with respect to y_mean and covariance
        neg_min_mean.sum().backward()
        # Partial derivative of expected improvement (EI) wrt. y_mean and covariance
        grad_y_mean = y_mean.grad.detach().numpy()
        # (There won't be any gradient wrt. covariance if the batch-size is one)
        grad_covariance = covariance.grad.detach().numpy() if covariance.shape[-1] > 1 else np.zeros(covariance.shape)
        neg_min_mean = neg_min_mean.detach().numpy()
        return float(neg_min_mean), grad_y_mean, grad_covariance


class SequentialMomentMatchingDecorrelatingEI(SequentialMomentMatchingEI):
    @staticmethod
    def _evaluate_cumulative_min_from_selected(
        y_mean: torch.Tensor,
        y_var: torch.Tensor,
        cross_covariance: torch.Tensor,
        selected_y_mean: torch.Tensor,
        selected_y_cov: torch.Tensor,
        selected_cumulative_min_mean: torch.Tensor,
        selected_cumulative_min_var: torch.Tensor,
        selected_points_prob_is_min: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        selected_y_vars = torch.diagonal(selected_y_cov, dim1=-2, dim2=-1)

        # Checks for numerical issues
        assert torch.all(y_var >= 0)
        assert torch.all(selected_y_vars >= 0)

        # Get the correlation from new points to selected:
        correlation = numeric.correlation_between_distinct_sets_from_covariance(
            covariance=cross_covariance,
            std1=torch.sqrt(y_var.flatten()),
            std2=torch.sqrt(selected_y_vars.flatten()),
        )  # Shape (n_candidates, n_selected)

        #  Find the point from among selected with highest correlation for each candidate point
        _, highest_corr_selected_idx = torch.max(correlation, dim=1)  # Shape (n_candidates)

        # Re-add the point with highest correlation to the approximation as a "dummy" point
        fake_point_mean = selected_y_mean[highest_corr_selected_idx]  # Shape (n_candidates, 1)
        fake_point_var = selected_y_vars[highest_corr_selected_idx]  # Shape (n_candidates, 1)

        cov_fake_point_to_selected = selected_y_cov[highest_corr_selected_idx, :]  #  Shape (num_candidates, n_selected)
        cov_candidates_to_fake_point = torch.gather(
            cross_covariance, dim=-1, index=highest_corr_selected_idx[:, None]
        )  #  Shape (num_candidates, 1)

        next_cum_min_mean, next_cum_min_var, _, new_alpha_pdf, new_alpha_cdf = get_next_cumulative_min_moments(
            mean=fake_point_mean.flatten(),
            variance=fake_point_var.flatten(),
            cov_to_next=cov_fake_point_to_selected,
            cum_min_mean=selected_cumulative_min_mean,
            cum_min_var=selected_cumulative_min_var,
            prob_is_min=selected_points_prob_is_min,
        )

        next_prob_is_min = torch.cat(
            (selected_points_prob_is_min[-1] * (1.0 - new_alpha_cdf[..., None]), new_alpha_cdf[..., None]), dim=-1
        )
        #  Update the correlation to previous matrix with fake point:
        cross_covariance_with_fake = torch.cat(
            (cross_covariance, cov_candidates_to_fake_point), dim=1
        )  # Shape (n_candidates, n_selected + 1)

        min_mean, min_var, *_ = get_next_cumulative_min_moments(
            mean=y_mean.flatten(),
            variance=y_var.flatten(),
            cov_to_next=cross_covariance_with_fake,
            cum_min_mean=next_cum_min_mean,
            cum_min_var=next_cum_min_var,
            prob_is_min=next_prob_is_min,
        )  # Shape (num_candidates) for both
        return min_mean, min_var


class MomentMatchingGMES(gp.base.MomentMatchingGMESBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._representer_points_means: Optional[torch.Tensor] = None
        self._representer_points_covariance: Optional[torch.Tensor] = None
        self._representer_points_min_mean: Optional[torch.Tensor] = None
        self._representer_points_min_var: Optional[torch.Tensor] = None
        self._representer_points_prob_is_min: Optional[torch.Tensor] = None

    def update_representer_points(self, representer_points: np.ndarray) -> None:
        super().update_representer_points(representer_points)
        repr_points_means, repr_points_cov = self.model.predict_with_full_covariance(representer_points)

        self._representer_points_means: torch.Tensor = torch.tensor(
            repr_points_means, dtype=torch.float64, requires_grad=False
        ).flatten()  #  Shape [n_representer]
        self._representer_points_covariance: torch.Tensor = torch.tensor(
            repr_points_cov, dtype=torch.float64, requires_grad=False
        )

        repr_points_min_mean, repr_points_min_var, repr_points_prob_is_min = approximate_minimum_with_prob_is_min(
            means=self._representer_points_means, covariance=self._representer_points_covariance
        )
        self._representer_points_min_mean: torch.Tensor = repr_points_min_mean
        self._representer_points_min_var: torch.Tensor = repr_points_min_var
        self._representer_points_prob_is_min: torch.Tensor = repr_points_prob_is_min

    def _evaluate_with_gradients(
        self, y_mean: np.ndarray, covariance: np.ndarray, cross_covariance: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """[summary]

        Args:
            y_mean: Array of means of shape (n_points, 1)
            covariance: Covariance of shape (n_points, n_points)

        Returns:
            Tuple[float, np.ndarray, np.ndarray]:
                float acquisition value for this batch of points.
                Gradient of the acquisition with respect to the passed means at the batch locations.
                Gradient of the acquisition with respect to the passed covariance between
                    points at the batch locations.
        """
        if (self._representer_points_covariance is None) or (self._representer_points_prob_is_min is None):
            raise ValueError(
                "Representer points must be set by calling update_representer_points() "
                "before this acquisition function is evaluated."
            )

        #  -- Transfer to Torch
        #  Map all arrays to torch tensors
        y_mean, covariance, cross_covariance = map(
            lambda x: torch.tensor(x, requires_grad=True, dtype=torch.float64), (y_mean, covariance, cross_covariance)
        )
        #  Get the cross-covariance to minimum at representer points:
        min_to_batch_cross_cov = calc_output_to_cum_min_cov(
            cross_covariance.transpose(dim0=-2, dim1=-1), prob_is_min=self._representer_points_prob_is_min
        )  #  Shape [..., n_points]

        #  Construct the full covariance matrix for minimum at representer points and chosen batch
        covariance_with_min = stack_covariance_from_subcovariances(
            cov1=self._representer_points_min_var[..., None, None],
            cov2=covariance,
            cross_cov=min_to_batch_cross_cov[..., None, :],
        )

        #  Calculate the entropy (+ constant) of the outputs at the batch of points
        covariance_cholesky = torch.linalg.cholesky(covariance)
        batch_half_log_det = covariance_cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)

        #  Calculate the entropy (+ constant) of the outputs and the minimum at representer points
        covariance_with_min_cholesky = torch.linalg.cholesky(covariance_with_min)
        batch_with_min_half_log_det = covariance_with_min_cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)

        #  Mutual information is the below + terms that don't depend on the points chosen for the batch
        gmes = batch_half_log_det - batch_with_min_half_log_det

        # -- Get the gradients with respect to y_mean, covariance and cross-covariance
        gmes.sum().backward()
        # Partial derivative of expected improvement (EI) wrt. covariance and cross-covariance
        #  (partial derivative wrt. to mean is 0)
        grad_covariance = covariance.grad.detach().numpy()
        grad_cross_covariance = cross_covariance.grad.detach().numpy()
        gmes = gmes.detach().numpy()
        return float(gmes), grad_covariance, grad_cross_covariance
