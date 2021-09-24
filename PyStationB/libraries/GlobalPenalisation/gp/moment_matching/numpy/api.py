# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This module implements an approximation to multi-point expected improvement using the moment-matching strategy
introduced in:

C.E. Clark, The Greatest of a Finite Set of Random Variables, Operations Research, Vol. 9, No. 2 (1961).

Note: The API in this module is out of date with the standard in gp.base
"""
import numpy as np

import gp.base as base
import gp.numeric as numeric
from gp.numeric import correlation_between_distinct_sets_from_covariance
from gp.moment_matching.numpy.moment_matching_minimum import (
    calculate_cumulative_min_moments,
    get_next_cumulative_min_moments,
)


class SequentialMomentMatchingEI(base.SequentialMomentMatchingBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Calculate and save the cumulative minima means and standard deviations
        (
            self.selected_cumulative_min_means,
            self.selected_cumulative_min_std,
            self.selected_points_alpha_pdfs,
            self.selected_points_alpha_cdfs,
        ) = calculate_cumulative_min_moments(
            means=self.selected_y_mean.T,
            stds=self.selected_y_std.T,
            corr_matrix=self.selected_y_correlation[None, :, :],
        )  # Shape (n_selected, 1) each

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        candidate_mean, candidate_variance = self.model.predict(x)  # Two arrays of shape (n_candidates, 1)
        candidate_std: np.ndarray = np.sqrt(candidate_variance)  # Array of shape (n_candidates, 1)

        # Covariance between candidates and selected points. Shape (n_candidates, n_selected)
        covariance_to_collected: np.ndarray = self.model.get_covariance_between_points(x, self.selected_x)

        # Get the minimum as observed so far
        y_min = self.model.Y.min()

        # Get the correlation from new points to selected:
        correlation = correlation_between_distinct_sets_from_covariance(
            covariance=covariance_to_collected,
            std1=candidate_std.ravel(),
            std2=self.selected_y_std.ravel(),
        )  # Shape (n_candidates, n_selected)

        # If D points already selected, the index of next point to be selected will be D
        next_point_idx = self.selected_y_mean.shape[0]

        cumulative_min_mean, cumulative_min_std, *_ = get_next_cumulative_min_moments(
            next_output_idx=next_point_idx,
            mean=candidate_mean.ravel(),
            std=candidate_std.ravel(),
            prev_stds=self.selected_y_std.T,
            corr_to_next=correlation,
            theta_means=self.selected_cumulative_min_means,
            theta_stds=self.selected_cumulative_min_std,
            alpha_pdfs=self.selected_points_alpha_pdfs,
            alpha_cdfs=self.selected_points_alpha_cdfs,
        )  # Shape (num_candidates) for both

        #  Calculate the Expected Improvement
        expected_improvement = numeric.expected_improvement(
            y_min=y_min, mean=cumulative_min_mean, standard_deviation=cumulative_min_std
        )  #  Shape (num_candidates)

        return expected_improvement[:, None]

    @property
    def has_gradients(self) -> bool:
        return False
