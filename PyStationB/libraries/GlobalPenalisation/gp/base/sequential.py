# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""A base class for moment-matching approximation to an acquisition function.

The batch is supposed to be optimized in a sequential manner."""
from typing import Tuple

import abc
import numpy as np

from emukit.core.acquisition import Acquisition
from emukit.model_wrappers import GPyModelWrapper


from gp.base.chain_rule import chain_rule_means_vars, chain_rule_cross_covariance


class SequentialMomentMatchingBase(Acquisition, metaclass=abc.ABCMeta):
    """
    Abstract class for the sequential evaluation of a multi-point acquisition function.

    This class assumes that there are already ``n_selected`` points in the batch and we consider ``n_candidates``
    candidate points for the ``n_selected+1``th point in the batch.

    Attributes:
        model: wrapped GPy model
        selected_x: array of shape (n_selected, input_dim), with the inputs of points already selected to the batch
        selected_y_mean: array of shape (n_selected, 1), with the means of Gaussian variables at ``selected_x``
        selected_y_cov: array of shape (n_selected, n_selected), with the covariances between these variables
        selected_cumulative_min_means: array of shape (n_selected, 1) with means of cumulative minima of selected points
        selected_cumulative_min_std: array of shape (n_selected, 1) with std. deviations of cumulative minima of
            selected points
        selected_points_alpha_pdfs: array of shape (n_selected, 1)
        selected_points_alpha_cdfs: array of shape (n_selected, 1)

    Note: It is highly recommended not too shorten _cumulative_min_.. in variable names any further.
    """

    def __init__(self, model: GPyModelWrapper, selected_x: np.ndarray) -> None:
        """
        Args:
            model: Model
            selected_x: Inputs already selected for the batch. Shape (n_selected, n_inputs).
        """
        # Save the model and the input locations
        self.model: GPyModelWrapper = model
        self.selected_x: np.ndarray = selected_x  # (n_selected, input_dim)

        # Get the means and covariance matrix
        y_mean, y_cov = model.predict_with_full_covariance(self.selected_x)

        self.selected_y_mean: np.ndarray = y_mean  # (n_selected, 1)
        self.selected_y_cov: np.ndarray = y_cov  # (n_selected, n_selected)

    @property
    def has_gradients(self) -> bool:
        return True

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the acquisition function for candidate ``n_selected+1``-sized batches. (There are already
        ``n_selected`` points already in the batch and we consider many candidates for the last point).

        This method calls self.evaluate_with_gradients by default, but sub-classes can override this method to
        change this behaviour (potentially saving time on computing the gradients).

        Args:
            x: candidates for the next point added to the batch, shape (n_candidates, input_dim).

        Returns:
            values of the acquisition function of the points already selected plus the point added.
            Shape (n_candidates, 1).
        """
        result, _ = self.evaluate_with_gradients(x)
        return result

    @abc.abstractmethod
    def _evaluate_with_gradients(
        self, y_mean: np.ndarray, y_var: np.ndarray, cross_covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The abstract function that needs to be implemented.

        Args:
            y_mean: the mean of each Gaussian variable. Shape (n_candidates, 1)
            y_var: the variances of each Gaussian variable. Shape (n_candidates, 1)
            cross_covariance: the cross-covariance matrix between candidates and already selected samples.
                Shape (n_candidates, n_selected), where n_selected is the number of already selected samples

        Returns:
            acquisition function, shape (n_candidates, 1)
            derivative of the acquisition function with respect to the entries of ``y_mean``, shape (n_candidates, 1)
            derivative of the acquisition function with respect to the entries of ``y_var``, shape (n_candidates, 1)
            derivative of the acquisition function with respect to the entries of ``covariance``,
                shape (n_candidates, n_selected)
        """
        raise NotImplementedError

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the acquisition function value and gradients with respect to the candidate point coordinates.

        Args:
            x: candidates for the next point added to the batch, shape (n_candidates, input_dim).

        Returns:
            values of the acquisition function of the points already selected plus the point added.
                Shape (n_candidates, 1)
            gradients of the acquisition function with respect to the inputs, shape (n_candidates, input_dim)
        """
        # Predict mean and variance for the new point
        candidate_mean, candidate_variance = self.model.predict(x)  # Two arrays of shape (n_candidates, 1)

        # Covariance between candidates and selected points. Shape (n_candidates, n_selected)
        covariance_to_collected: np.ndarray = self.model.get_covariance_between_points(x, self.selected_x)

        # - Get the gradients of acquisition function with respect to the means, standard deviations and covariances
        # First three have shape (n_candidates, 1), the last one has shape (n_candidates, n_collected)
        acquisition, d_acq_d_mean, d_acq_d_var, d_acq_d_cross_cov = self._evaluate_with_gradients(
            y_mean=candidate_mean, y_var=candidate_variance, cross_covariance=covariance_to_collected
        )

        # - Get the gradients of means, variances and cross-covariance with respect to candidate coordinates

        # First, get gradients of means and variances. Both arrays have shape (n_candidates, n_inputs)
        d_means_dx, d_var_dx = self.model.get_prediction_gradients(x)  # Both have shape

        # Now get the gradient of covariances. Shape (n_candidates, n_selected, n_inputs)
        # Note that entry (i,j,k) represents dCov(y_candidate[i], y_selected[j]) / dx[i, k].
        d_cross_cov_dx = self.model.get_covariance_between_points_gradients(x, self.selected_x)

        # - Apply the chain rule
        d_acq_dx = (
            chain_rule_means_vars(d_acq_d_mean, d_means_dx)
            + chain_rule_means_vars(d_acq_d_var, d_var_dx)
            + chain_rule_cross_covariance(d_acq_d_cross_cov, d_cross_cov_dx)
        )

        return acquisition, d_acq_dx


class SequentialMomentMatchingBaseLCB(SequentialMomentMatchingBase):
    """
    Abstract class for the sequential evaluation of the multi-point Lower-Confidence Bound (LCB) acquisition function,
    which has an additional `beta` argument in the constructor.
    """

    def __init__(self, model: GPyModelWrapper, selected_x: np.ndarray, beta: float) -> None:
        super().__init__(model=model, selected_x=selected_x)
        self.beta: float = beta


class SequentialMomentMatchingDecorrelatingBaseEI(SequentialMomentMatchingBase):
    # TODO: if there is an issue with sometimes picking the same point
    #  a very simple solution to deal with this would be to add one more point to the "already-selected"
    # batch again – the one with the highest correlation to the newest point
    pass
