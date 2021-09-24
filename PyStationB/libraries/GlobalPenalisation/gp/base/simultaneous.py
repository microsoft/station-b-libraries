# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""A base class for moment-matching approximation to an acquisition function.

The points in the batch are supposed to be optimized simultaneously."""
from typing import Optional, Tuple, Union

import abc
import numpy as np

from emukit.core.acquisition import Acquisition
from emukit.model_wrappers import GPyModelWrapper


from gp.base.chain_rule import chain_rule_means_from_predict_joint, chain_rule_covariance, chain_rule_cross_covariance
from gp.model_wrapper import GPyTorchModelWrapper


class SimultaneousMomentMatchingBase(Acquisition, metaclass=abc.ABCMeta):
    """This class implements the moment-matched acquisition function for the case in which all points in the batch
    are optimized at once.

    Attributes:
        model: wrapped GPy model
    """

    def __init__(self, model: Union[GPyModelWrapper, GPyTorchModelWrapper]) -> None:
        """
        Args:
            model: Model
        """
        self.model = model

    @property
    def has_gradients(self) -> bool:
        return True

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluates the moment-matched approximation to the multipoint Expected Improvement on a batch of points.

        Args:
            x: a batch of candidate points. Shape (n_points, input_dim).

        Returns:
            the value of moment-matched qEI on the batch of points
        """
        result, _ = self.evaluate_with_gradients(x)
        return result

    @abc.abstractmethod
    def _evaluate_with_gradients(
        self, y_mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluates the gradients of the acquisition function with respect to the means vector and the covariance
        matrix for a candidate batch.

        Args:
            y_mean: the mean vector of the multinormal variable. Shape (n_points, 1).
            covariance: the covariance matrix between the points. Shape (n_points, n_points).

        Returns:
            moment-matching approximation of the multi-point EI
            derivative of the acquisition function with respect to the entries of ``y_mean``, shape (n_points, 1)
            derivative of the acquisition function with respect to the entries of ``covariance``, shape
                (n_points, n_points)
        """
        pass

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluates the acquisition function value and gradients with respect to a candidate batch of points.

        Args:
            x: a batch of points, shape (n_points, input_dim)

        Returns:
            moment-matched approximation of q-EI for this batch
            the gradient of this acquisition function with respect to the batch entries.
                Shape (n_points, input_dim).
        """
        # Predict the means vector (shape (n_points, 1)) and
        # covariance matrix (shape (n_points, n_points)).
        means, covariance = self.model.predict_with_full_covariance(x)

        # Evaluate acquisition function and gradients with respect to means and covariance
        acquisition, d_acq_d_means, d_acq_d_covariance = self._evaluate_with_gradients(means, covariance)

        # Get gradients of means and covariances.
        # Shapes (n_points, n_points, input_dim)
        # and (n_points, n_points, n_points, input_dim)
        d_means_dx, d_covariance_dx = self.model.get_joint_prediction_gradients(x)

        # Use chain rule
        d_acq_dx = chain_rule_means_from_predict_joint(d_acq_d_means, d_means_dx) + chain_rule_covariance(
            d_acq_d_covariance, d_covariance_dx
        )

        return acquisition, d_acq_dx


class SimultaneousMomentMatchingBaseLCB(SimultaneousMomentMatchingBase, metaclass=abc.ABCMeta):
    """An abstract class for LCB (Lower Confidence Bound), which has an additional `beta` argument in the constructor.

    Parameters:
        beta: trade off between mean and standard deviation
    """

    def __init__(self, model: GPyModelWrapper, beta: float) -> None:
        super().__init__(model=model)
        self.beta: float = beta


class MomentMatchingGMESBase(SimultaneousMomentMatchingBase, metaclass=abc.ABCMeta):
    """An abstract class for Generalised Max-Value Entropy Search, which takes as input a set of
    inputs as a discretisation of the input domain to form an approximation over the minimum
    of the function.

    Parameters:
        beta: trade off between mean and standard deviation
    """

    def __init__(self, model: GPyModelWrapper) -> None:
        super().__init__(model=model)
        #  A finite number of locations the outputs of which will be used to approximate the
        #  minimum of the model over the entire input space
        self._representer_points: Optional[np.ndarray] = None

    @property
    def representer_points(self) -> Optional[np.ndarray]:
        return self._representer_points

    def update_representer_points(self, representer_points: np.ndarray) -> None:
        self._representer_points = representer_points

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluates the acquisition function value and gradients with respect to a candidate batch of points.

        Args:
            x: a batch of points, shape (n_points, input_dim)
        """
        # Predict the means vector (shape (n_points, 1)) and
        # covariance matrix (shape (n_points, n_points)).
        means, covariance = self.model.predict_with_full_covariance(x)

        #  Predict cross-covariance between x and the representer points
        cross_covariance = self.model.get_covariance_between_points(x, self.representer_points)

        # Evaluate acquisition function and gradients with respect to means and covariance
        acquisition, d_acq_d_covariance, d_acq_d_cross_cov = self._evaluate_with_gradients(
            means, covariance, cross_covariance
        )

        # Get gradients of means and covariances.
        # Shapes (n_points, n_points, input_dim)
        # and (n_points, n_points, n_points, input_dim)
        d_means_dx, d_covariance_dx = self.model.get_joint_prediction_gradients(x)

        # Now get the gradient of cross-covariances. Shape (n_point, n_representer, input_dim)
        # Note that entry (i,j,k) represents dCov(y_candidate[i], y_selected[j]) / dx[i, k].
        d_cross_cov_dx = self.model.get_covariance_between_points_gradients(x, self.representer_points)

        # Use chain rule
        d_acq_dx = (
            # chain_rule_means_from_predict_joint(d_acq_d_means, d_means_dx)
            chain_rule_covariance(d_acq_d_covariance, d_covariance_dx)
            + chain_rule_cross_covariance(d_acq_d_cross_cov, d_cross_cov_dx)
        )

        return acquisition, d_acq_dx

    @abc.abstractmethod
    def _evaluate_with_gradients(
        self, y_mean: np.ndarray, covariance: np.ndarray, cross_covariance: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluates the gradients of the acquisition function with respect to the means vector and the covariance
        matrix for a candidate batch.

        Args:
            y_mean: the mean vector of the multinormal variable. Shape (n_points, 1).
            covariance: the covariance matrix between the points. Shape (n_points, n_points).
            cross_covariance: the covariance between the points in the candidate batch and the
                previously set representer points. Shape(n_points, n_representer_points)

        Returns:
            moment-matching approximation of the generalised max-value entropy search
            # derivative of the acquisition function with respect to the entries of ``y_mean``, shape (n_points, 1)
            derivative of the acquisition function with respect to the entries of ``covariance``, shape
                (n_points, n_points)
            derivative of the acquisition function with respect to the entries of ``cross_covariance``, shape
                (n_points, n_representer_points)
        """
        pass
