# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
TODO: This file doesn't belong here.
"""
from typing import Tuple, Dict, Optional
import numpy as np
from emukit.core.acquisition import Acquisition
from emukit.core.loop import CandidatePointCalculator, LoopState
from emukit.core import ParameterSpace


class CorrelationPenalization(Acquisition):
    """Correlation based penalizer."""

    def __init__(self, model, prev_x: np.ndarray):
        self.prev_x = prev_x
        self.prev_y_mean, prev_y_var = model.predict(self.prev_x)
        self.prev_y_std = np.sqrt(prev_y_var)
        self.model = model

    @property
    def has_gradients(self) -> bool:
        return False

    def update_batches(self, x_batch: np.ndarray):
        pass

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the penalization function value. x is of shape [num_points, input_dim].
        """
        covar = self.model.get_covariance_between_points(x, self.prev_x)
        _, new_y_variance = self.model.predict(x)
        new_y_std = np.sqrt(new_y_variance)
        # covar is of shape [x.shape[0], prev_x.shape[0]]. Normalise each entry by the
        # std of corresponding observation at entries in x and prev_x
        correlation = covar / (new_y_std * self.prev_y_std.T)
        penalization = (1.0 - correlation).prod(axis=1, keepdims=True)
        return penalization

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the penalization function value and gradients with respect to x
        """
        # TODO: The below method computes many unnecessary gradientes. Computational overhead.
        dmean_dx, dvariance_dx = self.model.get_joint_prediction_gradients(np.concatenate((self.prev_x, x), axis=0))
        # if not isinstance(self.model, IJointlyDifferentiable):
        #     raise AttributeError("Model is not jointly differentiable.")
        # TODO
        raise NotImplementedError()


class CorrelationPenalizationPointCalculator(CandidatePointCalculator):
    """
    Probability of Improvement insipred global penalization point calculator
    """

    def __init__(
        self, acquisition: Acquisition, acquisition_optimizer, model, parameter_space: ParameterSpace, batch_size: int
    ):
        """
        :param acquisition: Base acquisition function to use without any penalization applied, this acquisition should
                            output positive values only.
        :param acquisition_optimizer: AcquisitionOptimizer object to optimize the penalized acquisition
        :param model: Model object, used to compute the parameters of the local penalization
        :param parameter_space: Parameter space describing input domain
        :param batch_size: Number of points to collect in each batch
        """
        self.acquisition = acquisition
        self.acquisition_optimizer = acquisition_optimizer
        self.batch_size = batch_size
        self.model = model
        self.parameter_space = parameter_space

    def compute_next_points(self, loop_state: LoopState, context: Optional[Dict] = None) -> np.ndarray:
        """
        Computes a batch of points using local penalization.
        :param loop_state: Object containing the current state of the loop
        """
        self.acquisition.update_parameters()

        #  Compute first point:
        x1, _ = self.acquisition_optimizer.optimize(self.acquisition)
        x_batch = [x1]

        # Compute the next points:
        for i in range(1, self.batch_size):
            penalization_acquisition = CorrelationPenalization(self.model, prev_x=x1)
            acquisition = self.acquisition + penalization_acquisition
            # Collect point
            x_next, _ = self.acquisition_optimizer.optimize(acquisition)
            x_batch.append(x_next)
        assert len(x_batch) == self.batch_size  # TODO: Remove
        return np.concatenate(x_batch, axis=0)
