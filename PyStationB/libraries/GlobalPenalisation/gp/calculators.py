# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Batch calculators."""
from typing import Dict, List, Optional, Type

import numpy as np
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
from emukit.core.loop import CandidatePointCalculator, LoopState
from emukit.core.loop.candidate_point_calculators import RandomSampling
from emukit.core.optimization import AcquisitionOptimizerBase
from emukit.model_wrappers import GPyModelWrapper

from gp.base import MomentMatchingGMESBase, SequentialMomentMatchingBase, SimultaneousMomentMatchingBase


class SequentialMomentMatchingCalculator(CandidatePointCalculator):
    """Sequential point calculator.

    For attributes, see the ``__init__`` method.
    """

    def __init__(
        self,
        acquisition_optimizer: AcquisitionOptimizerBase,
        model: GPyModelWrapper,
        parameter_space: ParameterSpace,
        batch_size: int,
        sequential_acquisition_generator: Type[SequentialMomentMatchingBase],
        single_point_acquisition_generator: Type[Acquisition],
    ) -> None:
        """

        Args:
            acquisition_optimizer: AcquisitionOptimizer object to optimize the penalized acquisition
            model: Model object, used to compute the parameters of the local penalization
            parameter_space: Parameter space describing input domain
            batch_size: Number of points to collect in each batch
            sequential_acquisition: The batch acquisition function to use to sequentially select
                each of the points for the batch.
            single_point_acquisition: Optionally, pass a different single-point acquisition function to apply
                only to select the first element in the batch. TODO: this is currently required, it would be great
                if this wasn't necessary
        """
        self.acquisition_optimizer = acquisition_optimizer
        self.model = model
        self.parameter_space = parameter_space
        self.batch_size = batch_size
        self.sequential_acquisition_generator = sequential_acquisition_generator
        self.single_point_acquisition_generator = single_point_acquisition_generator

    def compute_next_points(self, loop_state: LoopState, context: Optional[Dict] = None) -> np.ndarray:
        """Computes a batch of points using moment-matched approximation of q-EI.

        Args:
            loop_state: Object containing the current state of the loop
            context: it is required to match Emukit's API, but ignored for now

        Returns:
            a batch of points
        """
        # Compute first point:  # TODO: Shapes unclear.
        x1, _ = self.acquisition_optimizer.optimize(self.single_point_acquisition_generator(self.model))
        x_batch: List[np.ndarray] = [x1]

        # Compute the next points:
        for i in range(1, self.batch_size):
            selected_x = np.concatenate(x_batch, axis=0)
            acquisition = self.sequential_acquisition_generator(model=self.model, selected_x=selected_x)

            # Collect a new point
            x_next, _ = self.acquisition_optimizer.optimize(acquisition)
            x_batch.append(x_next)

        assert len(x_batch) == self.batch_size, f"Collected only {len(x_batch)} points. (Required: {self.batch_size}."
        return np.concatenate(x_batch, axis=0)


#  TODO: Any batch acquisition could be plugged in here really. Rename?
class SimultaneousMomentMatchingCalculator(CandidatePointCalculator):
    """Simultaneous point calculator for batch acquisitions.

    For attributes, see the ``__init__`` method.
    """

    def __init__(
        self,
        acquisition_optimizer: AcquisitionOptimizerBase,
        batch_acquisition: SimultaneousMomentMatchingBase,
    ) -> None:
        """

        Args:
            acquisition_optimizer: AcquisitionOptimizer object to optimize the penalized acquisition
            simultaneous_batch_acquisition: The batch acquisition function to use to sequentially select
                each of the points for the batch.
        """
        self.acquisition_optimizer = acquisition_optimizer
        self.batch_acquisition = batch_acquisition

    def compute_next_points(self, loop_state: LoopState, context: Optional[Dict] = None) -> np.ndarray:
        """Computes a batch of points using moment-matched approximation of q-EI.

        Args:
            loop_state: Object containing the current state of the loop
            context: it is required to match Emukit's API, but ignored for now

        Returns:
            a batch of points
        """
        # Compute first point:  # TODO: Shapes unclear.
        x_batch, _ = self.acquisition_optimizer.optimize(self.batch_acquisition)
        return x_batch


class MomentMatchingGMESCalculator(CandidatePointCalculator):
    """
    Simultaneous point calculator for GMES batch acquisitions.
    """

    def __init__(
        self,
        representer_point_acquisition: SimultaneousMomentMatchingBase,
        representer_acquisition_optimizer: AcquisitionOptimizerBase,
        batch_acquisition: MomentMatchingGMESBase,
        batch_acquisition_optimizer: AcquisitionOptimizerBase,
    ) -> None:
        self.batch_acquisition_optimizer = batch_acquisition_optimizer
        self.batch_acquisition = batch_acquisition
        self.representer_point_acquisition = representer_point_acquisition
        self.representer_acquisition_optimizer = representer_acquisition_optimizer

    def compute_next_points(self, loop_state: LoopState, context: Optional[Dict] = None) -> np.ndarray:
        """Computes a batch of points using moment-matched approximation of q-EI."""
        representer_points, _ = self.representer_acquisition_optimizer.optimize(self.representer_point_acquisition)
        self.batch_acquisition.update_representer_points(representer_points)

        x_batch, _ = self.batch_acquisition_optimizer.optimize(self.batch_acquisition)
        return x_batch


class BatchRandomSampling(RandomSampling):
    def __init__(self, parameter_space: ParameterSpace, batch_size: int):
        if (not isinstance(batch_size, int)) or (batch_size < 1):
            raise ValueError(f"Batch size must be a positive integer, but got {batch_size}")
        super().__init__(parameter_space)
        self.batch_size = batch_size

    def compute_next_points(self, loop_state, context=None):
        next_points = []
        for _ in range(self.batch_size):
            next_points.append(super().compute_next_points(loop_state, context=context))
        return np.concatenate(next_points, axis=0)


class GreedyRandomBatchPointCalculator(CandidatePointCalculator):
    """
    Batch point calculator. This point calculator calculates the first point in the batch using
    a single-point acquisition function,
    and then selects the rest of the points uniformly at random
    """

    def __init__(
        self,
        model: IModel,
        acquisition: Acquisition,
        acquisition_optimizer: AcquisitionOptimizerBase,
        batch_size: int = 1,
    ):
        """
        :param model: Model that is used by the acquisition function
        :param acquisition: Acquisition to be optimized to find each point in batch
        :param acquisition_optimizer: Acquisition optimizer that optimizes acquisition function
                                      to find each point in batch
        :param batch_size: Number of points to calculate in batch
        """
        if (not isinstance(batch_size, int)) or (batch_size < 1):
            raise ValueError(f"Batch size must be a positive integer, but got {batch_size}")
        elif batch_size == 1:
            raise ValueError(
                "This is a batch acquisition function. If batch-size is 1, use the single-point acquisition instead"
            )
        self.model = model
        self.acquisition = acquisition
        self.acquisition_optimizer = acquisition_optimizer
        self.batch_size = batch_size
        self.batch_random_sampler = BatchRandomSampling(
            parameter_space=acquisition_optimizer.space,
            batch_size=batch_size - 1,
        )

    def compute_next_points(self, loop_state: LoopState, context: dict = None) -> np.ndarray:
        """
        Args:
            loop_state: Object containing history of the loop
            context: Contains variables to fix through optimization of acquisition function. The dictionary key is
                        the parameter name and the value is the value to fix the parameter to.
        Return:
            2d array of size (batch_size x input dimensions) of new points to evaluate
        """
        self.acquisition.update_parameters()
        greedy_x, _ = self.acquisition_optimizer.optimize(self.acquisition, context)
        random_xs = self.batch_random_sampler.compute_next_points(loop_state, context)
        return np.concatenate((greedy_x, random_xs), axis=-2)
