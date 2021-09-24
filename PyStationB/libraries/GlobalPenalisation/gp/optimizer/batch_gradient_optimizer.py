# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import logging
from typing import List, Tuple

import numpy as np
import scipy.optimize

from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.optimization.acquisition_optimizer import AcquisitionOptimizerBase
from emukit.core.optimization.context_manager import ContextManager

_log = logging.getLogger(__name__)


class GradientBatchAcquisitionOptimizer(AcquisitionOptimizerBase):
    """
    Optimizes a batch acquisition function using a quasi-Newton method (L-BFGS).
    Can be used for continuous acquisition functions.

    TODO: This currently doesn't integrate with EmuKit's categorical (context) variable optimization.

    """

    def __init__(
        self, space: ParameterSpace, batch_size: int, n_anchor_batches: int = 1, n_candidate_anchor_batches: int = 100
    ) -> None:
        """
        Args:
            space: The parameter space spanning the search problem.
            batch_size: number of points in the batch
            n_anchor_batches: number of anchor batches, from which L-BFGS is started
            n_candidate_anchor_batches: anchor batches are selected by sampling randomly this number of batches
        """
        super().__init__(space=space)
        self.batch_size = batch_size

        if n_anchor_batches > n_candidate_anchor_batches:
            raise ValueError(
                f"Number of random samples ({n_candidate_anchor_batches}) "
                f"must be greater or equal to the number of "
                f"anchor batches ({n_anchor_batches}) to be selected."
            )

        self.n_anchor_batches = n_anchor_batches
        self.n_candidate_anchor_batches = n_candidate_anchor_batches
        self._manual_anchor_batches: List[np.ndarray] = []

    @property
    def manual_anchor_batches(self) -> List[np.ndarray]:
        return self._manual_anchor_batches.copy()

    @manual_anchor_batches.setter
    def manual_anchor_batches(self, anchor_batches: List[np.ndarray]) -> None:
        self._manual_anchor_batches = anchor_batches

    def _optimize(self, acquisition: Acquisition, context_manager: ContextManager) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of abstract method.
        Taking into account gradients if acquisition supports them.
        See AcquisitionOptimizerBase._optimizer for parameter descriptions.
        """
        if len(context_manager.context_space.parameters) != 0:
            raise NotImplementedError

        # This is only useful when empirical gradients allowed as well:
        # def f(x: np.ndarray) -> float:
        #     return -acquisition.evaluate(np.reshape(x, (self.batch_size, -1)))

        bounds = context_manager.space.get_bounds() * self.batch_size

        # Context validation
        if len(context_manager.contextfree_space.parameters) == 0:
            raise ValueError("All context variables cannot be fixed when using a batch acquisition.")

        if acquisition.has_gradients:

            # Take negative of acquisition function because they are to be maximised and the optimizers minimise
            # The scipy optimizer only takes flat arrays, so reshape to match the acqusition
            def f_df(x):
                x_reshaped = np.reshape(x, (self.batch_size, -1))
                f_value, df_value = acquisition.evaluate_with_gradients(x_reshaped)
                return -f_value, -df_value.ravel()

        else:
            raise NotImplementedError()
            f_df = None

        # Select the anchor points (initial points)
        anchor_batches = _get_anchor_batches(
            space=self.space,
            acquisition=acquisition,
            batch_size=self.batch_size,
            num_samples=self.n_candidate_anchor_batches,
            num_batches=self.n_anchor_batches,
        )

        _log.info("Starting gradient-based optimization of acquisition function {}".format(type(acquisition)))

        # Location of the minimum and the value. We will update this, as we have multiple anchor points
        minimum_x, minimum_f = None, float("inf")

        for anchor_batch in anchor_batches:
            # Find the minimum starting from a given anchor point
            x_min_flat, fx_min, _ = scipy.optimize.fmin_l_bfgs_b(func=f_df, x0=anchor_batch.ravel(), bounds=bounds)
            x_min = np.reshape(x_min_flat, (self.batch_size, -1))

            # If the minimum is better than the observed so far, we update it
            if fx_min < minimum_f:
                minimum_f = fx_min
                minimum_x = x_min

        assert minimum_x is not None
        return minimum_x, -minimum_f


def _get_anchor_batches(
    space: ParameterSpace,
    acquisition: Acquisition,
    batch_size: int,
    num_batches: int,
    num_samples: int,
) -> List[np.ndarray]:
    """Randomly sample batches of points, and return the batches that yields the highest acquisition scores.

    Args:
        space (ParameterSpace): parameter space
        acquisition (Acquisition): acquisition function
        batch_size (int): batch size
        num_batches: number of batches to be returned
        num_samples (int): number of samples

    Returns:
        list of np.ndarray of length `num_batches`
    """
    if num_batches > num_samples:
        raise ValueError(
            f"Number of samples ({num_samples}) must be greater or equal to the number of "
            f"batches ({num_batches}) to be selected."
        )

    batch_candidates = [space.sample_uniform(batch_size) for _ in range(num_samples)]
    # Pairs (batch index, score). We sort them by decreasing score, to take the points with the highest scores
    scores = [(i, acquisition.evaluate(batch)) for i, batch in enumerate(batch_candidates)]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    indices_to_take = [i for i, _ in scores[:num_batches]]

    return [batch_candidates[i] for i in indices_to_take]
