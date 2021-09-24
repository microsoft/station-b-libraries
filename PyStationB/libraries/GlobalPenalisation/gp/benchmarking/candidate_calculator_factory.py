# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import enum
import numpy as np
from typing import Tuple

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition
from emukit.bayesian_optimization.local_penalization_calculator import LocalPenalizationPointCalculator
from emukit.core import ParameterSpace
from emukit.core.loop import CandidatePointCalculator
from emukit.core.loop.candidate_point_calculators import GreedyBatchPointCalculator
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.model_wrappers import GPyModelWrapper

from gp.calculators import (
    MomentMatchingGMESCalculator,
    SequentialMomentMatchingCalculator,
    SimultaneousMomentMatchingCalculator,
    BatchRandomSampling,
    GreedyRandomBatchPointCalculator,
)
from gp.moment_matching.pytorch import (
    MomentMatchingGMES,
    SequentialMomentMatchingEI,
    SequentialMomentMatchingDecorrelatingEI,
    SimultaneousMomentMatchingEI,
    SimultaneousMomentMatchingExpectedMin,
)
from gp.optimizer.batch_gradient_optimizer import GradientBatchAcquisitionOptimizer
from gp.qei import MultipointExpectedImprovement


class BatchMethod(enum.Enum):
    """
    Attributes:
        LOCAL_PENALIZATION: Local Penalization from http://proceedings.mlr.press/v51/gonzalez16a.pdf
        SEQUENTIAL_MMEI: Sequentially optimized moment-matched qEI
        SIMULTANEOUS_MMEI: Simultaneously optimized moment-matched qEI
        EXACT_EI: Exact simultaneously optimized qEI
        SIMULTANEOUS_GMES: Simultaneously optimized moment-matched GMES
            (Generalised Max-value Entropy Search)
        RANDOM: Select the batch by sampling uniformly at random
        EI_GREEDY_RANDOM: Select the batch by picking the first point by maximising Expected Improvement,
            then uniformly at random
        EI_PRED_GREEDY: Select the batch by optimizing Expected Improvement, and then appending
            a fake observation to the model at the mean of the model's prediction.
    """

    LOCAL_PENALIZATION = "local-penalization"
    SEQUENTIAL_MMEI = "sequential-moment-matched-ei"
    DECORRELATING_SEQUENTIAL_MMEI = "decorrelating-sequential-mmei"
    SIMULTANEOUS_MMEI = "simultaneous-moment-matched-ei"
    EXACT_EI = "exact-ei"
    SIMULTANEOUS_GMES = "simultaneous-gmes"
    RANDOM = "random"
    EI_GREEDY_RANDOM = "ei-greedy-random"
    EI_PRED_GREEDY = "ei-pred-greedy"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        if self == BatchMethod.LOCAL_PENALIZATION:
            return "Local Penalization"
        elif self == BatchMethod.SEQUENTIAL_MMEI:
            return "Moment-Matched qEI (sequential)"
        elif self == BatchMethod.DECORRELATING_SEQUENTIAL_MMEI:
            return "Moment-Matched qEI (decorr. sequential)"
        elif self == BatchMethod.SIMULTANEOUS_MMEI:
            return "Moment-Matched qEI (simultaneous)"
        elif self == BatchMethod.EXACT_EI:
            return "Exact qEI"
        elif self == BatchMethod.SIMULTANEOUS_GMES:
            return "GMES"
        elif self == BatchMethod.RANDOM:
            return "Random Search"
        elif self == BatchMethod.EI_GREEDY_RANDOM:
            return "EI + Random"
        elif self == BatchMethod.EI_PRED_GREEDY:
            return "EI + Pred. Greedy"
        else:
            raise NotImplementedError


class UnnegatedMultipointExpectedImprovement(MultipointExpectedImprovement):
    """
    Multipoint Expected Improvement in EmuKit is negated for some reason.

    This wrapper un-negates the negations of the multi-point expected improvement.
    """

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return -super().evaluate(x)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f, f_dx = super().evaluate_with_gradients(x)
        return -f, -f_dx


def get_candidate_point_calculator(
    batch_method: BatchMethod,
    model: GPyModelWrapper,
    parameter_space: ParameterSpace,
    batch_size: int,
    n_anchor_batches: int,
    n_candidate_anchor_batches: int,
) -> CandidatePointCalculator:
    # Define the optimizer used for simultaneous batch optimization
    gradient_batch_acquisition_optimizer = GradientBatchAcquisitionOptimizer(
        space=parameter_space,
        batch_size=batch_size,
        n_anchor_batches=n_anchor_batches,
        n_candidate_anchor_batches=n_candidate_anchor_batches,
    )

    if batch_method == BatchMethod.LOCAL_PENALIZATION:
        acquisition_function = ExpectedImprovement(model=model)
        log_acquisition = LogAcquisition(acquisition_function)
        return LocalPenalizationPointCalculator(
            acquisition=log_acquisition,
            acquisition_optimizer=GradientAcquisitionOptimizer(parameter_space),
            model=model,
            parameter_space=parameter_space,
            batch_size=batch_size,
        )
    elif batch_method == BatchMethod.SEQUENTIAL_MMEI:
        return SequentialMomentMatchingCalculator(
            acquisition_optimizer=GradientAcquisitionOptimizer(parameter_space),
            model=model,
            parameter_space=parameter_space,
            batch_size=batch_size,
            sequential_acquisition_generator=SequentialMomentMatchingEI,
            single_point_acquisition_generator=ExpectedImprovement,
        )
    elif batch_method == BatchMethod.DECORRELATING_SEQUENTIAL_MMEI:
        return SequentialMomentMatchingCalculator(
            acquisition_optimizer=GradientAcquisitionOptimizer(parameter_space),
            model=model,
            parameter_space=parameter_space,
            batch_size=batch_size,
            sequential_acquisition_generator=SequentialMomentMatchingDecorrelatingEI,
            single_point_acquisition_generator=ExpectedImprovement,
        )
    elif batch_method == BatchMethod.SIMULTANEOUS_MMEI:
        acquisition_function = SimultaneousMomentMatchingEI(model=model)
        return SimultaneousMomentMatchingCalculator(
            acquisition_optimizer=gradient_batch_acquisition_optimizer,
            batch_acquisition=acquisition_function,
        )
    elif batch_method == BatchMethod.EXACT_EI:
        acquisition_function = UnnegatedMultipointExpectedImprovement(model=model)
        return SimultaneousMomentMatchingCalculator(
            acquisition_optimizer=gradient_batch_acquisition_optimizer,
            batch_acquisition=acquisition_function,  # type: ignore
        )
    elif batch_method == BatchMethod.SIMULTANEOUS_GMES:
        # Use twice as many points for the representer points (TODO: experiment with more?)
        representer_optimizer = GradientBatchAcquisitionOptimizer(space=parameter_space, batch_size=batch_size * 2)
        return MomentMatchingGMESCalculator(
            representer_point_acquisition=SimultaneousMomentMatchingExpectedMin(model=model),
            representer_acquisition_optimizer=representer_optimizer,
            batch_acquisition=MomentMatchingGMES(model=model),
            batch_acquisition_optimizer=gradient_batch_acquisition_optimizer,
        )
    elif batch_method == BatchMethod.RANDOM:
        return BatchRandomSampling(parameter_space=parameter_space, batch_size=batch_size)
    elif batch_method == BatchMethod.EI_GREEDY_RANDOM:
        acquisition_function = ExpectedImprovement(model=model)
        return GreedyRandomBatchPointCalculator(
            model=model,
            acquisition=acquisition_function,
            acquisition_optimizer=GradientAcquisitionOptimizer(parameter_space),
            batch_size=batch_size,
        )
    elif batch_method == BatchMethod.EI_PRED_GREEDY:
        acquisition_function = ExpectedImprovement(model=model)
        return GreedyBatchPointCalculator(
            model=model,
            acquisition=acquisition_function,
            acquisition_optimizer=GradientAcquisitionOptimizer(parameter_space),
            batch_size=batch_size,
        )
    else:
        raise ValueError
