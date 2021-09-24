# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Tests to make sure that the batch calculator picks a sensible set of points

All of these scenarios operate on a simple scenario:
    f(x) = x^2
where x is between -8.5 to 8.5, and the points collected so far are:
  x, y
(-8, 64)
(-6, 36)
(-5, 25)
(-4, 16)
( 4, 16)
( 5, 25)
( 6, 36)
( 8, 64)

Hopefully, the collected batch should be within  the range [-4, 4], and the output points shouldn't
be too similar.
"""
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.loop import CandidatePointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.model_wrappers import GPyModelWrapper
from GPy.models import GPRegression
import numpy as np
import pytest


from gp.calculators import (
    SequentialMomentMatchingCalculator,
    SimultaneousMomentMatchingCalculator,
    MomentMatchingGMESCalculator,
)
from gp.optimizer.batch_gradient_optimizer import GradientBatchAcquisitionOptimizer
from gp.moment_matching.pytorch import SequentialMomentMatchingEI, SimultaneousMomentMatchingEI, MomentMatchingGMES


def objective_function(x: np.ndarray):
    return x ** 2


def get_parameter_space():
    return ParameterSpace([ContinuousParameter("x", -8.5, 8.5)])


def get_dummy_data() -> np.ndarray:
    x = np.array([-8, -6, -5, -4, 4, 5, 6, 8])
    x = np.reshape(x, (-1, 1))
    return x, objective_function(x)


def get_gpy_model() -> GPyModelWrapper:
    return GPyModelWrapper(gpy_model=GPRegression(*get_dummy_data()))


def get_candidate_point_calculators(batch_size=3):
    ps = get_parameter_space()
    model = get_gpy_model()
    candidate_point_calculators = [
        SequentialMomentMatchingCalculator(
            acquisition_optimizer=GradientAcquisitionOptimizer(space=ps),
            model=model,
            parameter_space=ps,
            batch_size=batch_size,
            sequential_acquisition_generator=SequentialMomentMatchingEI,
            single_point_acquisition_generator=ExpectedImprovement,
        ),
        SimultaneousMomentMatchingCalculator(
            acquisition_optimizer=GradientBatchAcquisitionOptimizer(
                space=ps,
                batch_size=batch_size,
                n_candidate_anchor_batches=10,
                n_anchor_batches=2,
            ),
            batch_acquisition=SimultaneousMomentMatchingEI(model=model),
        ),
        MomentMatchingGMESCalculator(
            representer_point_acquisition=SimultaneousMomentMatchingEI(model=model),
            representer_acquisition_optimizer=GradientBatchAcquisitionOptimizer(
                space=ps,
                batch_size=batch_size + 2,
                n_candidate_anchor_batches=10,
                n_anchor_batches=1,
            ),
            batch_acquisition=MomentMatchingGMES(model=model),
            batch_acquisition_optimizer=GradientBatchAcquisitionOptimizer(
                space=ps,
                batch_size=batch_size,
                n_candidate_anchor_batches=5,
                n_anchor_batches=2,
            ),
        ),
    ]
    return zip(candidate_point_calculators, [model] * len(candidate_point_calculators))


@pytest.mark.parametrize("calculator, model", get_candidate_point_calculators())
def test_point_calculator_end_to_end(calculator: CandidatePointCalculator, model: GPyModelWrapper):
    loop_state = create_loop_state(model.X, model.Y)
    batch = calculator.compute_next_points(loop_state)
    assert batch.shape[1] == 1
    assert np.all(batch > -4)
    assert np.all(batch < 4), "All points in batch must be in the 'good' region"
    for point1_idx in range(batch.shape[0]):
        for point2_idx in range(batch.shape[0]):
            if point1_idx != point2_idx:
                # All points in batch must be sufficiently far apart
                assert np.all(np.abs(batch[point1_idx] - batch[point2_idx])) > 0.2
