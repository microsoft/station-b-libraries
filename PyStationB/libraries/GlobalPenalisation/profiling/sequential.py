# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""A simple profiling script for the moment-matched qEI calculated in a sequential manner using JAX.

Use:
    python profiling/sequential.py -h

for help.
"""
import argparse
import time
from typing import Literal


import emukit.bayesian_optimization.local_penalization_calculator as loc
import emukit.core as core
import numpy as np
from jax.config import config

import gp.moment_matching.jax as jax_backend
import gp.moment_matching.pytorch as torch_backend
import gp.testing as testing


np.random.seed(18920103)


def print_initialization(n_selected_points: int, time_delta: float) -> None:
    print(f"Model initialization with {n_selected_points} selected points in a batch: {time_delta:.3}.")


def profile_moment_matching(
    n_selected_points: int,
    n_candidate_points: int,
    n_training_points: int,
    input_dim: int,
    n_steps: int,
    backend: Literal["JAX", "TORCH"],
) -> None:
    # Train a GP model and choose some points to be in the batch
    model = testing.get_gpy_model(input_dim=input_dim, n_points=n_training_points)
    selected_x = np.random.rand(n_selected_points, input_dim)

    # Profile initialization of the acquisition (this is used to update the batch whenever we add a new point)
    t_start = time.time()
    if backend == "JAX":
        acquisition = jax_backend.SequentialMomentMatchingEI(model, selected_x=selected_x)
    elif backend == "TORCH":
        acquisition = torch_backend.SequentialMomentMatchingEI(model, selected_x=selected_x)
    else:
        raise ValueError(f"Backend {backend} not recognized -- must be 'JAX' or 'TORCH'")

    t_end = time.time()
    print_initialization(n_selected_points=n_selected_points, time_delta=t_end - t_start)

    # Profile the optimization
    _profile_optimization(
        acquisition=acquisition, n_steps=n_steps, n_candidate_points=n_candidate_points, input_dim=input_dim
    )


def _profile_optimization(acquisition: loc.Acquisition, n_steps: int, n_candidate_points: int, input_dim: int) -> None:
    # Evaluate the acquisition (with gradients) at several batches of candidate points
    print(f"Evaluating the acquisition function at {n_candidate_points} candidate points.")
    for step in range(1, n_steps + 1):
        candidate_x = np.random.rand(n_candidate_points, input_dim)
        t_start = time.time()
        acquisition.evaluate_with_gradients(candidate_x)
        t_end = time.time()
        print(f"\tStep {step}/{n_steps}: {t_end - t_start:.3}")


def profile_local_penalization(
    n_selected_points: int, n_candidate_points: int, n_training_points: int, input_dim: int, n_steps: int
) -> None:
    # Train a GP model
    model = testing.get_gpy_model(input_dim=input_dim, n_points=n_training_points)
    selected_x = np.random.rand(n_selected_points, input_dim)

    # Initialize the acquisition (this is used to update the batch whenever we add a new point)
    parameter_space = core.ParameterSpace(
        [core.ContinuousParameter(name=f"x{i}", min_value=0.0, max_value=1.0) for i in range(input_dim)]
    )
    acquisition = loc.LocalPenalization(model=model)

    t_start = time.time()
    f_min = 0.4
    lipschitz_constant = loc._estimate_lipschitz_constant(parameter_space, model)
    acquisition.update_batches(selected_x, lipschitz_constant, f_min)
    t_end = time.time()

    print_initialization(n_selected_points=n_selected_points, time_delta=t_end - t_start)

    # Profile the optimization
    _profile_optimization(
        acquisition=acquisition, n_steps=n_steps, n_candidate_points=n_candidate_points, input_dim=input_dim
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the Sequential methods.")

    parser.add_argument(
        "--no_jit",
        action="store_true",
        help="Disables JIT.",
    )
    parser.add_argument(
        "--selected", type=int, help="Number of points already selected to the batch. Default: 5.", default=5
    )
    parser.add_argument(
        "--candidates",
        type=int,
        help="Number of 'candidate' points, considered as the next point in the batch. Default: 100.",
        default=100,
    )
    parser.add_argument(
        "--steps", type=int, help="Number of optimization steps to find the next point. Default: 10.", default=10
    )
    parser.add_argument(
        "--training", type=int, help="Number of the points used to train the GP model. Default: 30.", default=30
    )
    parser.add_argument("--dim", type=int, help="The input dimension of the GP model. Default: 3.", default=3)

    # TODO: Refactor using Enum.
    parser.add_argument(
        "--method",
        help="Method. Use JAX, TORCH or LOCAL. Default: LOCAL.",
        default="LOCAL",
        choices=["JAX", "TORCH", "LOCAL"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        f"Profiler settings:\n"
        f"\tBackend used: {args.method}\n"
        f"\tSelected points: {args.selected}\n"
        f"\tCandidate points: {args.candidates}\n"
        f"\tOptimization steps: {args.steps}\n"
        f"\tGaussian Process training points: {args.training}\n"
        f"\tInput dimension: {args.dim}\n"
        f"\tJIT turned off: {args.no_jit} (only applies to JAX backend)\n\n"
    )

    if args.no_jit:
        config.update("jax_disable_jit", True)

    if args.method in {"JAX", "TORCH"}:
        profile_moment_matching(
            n_selected_points=args.selected,
            n_candidate_points=args.candidates,
            n_steps=args.steps,
            n_training_points=args.training,
            input_dim=args.dim,
            backend=args.method,
        )
    elif args.method == "LOCAL":
        profile_local_penalization(
            n_selected_points=args.selected,
            n_candidate_points=args.candidates,
            n_steps=args.steps,
            n_training_points=args.training,
            input_dim=args.dim,
        )
    else:
        raise ValueError("Method can only be JAX, TORCH or LOCAL.")

    print("\nRun finished.\n")


if __name__ == "__main__":
    main()
