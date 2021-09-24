# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import argparse
import functools
import json
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
from emukit.core import ParameterSpace
from emukit.core.initial_designs import RandomDesign
from emukit.core.interfaces import IModel
from emukit.core.loop import CandidatePointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.model_wrappers import GPyModelWrapper

import gp.benchmarking.model_factory as mf
from gp.analytics_util import get_model_hyperparam
from gp.benchmarking.benchmark_functions import BenchmarkFunctions
from gp.benchmarking.candidate_calculator_factory import BatchMethod, get_candidate_point_calculator
from gp.numeric import standardize


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Global Penalization Benchmark.")
    parser.add_argument(
        "test_scenario", type=BenchmarkFunctions, choices=list(BenchmarkFunctions), help="Test scenario to use"
    )
    parser.add_argument(
        "--scenario_dim",
        type=int,
        default=None,
        help="For scenarios with variable dimensionality, set the dimensionality",
    )
    parser.add_argument(
        "--n_restarts", type=int, default=2, help="Number of restarts for the GPy hyperparam. optimization"
    )
    parser.add_argument(
        "--batch_method", type=BatchMethod, choices=list(BatchMethod), default=BatchMethod.SEQUENTIAL_MMEI
    )
    parser.add_argument(
        "--num_batches", type=int, default=30, help="Number of Bayesian Optimization loop iterations to execute."
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="Number of Bayesian Optimization loop iterations to execute."
    )
    parser.add_argument(
        "--n_anchor_batches", type=int, default=10, help="Number of anchor batches for simultaneous optimization."
    )
    parser.add_argument(
        "--n_candidate_anchor_batches", type=int, default=200, help="Number of samples to select anchor batches."
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=50,
        help="Number of randomly seeded experiments for statistical significance.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Random seed for experiments (experiments "
        "will have seeds: base_seed, base_seed + 1, base_seed + 2, ...).",
    )
    parser.add_argument("--use_gpy", help="If true, GPy (rather than GPyTorch) backend is used.", action="store_true")
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to override any previously saved files with experiment results.",
    )
    parser.add_argument(
        # "outputs" is a magic directory name, used to store any results in Azure ML
        "--results_dir",
        default=Path("outputs"),
        type=Path,
        help="Directory where to store results.",
    )

    return parser


def optimization_loop(
    X: np.ndarray,
    Y: np.ndarray,
    model,
    candidate_point_calculator: CandidatePointCalculator,
    num_batches: int,
    test_function: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Do a single Bayesian optimization run by looping the collection of batches from the
    test function for `num_batches` iterations.

    Args:
        X: Initial data input locations
        Y: Observations of outputs for initial data input locations
        model: The probabilistic model to fit
        candidate_point_calculator: Class responsible for decididing how to compute the next batch
            of points
        num_batches: How many batches to collect in this optimization loop
        test_function: The "true" function the minimum of which we are trying to find

    Returns:
        - X - the input locations for all the points the test function was queried at in this run
          (including the initial data)
        - Y - the observed outputs corresponding to points in X
        - times - A 1D array of size [num_batches] of wall-clock times at which each iteration of
          finding the next batch of points was finished
        - parameters - A pandas dataframe with the model hyperparameters at each iteration
          of this Bayesian Optimization loop
    """
    times = np.array([])

    parameters = get_model_hyperparam(model)

    for i in range(num_batches):
        loop_state = create_loop_state(X, Y)

        try:
            X_new = candidate_point_calculator.compute_next_points(loop_state)
        except Exception as e:
            print(
                f"Optimization of acquisition function failed with model hyperparameters:\n"
                f"{get_model_hyperparam(model)}"
            )
            raise e

        Y_new = test_function(X_new)

        X = np.append(X, X_new, axis=0)
        Y = np.append(Y, Y_new, axis=0)
        times = np.append(times, time.time())

        # Update the model and check its performance
        model.set_data(X, standardize(Y))
        model.optimize()
        # Store the model parameters
        parameters = parameters.append(get_model_hyperparam(model), ignore_index=True)

    return X, Y, times, parameters


def optimization_run(
    test_function: Callable[[np.ndarray], np.ndarray],
    parameter_space: ParameterSpace,
    model_constructor: Callable[[np.ndarray, np.ndarray], GPyModelWrapper],
    batch_method: BatchMethod,
    batch_size: int,
    num_batches: int,
    n_anchor_batches: int,
    n_candidate_anchor_batches: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    # Generate Initial Data
    X_init = RandomDesign(parameter_space).get_samples(batch_size)
    Y_init = test_function(X_init)

    # Log start-time
    start_time = time.time()

    # Model
    model = model_constructor(X_init, standardize(Y_init))
    model_params = get_model_hyperparam(model)  # Log hyperparams in a DataFrame
    model.optimize()
    # Define the acquisition function
    candidate_point_calculator = get_candidate_point_calculator(
        batch_method=batch_method,
        model=model,
        parameter_space=parameter_space,
        batch_size=batch_size,
        n_anchor_batches=n_anchor_batches,
        n_candidate_anchor_batches=n_candidate_anchor_batches,
    )

    X, Y, times, model_params_optim = optimization_loop(
        X=X_init,
        Y=Y_init,
        model=model,
        candidate_point_calculator=candidate_point_calculator,
        num_batches=num_batches,
        test_function=test_function,
    )
    times -= start_time  # Make all the times relative to start-time
    times = np.insert(times, 0, 0.0)  # Insert the starting time at the beginning of times array
    return X, Y, times, model_params.append(model_params_optim)


def get_save_dir(args: argparse.Namespace, seed: int):
    experiment_name = f"batch_method={args.batch_method}_batch_size={args.batch_size}"
    if args.scenario_dim is not None:
        experiment_name += f"_dim={args.scenario_dim}"
    assert 0 <= seed < 1e7
    save_dir: Path = args.results_dir / str(args.test_scenario) / experiment_name / f"seed{seed:06g}"
    save_dir.mkdir(parents=True, exist_ok=args.override)
    return save_dir


def model_constructor(
    use_gpytorch: bool, scenario: BenchmarkFunctions, n_restarts: int
) -> Callable[[np.ndarray, np.ndarray], IModel]:
    return functools.partial(
        mf.get_model_for_scenario_gpytorch if use_gpytorch else mf.get_model_for_scenario_gpy,
        scenario=scenario,
        n_restarts=n_restarts,
    )


def main(args: argparse.Namespace) -> None:
    scenario: BenchmarkFunctions = args.test_scenario

    # Get the benchmark scenario function and parameter space
    func, parameter_space = BenchmarkFunctions.get_function_and_space(scenario.value)

    # Count how many runs failed
    num_exp = args.num_experiments
    failed_runs: int = 0

    # Get the scenario model constructor
    for i in tqdm.tqdm(range(num_exp)):
        random_seed = args.base_seed + i

        last_exception: Optional[Exception] = None
        try:
            np.random.seed(random_seed)
            X, Y, times, model_params = optimization_run(
                test_function=func,
                parameter_space=parameter_space,
                model_constructor=model_constructor(
                    use_gpytorch=not args.use_gpy, scenario=args.test_scenario, n_restarts=args.n_restarts
                ),
                batch_method=args.batch_method,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
                n_anchor_batches=args.n_anchor_batches,
                n_candidate_anchor_batches=args.n_candidate_anchor_batches,
            )
            save_dir = get_save_dir(args, seed=random_seed)
            np.savetxt(save_dir / "X.csv", X)
            np.savetxt(save_dir / "Y.csv", Y)
            np.savetxt(save_dir / "times.csv", times)
            model_params.to_csv(save_dir / "model_params.csv", index=False)

            with open(save_dir / "config.json", "w") as fp:
                # Save the command-line settings as well (this is just a log, not meant to be reloadable)
                config_log = {key: str(value) for key, value in vars(args).items()}
                json.dump(config_log, fp)

        except Exception as e:
            failed_runs += 1
            last_exception = e
            print(f"Run {i} (seed={random_seed}) failed due to {e}.")

    percentage_successful = round(100 * (1 - failed_runs / num_exp))
    print(f"Experiment finished. Successful runs: {num_exp - failed_runs}/{num_exp} ({percentage_successful}%)")

    # If any exceptions were raised, re-raise the last one
    if last_exception is not None:
        raise last_exception


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
