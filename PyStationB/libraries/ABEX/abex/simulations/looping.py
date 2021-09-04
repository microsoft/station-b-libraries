# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------

import logging
from multiprocessing import Process
from typing import List, Type, Generator, Tuple, Dict, Any, cast

import matplotlib
from abex.scripts.run import RunConfig
from azureml.core import Run

matplotlib.use("Agg")

from abex.optimizers.optimizer_base import OptimizerBase  # noqa: E402
from abex.settings import OptimizerConfig, OptimizationStrategy, load_resolutions  # noqa: E402
from abex.simulations import DataLoop, SimulatedDataGenerator  # noqa: E402
from abex.simulations.interfaces import SimulatorBase  # noqa: E402


def run_single_loop(
    config: OptimizerConfig,
    num_iter: int,
    optim_strategy: OptimizationStrategy,
    plot_simulated_slices: bool,
) -> None:  # pragma: no cover
    """
    Perform a single run of Bayesian Optimization (or other optimization) on a simulator, with a particular
    random seed.

    Args:
        config (OptimizerConfig): OptimizerConfig for the run
        num_iter (int): Number of optimization/batch-collection iterations to perform in this run
        optim_strategy (OptimizationStrategy): Selection of optimization strategy to use in this run (Bayes, grid, etc.)
        plot_simulated_slices (bool): Whether to make plots of simulated slices against collected batches
            at the end of the run

    Example, if the main() function of a calling script calls this function:

        python calling_script.py config.yml --num_iter 5 --plot_simulated_slices

    where the config.yml is a config file specifying all the settings required for ABEX (see OptimizerConfig in
    abex/settings.py) and there are additional simulator-specific fields in (a subclass of) LoopConfig.
    """
    logging.info(f"Running a simulated loop run with random seed {config.seed}.")

    simulator = cast(SimulatorBase, config.get_simulator())
    data_generator = SimulatedDataGenerator(simulator)

    if config.num_init_points is not None:
        num_init_points = config.num_init_points
    elif optim_strategy == OptimizationStrategy.BAYESIAN:
        num_init_points = config.bayesopt.batch
    else:
        assert config.zoomopt is not None  # for mypy
        num_init_points = config.zoomopt.batch

    sim_loop = DataLoop.from_config(
        data_generator=data_generator,
        config=config,
        num_init_points=num_init_points,
        design_type=config.init_design_strategy,
    )
    sim_loop.run_loop(
        num_iter=num_iter,
        optimizer=OptimizerBase.from_strategy(config, optim_strategy),
        plot_sim_function_slices=plot_simulated_slices,
    )


def load_resolutions_from_command(
    args: RunConfig,
    loop_config_class: Type[OptimizerConfig],
) -> Generator[List[Tuple[Dict[str, Any], OptimizerConfig]], None, None]:  # pragma: no cover
    """
    Unpacks the attributes of args needed for load_resolutions, and returns the same values.
    """
    yield from load_resolutions(
        args.spec_file,
        seed=args.base_seed,
        num_runs=args.num_runs,
        config_class=loop_config_class,
        max_resolutions=args.max_resolutions,
    )


def tag_run(config: OptimizerConfig) -> None:  # pragma: no cover
    """
    Get context of AML run (local or remote) and add tags, to enable quick lookup and comparison
    Args:
        config:

    Returns:

    """
    run = Run.get_context()
    run.tag("optimization strategy", config.optimization_strategy)
    run.tag("batch strategy", config.bayesopt.batch_strategy.value)
    run.tag("acquisition", config.bayesopt.acquisition)
    run.tag("batch size", str(config.bayesopt.batch))
    run.tag("hmc", str(config.training.hmc))
    if config.zoomopt:
        if config.zoomopt.shrinking_factor is not None:
            run.tag("shrinking factor", str(config.zoomopt.shrinking_factor))


def run_multiple_seeds(
    args: RunConfig, pair_list: List[Tuple[Dict[str, Any], OptimizerConfig]]
) -> None:  # pragma: no cover
    processes = []
    logging.info("Logging convergence plot to AML")
    first_config = pair_list[0][1]
    tag_run(first_config)

    for _, config in pair_list:
        logging.info(f"Running config with data.folder={config.data.folder} and results_dir={config.results_dir}")
        if args.enable_multiprocessing:
            # Parse yaml config file
            process = Process(
                target=run_single_loop,
                args=(config, args.num_iter, config.optimization_strategy, args.plot_simulated_slices),
            )
            process.start()
            processes.append(process)
        else:
            run_single_loop(
                config=config.copy(deep=True),  # Copy the config (as it will be changed in place by this function)
                num_iter=args.num_iter,
                optim_strategy=config.optimization_strategy,
                plot_simulated_slices=args.plot_simulated_slices,
            )
    logging.info(f"All {args.num_runs} runs started.")
    if len(processes) > 1:
        for process in processes:
            process.join()
