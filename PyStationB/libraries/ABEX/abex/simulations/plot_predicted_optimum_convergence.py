# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Type, cast, Optional

import numpy as np
import pandas as pd
from abex.plotting.convergence_plotting import plot_objective_distribution_convergence
from abex.settings import OptimizerConfig, OptimizationStrategy, simple_load_config
from abex.simulations import SimulatorBase
from azureml.core import Run
from azureml.core.run import _OfflineRun
from matplotlib import pyplot as plt
from psbutils.psblogging import logging_to_stdout


def create_parser() -> argparse.ArgumentParser:  # pragma: no cover
    """
    Argument parser for plotting.
    """
    parser = argparse.ArgumentParser(
        description="Plot convergence over several iterations of Bayesian Optimization on a simulator by "
        "plotting the distribution of the objective from the simulator at the optima predicted during the run."
    )
    parser.add_argument(
        "--experiment_dirs",
        type=Path,
        action="append",
        required=True,
        help="A sequence of directories corresponding to different configurations for an experiment "
        "(different runs). Each directory should contain multiple sub-directories with multiple sub-runs "
        "corresponding to different random seeds. Those sub-directories should contain result directories "
        "for each iteration of the optimisation algorithm.",
    )
    parser.add_argument(
        "--experiment_labels",
        type=str,
        action="append",
        choices=["acquisition", "batch_strategy", "optimization_strategy", "batch", "shrinking_factor", "hmc"],
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The resulting plot will be saved at this location.",
    )
    parser.add_argument("--title", type=str, default=None, help="The title for the plot.")
    parser.add_argument(
        "--max_batch_number",
        type=int,
        default=None,
        help="Whether to clip the x-axis to a given number of batches.",
    )
    parser.add_argument(
        "--output_scale",
        type=str,
        default=None,
        choices=["symlog", "log", "linear"],
        help="What scale to use for the objective on the plot. Default to log if all objective values are positive, "
        "symlog otherwise.",
    )
    parser.add_argument(
        "--styled_subset_param",
        type=str,
        default=[],
        action="append",
        help="Group experiments by given parameter(s) to distinguish them as related in plots. Currently only"
        "acquisition, batch_strategy and optimization_strategy are accepted values.",
    )
    parser.add_argument(
        "--style_category_name",
        type=str,
        default="Category",
        help="Name to use on the legend for the style categories.",
    )
    parser.add_argument(
        "--num_simulator_samples_per_optimum",
        type=int,
        default=1,
        help="The number of objective samples to draw from the simulator at each suggested optimum location. "
        "The higher the number, the more accurrate the plot of the distribution at suggested optimum will be.",
    )
    return parser


BATCH_COLUMN = "Batch Number"
RUN_NAME_COLUMN = "Experiment Name"
SEED_COLUMN = "Sub-run Number"
OBJECTIVE_COLUMN = "Objective"
CONFIG_FILENAME = "config.yml"
ITERATION_DIR_PREFIX: str = "iter"
OPTIMA_FILE = "optima.csv"


def _extract_iteration_number(iteration_dir: Path) -> int:  # pragma: no cover
    """Converts `iterXY` into integer `XY`."""
    n = len(ITERATION_DIR_PREFIX)
    iteration_dir_name = iteration_dir.stem
    return int(iteration_dir_name[n:])


def load_optimum_file(
    path: Path, input_names: List[str], iteration_number: int
) -> Dict[str, float]:  # pragma: no cover
    """Reads the file with the location of the optimum

    Args:
        path: the CSV with the optimum
        input_names: inputs which should be retained
        iteration_number: iteration number, to be added to the returned dictionary

    Returns:
        dictionary which keys are `input_names` and `BATCH_COLUMN`.

    Raises:
        ValueError, if `path` contains more than one row (the optimum is not unique)
    """
    # Read the CSV with optima and raise an error if it contains more than one data point
    iteration_optimum: pd.DataFrame = pd.read_csv(path)  # type: ignore # auto

    if len(iteration_optimum) > 1:
        raise ValueError
    # Read the input values
    point = {name: iteration_optimum[name].values[0] for name in input_names}
    # Add the information about the iteration
    point[BATCH_COLUMN] = iteration_number

    return point  # type: ignore # auto


def load_seeded_subrun_df(subrun_dir: Path, input_names: List[str]) -> pd.DataFrame:  # pragma: no cover
    """Return a DataFrame with the optima suggested in this 'sub-run'. This function iterates over the result
    directories starting with "iter" in this directory , and assumes the remaining part of the result directory
    name indicates the iteration. Adds a column to the DataFrame to indicate batch number.
    """
    iteration_dir_pattern: str = f"{ITERATION_DIR_PREFIX}*"
    iteration_directories: List[Path] = list(subrun_dir.glob(iteration_dir_pattern))
    iteration_directories = sorted(iteration_directories, key=lambda path: _extract_iteration_number(path))

    # This is a list of optima (one for each iteration). Will be converted into a dataframe at the end.
    proto_dataframe = []

    # Now add the information about 1st, 2nd, ... iterations.
    for iteration_directory in iteration_directories:
        optimum_path = iteration_directory / OPTIMA_FILE
        iteration_number = _extract_iteration_number(iteration_directory)

        sample = load_optimum_file(path=optimum_path, input_names=input_names, iteration_number=iteration_number)
        proto_dataframe.append(sample)

    return pd.DataFrame(proto_dataframe)


def _get_input_names(experiment_dir: Path, loop_config: Type[OptimizerConfig]) -> List[str]:  # pragma: no cover
    """Read the input names in an experiment.

    Args:
        experiment_dir: experiment directory

    Returns:
        list with input names, as read from the config
    """
    config = load_config_from_expt_dir(experiment_dir, loop_config)
    input_names = [name for name in config.data.input_names]
    return input_names


def _get_seeded_runs(experiment_dir: Path) -> Dict[str, Path]:  # pragma: no cover
    """
    Returns:
       a dictionary in the format `subrun_name: path to the subrun` (containing many iterations)
    """
    subrun_paths = [child for child in experiment_dir.glob("*/seed*") if child.is_dir()]
    subrun_names = [_path_subrun_name(path) for path in subrun_paths]
    logging.info(f"experiment_dir is {experiment_dir} with subdirs {subrun_names}")

    runs_dictionary = dict(zip(subrun_names, subrun_paths))
    return runs_dictionary


def _path_subrun_name(path: Path) -> str:  # pragma: no cover
    """
    Returns a suitable name for a subrun located at .../selection_spec/seedNNN. If selection_spec
    is "fixed", just return seedNNN; otherwise, prepend selection_spec.

    TODO: This function may be modified at later stage, to return the name in a prettier format.
    """
    parent_stem = path.parent.stem
    if parent_stem == "fixed":
        return path.stem
    return f"{parent_stem}_{path.stem}"


def load_experiment_label(config: OptimizerConfig, experiment_label_params: List[str]) -> str:
    # TODO: make experiment_label_params a list of enums and define a more sophisticated label. e.g. include HMC
    """
    Create an experiment label based on options specified in the config file

    Args:
        config:
        experiment_label_params: A list of argument names to append to experiment label

    Returns: A string representing the label for one combination of acquisition plus batch strategy
    plus optimization strategy.

    E.g.
    EXPECTED_IMPROVEMENT - LocalPenalization - Bayes
    MEAN_PLUGIN_EXPECTED_IMPROVEMENT - LocalPenalization - Zoom(0.5)
    """
    assert any([config.bayesopt, config.zoomopt])

    optimization_strategy = config.optimization_strategy

    experiment_label = ""

    for label_param in experiment_label_params:
        if len(experiment_label) > 0:
            experiment_label += " "

        # start with properties common to all optimization strategies
        if label_param == "optimization_strategy":
            experiment_label += f"{config.optimization_strategy}"
        elif label_param == "hmc":
            if config.training.hmc:
                experiment_label += "hmc"
        else:
            # add properties specific to Bayesopt
            if optimization_strategy == OptimizationStrategy.BAYESIAN:
                if label_param == "acquisition":
                    experiment_label += f"{config.bayesopt.acquisition}"
                elif label_param == "batch_strategy":
                    experiment_label += f"{config.bayesopt.batch_strategy.value}"
                elif label_param == "batch":
                    experiment_label += f"batch{config.bayesopt.batch}"
            # add properties specific to Zoomopt
            elif optimization_strategy == OptimizationStrategy.ZOOM:
                assert config.zoomopt is not None
                if label_param == "shrinking_factor":
                    experiment_label += f"({config.zoomopt.shrinking_factor})"
                elif label_param == "batch":
                    experiment_label += f"batch{config.zoomopt.batch}"

    return experiment_label


def load_experiment_df(experiment_dir: Path, loop_config: Type[OptimizerConfig]) -> pd.DataFrame:  # pragma: no cover
    """Return a DataFrame with accumulated observations from each sub-run in this directory. Each sub-directory
    in the folder `experiment_dir` is assumed to correspond to a single optimization run (with possibly
    different random seeds). Adds a column to the DataFrame to indicate sub-run ID (the ID is arbitrary).
    """
    assert experiment_dir.exists(), f"A directory at {experiment_dir} must exist."
    assert experiment_dir.is_dir(), f"A directory at {experiment_dir} must exist, but is not a directory."

    # Get the input names
    input_names: List[str] = _get_input_names(experiment_dir, loop_config)

    # Get seeded runs
    seeded_runs = _get_seeded_runs(experiment_dir)

    experiment_dfs = []
    for subrun_name, subrun_dir in seeded_runs.items():
        subrun_df: pd.DataFrame = load_seeded_subrun_df(subrun_dir, input_names=input_names)
        subrun_df[SEED_COLUMN] = subrun_name
        experiment_dfs.append(subrun_df)

    if len({len(one_seed_subrun_df) for one_seed_subrun_df in experiment_dfs}) != 1:
        logging.warning(f"Not all subruns in {experiment_dir} have the same length.")

    return pd.concat(experiment_dfs)  # type: ignore


def validate_simulator_settings_same(configs: List[OptimizerConfig]) -> None:  # pragma: no cover
    """
    Validates that the fields corresponding to the simulator settings are the same in all configs for all
    runs (experiments) being compared.

    Args:
        configs (List[OptimizerConfig]): List of configs for each run (experiment).
    """
    assert len(configs) > 0
    for config in configs:
        for simulator_config_attr in config.get_consistent_simulator_fields():
            if getattr(config, simulator_config_attr) != getattr(configs[0], simulator_config_attr):
                raise ValueError(
                    f"The value of {simulator_config_attr} is different in experiment directories given. "
                    f"It's: {getattr(config, simulator_config_attr)} vs {getattr(configs[0], simulator_config_attr)}"
                )


def load_config_from_expt_dir(experiment_dir: Path, loop_config: Type[OptimizerConfig]) -> OptimizerConfig:
    """
    Locate a config file in experiment_dir or one of its subdirectories (for a per-seed config).
    Config files are now normally in seed subdirectories, as they contain seed values.
    """
    config_files = sorted(experiment_dir.glob(f"*/seed*/{CONFIG_FILENAME}")) or [experiment_dir / CONFIG_FILENAME]
    config_file = config_files[0]
    if not config_file.exists():
        raise FileNotFoundError(f"Cannot find {CONFIG_FILENAME} at or under {experiment_dir}")  # pragma: no cover
    return cast(loop_config, simple_load_config(config_file, config_class=loop_config))  # type: ignore


def load_combined_df(
    experiment_dirs: List[Path],
    loop_config: Type[OptimizerConfig],
    experiment_label_params: List[str],
    styled_subset_params: List[str] = [],
) -> pd.DataFrame:  # pragma: no cover
    """Return a DataFrame with observations from each run specified in run_dirs. The returned DataFrame
    will have additional columns for: run name, sub-run id and batch number. Here, a sub-run is a single optimization
    run/experiment where multiple batches are collected. A run is a collection of those sub-runs (with different
    random initialisations) that share the same model/optimization configuration.
    """
    dfs = []
    for run_dir in experiment_dirs:
        run_df: pd.DataFrame = load_experiment_df(run_dir, loop_config)
        config = load_config_from_expt_dir(run_dir, loop_config)
        run_name = load_experiment_label(config, experiment_label_params)
        for styled_subset_param in styled_subset_params:
            config_dict = config.dict()
            # TODO: currently only acquisition, batch_strategy and optimization_strategy are allowed
            #  as styled_subset_params, hence indexing config by bayesopt makes sense, but this won't
            #  necessarily hold if this list is increased.
            if styled_subset_param == "acquisition":
                styled_subset_val = config_dict["bayesopt"][styled_subset_param]
            elif styled_subset_param == "batch_strategy":
                styled_subset_val = config_dict["bayesopt"][styled_subset_param].value
            elif styled_subset_param == "optimization_strategy":
                styled_subset_val = config_dict[styled_subset_param]
            else:
                raise ValueError(
                    f"styled_subset_param must be one of "
                    f"[acquisition, batch_strategy, optimization_strategy]."
                    f"Found {styled_subset_param}"
                )
            run_df[styled_subset_param] = styled_subset_val
        run_df[RUN_NAME_COLUMN] = run_name
        dfs.append(run_df)
    return pd.concat(dfs)  # type: ignore


def plot_predicted_optimum_covergence(
    arg_list: Optional[List[str]], loop_config: Type[OptimizerConfig]
) -> None:  # pragma: no cover
    """
    Main entry point for plotting combined results.
    :param arg_list: command-line arguments
    :param loop_config: OptimizerConfig subclass that has been used
    """
    args = create_parser().parse_args(arg_list)
    logging_to_stdout()
    styled_subset_params = args.styled_subset_param if len(args.styled_subset_param) > 0 else []
    # Load the optima location data
    combined_df = load_combined_df(
        experiment_dirs=args.experiment_dirs,
        loop_config=loop_config,
        experiment_label_params=args.experiment_labels,
        styled_subset_params=styled_subset_params,
    )

    # Clip the number of batches if max_batch_num specified
    combined_df = (
        combined_df[combined_df[BATCH_COLUMN] <= args.max_batch_number] if args.max_batch_number else combined_df
    )
    # Load the config (as given in run_loop_multiple_runs) for each experiment. Then assert that the
    # parts corresponding to simulator are the same.
    configs = [load_config_from_expt_dir(Path(experiment_dir), loop_config) for experiment_dir in args.experiment_dirs]
    assert len(configs) > 0

    # Validate all settings for the simulator are  the same in all the configs:
    validate_simulator_settings_same(configs)  # type: ignore # auto
    # Pick one config:
    config = configs[0]

    # Create a simulator from config
    simulator = cast(SimulatorBase, config.get_simulator())

    # Replicate each suggested optimum args.num_simulator_samples_per_optimum times.
    combined_df_replicated = pd.DataFrame(
        np.repeat(combined_df.values, args.num_simulator_samples_per_optimum, axis=0)  # type: ignore # auto
    )  # type: ignore # auto
    combined_df_replicated.columns = combined_df.columns
    optima_locations_array = combined_df_replicated[simulator.parameter_space.parameter_names].to_numpy(
        np.float64  # type: ignore
    )
    combined_df_replicated[OBJECTIVE_COLUMN] = simulator.sample_objective(optima_locations_array)

    # Determine the output scale for the plot
    if args.output_scale is not None:
        output_scale = args.output_scale
    else:
        output_scale = "log" if (combined_df_replicated[OBJECTIVE_COLUMN] > 0).all() else "symlog"  # type: ignore

    fig, _ = plot_objective_distribution_convergence(
        combined_df_replicated,
        objective_col=OBJECTIVE_COLUMN,
        batch_num_col=BATCH_COLUMN,
        run_col=RUN_NAME_COLUMN,
        style_cols=styled_subset_params,
        yscale=output_scale,
    )
    assert fig is not None
    # Possibly add title
    if args.title:  # type: ignore
        fig.suptitle(args.title)
    # Save the plot:
    args.output_dir.mkdir(exist_ok=True)
    output_path = args.output_dir / "styled_groups.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    run = Run.get_context()
    if not isinstance(run, _OfflineRun):
        fig.tight_layout()
        logging.info("Logging convergence plot to AML")
        run.log_image(name="styled_groups", plot=plt)
    plt.close(fig)
