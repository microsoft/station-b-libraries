# TODO: Missing script docstring. How should this script be run? What is this for?
# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import argparse

from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import yaml

from pydantic import BaseModel, Field

from abex.plotting import plot_convergence
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

BATCH_COLUMN = "Batch Number"
RUN_NAME_COLUMN = "Run Name"


class RunResultsConfig(BaseModel):
    """A class to collect configuration options for a run. Describes which files correspond to which batch number.

    Properties
        name: Name of the run. Used for plotting (labels etc).
        objective_column: Which column in data files corresponds to the objective
        folder: Which folder is the data located in
        init_data_files: List of files corresponding to the initial files (batch 0)
        batch_files: Dictionary mapping batch number to list of files corresponding to that batch.
            (Paths are relative to the directory specified in the folder field)
        batches_in_lexic_order: If True, ignore the batch_files field; instead, get all the files in the directory
            specified by the folder field (other than the ones specified in init_data_files),
            sort them lexicographically, and assume that they correspond to consecutive batches.
    """

    name: str = ""
    objective_column: Optional[str] = None
    folder: Path = Field(default_factory=Path)
    init_data_files: List[str] = Field(default_factory=list)
    batch_files: Dict[int, List[str]] = Field(default_factory=dict)
    batches_in_lexic_order: bool = False


def create_parser() -> argparse.ArgumentParser:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Plot convergence over several iterations of Bayesian Optimization, for possibly multiple runs."
        "Assumes one file corresponds to one batch collected."
    )
    parser.add_argument(
        "--config_files",
        type=Path,
        nargs="+",
        required=True,
        help="OptimizerConfig files describing which run and batch different files correspond to.",
    )
    parser.add_argument(
        "--results_dir", type=Path, default=Path("Results"), help="The directory in which to save the resulting plot."
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="If specified, the resulting path will be saved at this location "
        "(otherwise a plot name will be generated).",
    )
    parser.add_argument("--title", type=str, default=None, help="The title for the plot.")
    return parser


def load(file_paths: List[Path]) -> List[RunResultsConfig]:  # pragma: no cover
    configs = []
    for yaml_file_path in file_paths:
        with open(yaml_file_path) as f:
            parsed_file = yaml.safe_load(f)
        config = RunResultsConfig(**parsed_file)
        configs.append(config)

    return configs


def load_batches_with_run_and_batch_names(run_config: RunResultsConfig) -> pd.DataFrame:  # pragma: no cover
    # Get the initial_data
    init_batch_paths = list(map(lambda filename: run_config.folder / filename, run_config.init_data_files))
    init_batch_df = pd.concat(map(pd.read_csv, init_batch_paths))  # type: ignore # auto
    init_batch_df[BATCH_COLUMN] = 0

    # Get the DFs corresponding to batches
    run_dfs = [init_batch_df]

    if run_config.batches_in_lexic_order:
        # If the remaining batch files are in lexicographic order, get all the filenames in folder and sort:
        files_in_folder = [
            child for child in run_config.folder.glob("**/*") if child.is_file() and child.suffix == ".csv"
        ]
        # Get all csv files in folder that are not initial data files
        batch_files = list(set(files_in_folder) - set(init_batch_paths))
        # Sort in lexicographic order
        batch_files = sorted(batch_files)
        # Load into a DF
        batch_dfs = list(map(pd.read_csv, batch_files))
        for i, batch_df in enumerate(batch_dfs):
            batch_df[BATCH_COLUMN] = i + 1  # type: ignore # auto
        run_dfs.extend(batch_dfs)
    else:
        # Otherwise, get the files as specified by config
        for batch_num, files in run_config.batch_files.items():
            assert batch_num >= 0, "Batch number must be non-negative"
            batch_paths = map(lambda filename: run_config.folder / filename, files)
            batch_df = pd.concat(map(pd.read_csv, batch_paths))  # type: ignore # auto
            batch_df[BATCH_COLUMN] = batch_num
            run_dfs.append(batch_df)
    run_df_combined = pd.concat(run_dfs)
    run_df_combined[RUN_NAME_COLUMN] = run_config.name
    return run_df_combined


def main(args):  # pragma: no cover
    configs: List[RunResultsConfig] = load(args.config_files)
    # Assert all objective columns are the same:
    assert len(set(map(lambda run_config: run_config.objective_column, configs))) == 1

    all_runs_dfs = []
    # For every config, get the dataframes for each batch + initial data
    for run_config in configs:
        run_df = load_batches_with_run_and_batch_names(run_config)
        # Append the combined dataframe for this run to the list of all runs
        all_runs_dfs.append(run_df)

    combined_df = pd.concat(all_runs_dfs)
    fig, _ = plot_convergence(
        combined_df,
        objective_col=configs[0].objective_column,  # type: ignore # auto
        batch_num_col=BATCH_COLUMN,
        run_col=RUN_NAME_COLUMN,
    )
    assert fig is not None
    # Possibly add title
    if args.title:
        fig.suptitle(args.title)
    # Get output_path:
    if args.output_path:
        output_path = args.output_path
    else:
        filename = f"convergence_plot_{'__'.join([run_config.name for run_config in configs])}.png"
        output_path = args.results_dir / filename
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    args = create_parser().parse_args()
    main(args)
