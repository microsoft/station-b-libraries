# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Convergence plotting script for comparing multiple experiments that allows for aggregating over multiple randomised
sub-runs (runs with same configurations, but a different random seed). The randomly seeded sub-runs are expected
as sub-directories of the experiment directories.

Example command:
python plot_convergence_multiple_runs.py --experiment_dirs /path/to/experiment/one /path/to/experiment/two
  --experiment_labels "Experiment 1" "Experiment 2" --objective_column "Output"
  --init_data_filename "initial_batch.csv"
"""
import argparse
import logging
from functools import reduce
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from abex.plotting.convergence_plotting import (
    ConvergencePlotStyles,
    plot_multirun_convergence,
    plot_multirun_convergence_per_sample,
)

BATCH_COLUMN = "Batch Number"
RUN_NAME_COLUMN = "Experiment Name"
SEED_COLUMN = "Sub-run Number"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot convergence over several iterations of Bayesian Optimization, for possibly multiple runs."
        "Assumes one file corresponds to one batch collected."
    )
    parser.add_argument(
        "--experiment_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="A sequence of directories corresponding to different configurations for an experiment "
        "(different runs). Each directory should contain multiple sub-directories with multiple sub-runs"
        "corresponding to different random seeds.",
    )
    parser.add_argument(
        "--experiment_labels",
        type=str,
        nargs="+",
        help="A sequence of names to give to each experiment (collection of sub-runs). These will be used to "
        "label the experiments on resulting plots. These should appear in the"
        "same order as --experiment_dirs. If not specified, folder names will be used as experiment labels.",
    )
    parser.add_argument(
        "--objective_column",
        type=str,
        default="Crosstalk Ratio",
        help="The name of the objective column in data files.",
    )
    parser.add_argument(
        "--init_data_filename",
        type=Path,
        required=True,
        help="The filename for the initial data file (the other files in each seed-run subdirectory "
        "will be treated as batches).",
    )
    parser.add_argument(
        "--results_dir", type=Path, default=Path("Results"), help="The directory in which to save the resulting plot."
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="If specified, the resulting plot will be saved at this location "
        "(otherwise a plot name will be generated).",
    )
    parser.add_argument("--title", type=str, default=None, help="The title for the plot.")
    parser.add_argument(
        "--no_boxplot",
        action="store_true",
        help="Whether to remove the boxplot for the final plot (useful if the plot gets too cluttered).",
    )
    parser.add_argument(
        "--no_scatter",
        action="store_true",
        help="Whether to remove the scatter points for the final plot (useful if the plot gets too cluttered).",
    )
    parser.add_argument(
        "--max_batch_number",
        type=int,
        default=None,
        help="Whether to clip the x-axis to a given number of batches.",
    )
    parser.add_argument(
        "--make_per_sample_plot",
        action="store_true",
        help="Whether to make a per-sample plot as well in which x-axis is 'num. samples collected' rather than "
        "'num. batches collected'.",
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
        "--styled_lines",
        action="store_true",
        help="Whether to give each line a different style (dashed, solid, double-dashed, ...) in addition to it "
        "being a different colour.",
    )
    parser.add_argument(
        "--styled_subset",
        type=str,
        action="append",
        nargs="+",
        help="Style a subset (specified by run name) of the plotted traces to distinguish them from other traces.",
    )
    parser.add_argument(
        "--style_category_name",
        type=str,
        default="Category",
        help="Name to use on the legend for the style categories.",
    )
    parser.add_argument(
        "--styled_subset_names",
        type=str,
        nargs="+",
        default=None,
        help="Names for each consecutive subset that is differently styled.",
    )
    parser.add_argument(
        "--plot_style",
        type=str,
        default="boxplot",
        choices=[e.value for e in ConvergencePlotStyles],  # ["boxplot", "line"],
        help="Type of convergence plot (line, or slightly offset point-plot with error bars).",
    )
    return parser


def load_seeded_subrun_df(subrun_dir: Path, init_data_filename: str) -> pd.DataFrame:  # pragma: no cover
    """Return a DataFrame with the observations from this 'sub-run'. This funciton iterates over the files
    in this directory, and assumes they correspond to consecutive batches in a single optimization run in
    lexicographic order. Adds a column to the DataFrame to indicate batch number.
    """
    init_data_file = subrun_dir / init_data_filename
    assert init_data_file.is_file(), f"Initial data file not found at: {init_data_file}"
    batch_files: List[Path] = [child for child in subrun_dir.glob("**/*") if child.is_file() and child.suffix == ".csv"]
    # Only keep csv files in folder that are not initial data files:
    batch_files.remove(init_data_file)
    # Sort in lexicographic order
    batch_files = sorted(batch_files)
    # Load into a DF
    batch_dfs = list(map(pd.read_csv, batch_files))
    batch_dfs.insert(0, pd.read_csv(init_data_file))  # Prepend initial data at index 0
    for i, batch_df in enumerate(batch_dfs):
        batch_df[BATCH_COLUMN] = i  # type: ignore # auto
    if len({len(batch_df) for batch_df in batch_dfs}) != 1:  # type: ignore # auto
        logging.warning(f"Batches in subrun at {subrun_dir} have unequal sizes.")
    return pd.concat(batch_dfs)  # type: ignore # auto


def load_experiment_df(experiment_dir: Path, init_data_filename: str) -> pd.DataFrame:  # pragma: no cover
    """Return a DataFrame with accumulated observations from each sub-run in this directory. Each sub-directory
    in the folder experiment_dir is assumed to correspond to a single optimization run (with possibly
    different random seeds). Adds a column to the DataFrame to indicate sub-run ID (the ID is arbitrary).
    """
    assert experiment_dir.exists(), f"A directory at {experiment_dir} must exist."
    assert experiment_dir.is_dir(), f"A directory at {experiment_dir} must exist, but is not a directory."
    # Get all subdirectories (ASSUME they correspond to seeded runs)
    subrun_dirs_in_folder = [child for child in experiment_dir.glob("**/*") if child.is_dir()]
    experiment_dfs = []
    for subrun_id, subrun_dir in enumerate(subrun_dirs_in_folder):
        subrun_df = load_seeded_subrun_df(subrun_dir, init_data_filename)
        subrun_df[SEED_COLUMN] = subrun_id
        experiment_dfs.append(subrun_df)
    if len({len(one_seed_subrun_df) for one_seed_subrun_df in experiment_dfs}) != 1:
        logging.warning(f"Not all subruns in {experiment_dir} have the same length.")
    return pd.concat(experiment_dfs)


def load_combined_df(
    experiment_dirs: List[Path], experiment_labels: List[str], init_data_filename: str
) -> pd.DataFrame:  # pragma: no cover
    """Return a DataFrame with observations from each run specified in run_dirs. The returned DataFrame
    will have additional columns for: run name, sub-run id and batch number. Here, a sub-run is a single optimization
    run/experiment where multiple batches are collected. A run is a collection of those sub-runs (with different
    rando initialisations) that share the same model/optimization configuration.
    """
    dfs = []
    for run_name, run_dir in zip(experiment_labels, experiment_dirs):
        run_df = load_experiment_df(run_dir, init_data_filename)
        run_df[RUN_NAME_COLUMN] = run_name
        dfs.append(run_df)
    return pd.concat(dfs)


def get_experiment_labels(args) -> List[str]:  # pragma: no cover
    """Returns experiment labels, inferring them from `args.experiment_dirs`, if `args.experiment_labels` not
    explicitly provided.

    Raises:
        ValueError, if the labels don't match the experiment directories
    """
    experiment_labels: List[str]
    # If experiment_labels specified, assert the length matches the number of directories given
    if args.experiment_labels:
        if len(args.experiment_dirs) != len(args.experiment_labels):
            raise ValueError(
                f"Number of directories ({len(args.experiment_dirs)}) does not match the number of experiment "
                f"names ({len(args.experiment_labels)}).\nDirectories: {args.experiment_dirs}\n"
                f"Experiment names: {args.experiment_labels}"
            )
        experiment_labels = args.experiment_labels
    else:
        # If not specified, use folder names
        experiment_labels = list(map(lambda exp_dir: exp_dir.stem, args.experiment_dirs))
    # Ensure names unique:
    if len(experiment_labels) != len(set(experiment_labels)):
        raise ValueError(
            f"All experiment names must be unique, but are:\n{experiment_labels}\n"
            "Use the `--experiment_labels` flag if experiment directories don't have unique names."
        )

    return experiment_labels


def main(args):  # pragma: no cover
    # - Get experiment (run) names
    experiment_labels: List[str] = get_experiment_labels(args)

    # Load the data
    combined_df = load_combined_df(
        experiment_dirs=args.experiment_dirs,
        experiment_labels=experiment_labels,
        init_data_filename=args.init_data_filename,
    )

    # Assert all entries in objective columns non-zero
    assert args.objective_column in combined_df.columns
    assert not combined_df[args.objective_column].isnull().any()

    # Clip the number of batches if max_batch_num specified
    combined_df_clipped = (
        combined_df[combined_df[BATCH_COLUMN] <= args.max_batch_number] if args.max_batch_number else combined_df
    )
    if args.styled_subset:
        assert args.experiment_labels, "Experiment names must be given if style subsets specified"
        # Assert all the styled subsets cover all of the experiments
        assert reduce(lambda a, b: a.union(b), args.styled_subset, set()) == set(args.experiment_labels)

        # Assert there is the right number of styled subsets (no duplicates between subsets)
        assert sum([len(subset) for subset in args.styled_subset]) == len(args.experiment_labels)

        if args.styled_subset_names is None:
            # If styled subset names not given, set generic default names
            args.styled_subset_names = [f"Category {i}" for i in range(len(args.styled_subset))]
        assert len(args.styled_subset) == len(args.styled_subset_names)
        # Construct result_subdir_name to style subset name mapping
        experiment_to_style_subset = dict()
        for styled_subset, subset_name in zip(args.styled_subset, args.styled_subset_names):
            for result_subdir_name in styled_subset:
                experiment_to_style_subset[result_subdir_name] = subset_name
        # If styling a subset, add style column
        combined_df_clipped[args.style_category_name] = combined_df_clipped[RUN_NAME_COLUMN].map(  # type: ignore # auto
            lambda s: experiment_to_style_subset[s]
        )

    if args.styled_lines and args.styled_subset:
        raise ValueError("Both styled_lines and styled_subset can't be specified at the same time.")
    if args.styled_lines:
        style_cols = [RUN_NAME_COLUMN]
    elif args.styled_subset & isinstance(args.style_category_name, str):
        assert isinstance(args.style_category_name, str)
        style_cols = [args.style_category_name]
    else:
        style_cols = None
    # Determine the output scale for the plot
    if args.output_scale is not None:
        output_scale = args.output_scale
    elif (combined_df_clipped[args.objective_column] > 0).all():  # type: ignore # auto
        output_scale = "log"
    else:
        output_scale = "symlog"
    fig, _ = plot_multirun_convergence(
        combined_df_clipped,  # type: ignore # auto
        objective_col=args.objective_column,
        batch_num_col=BATCH_COLUMN,
        run_col=RUN_NAME_COLUMN,
        seed_col=SEED_COLUMN,
        add_boxplot=not args.no_boxplot,
        add_scatter=not args.no_scatter,
        style_cols=style_cols,
        plot_style=args.plot_style,
        yscale=output_scale,
    )
    assert fig is not None
    # Possibly add title
    if args.title:
        fig.suptitle(args.title)
    # Get output_path:
    if args.output_path:
        output_path = args.output_path
    else:
        filename = f"multirun_convergence_plot_{'__'.join(experiment_labels)}.png"
        output_path = args.results_dir / filename
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    if args.make_per_sample_plot:
        fig_per_sample, _ = plot_multirun_convergence_per_sample(
            combined_df,
            objective_col=args.objective_column,
            batch_num_col=BATCH_COLUMN,
            run_col=RUN_NAME_COLUMN,
            seed_col=SEED_COLUMN,
        )
        assert fig_per_sample is not None
        # Possibly add title
        if args.title:
            fig_per_sample.suptitle(args.title)
        filename = f"multirun_convergence_per_sample_plot_{'__'.join(experiment_labels)}.png"
        output_path = args.results_dir / filename
        fig_per_sample.savefig(output_path, bbox_inches="tight")
        plt.close(fig_per_sample)


if __name__ == "__main__":  # pragma: no cover
    args = create_parser().parse_args()
    main(args)
