# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import argparse
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from gp.benchmarking.candidate_calculator_factory import BatchMethod
from gp.benchmarking.convergence_plotting import (
    plot_running_best_traces_dist_comparison,
    plot_running_best_traces_samples_comparison,
    plot_ranking_comparison,
    plot_running_best_against_time_dist_comparison,
)


sns.set()


@dataclass
class ExperimentResult:
    """Class for keeping track of experiment results"""

    method_name: str
    # observed_inputs: List[np.ndarray]
    observed_outputs: List[np.ndarray]
    observed_times: List[np.ndarray]
    batch_size: int

    @property
    def num_experiments(self) -> int:
        assert len(self.observed_outputs) == len(self.observed_times)
        return len(self.observed_outputs)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plotting convergence of results.")
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--yscale", choices=["log", "linear", "symlog"], default="linear", type=str)
    parser.add_argument("--metric", choices=["mean", "median"], default="mean", type=str)
    parser.add_argument(
        "--ci", type=float, default=None, help="Confidence interval to plot. Standard deviation if None (default)."
    )
    parser.add_argument(
        "--plot_metric_ci",
        action="store_true",
        help="If set to True, plot the std./confidence interval in the metric (mean/median), rather"
        "than in the samples.",
    )
    parser.add_argument("--save_dir", type=Path, default=Path("outputs/Plots"))
    parser.add_argument(
        "--truncate_to_shortest_num_experiments",
        action="store_true",
        help="If set to True, the number of experiments used in the comparison will be set to the minimum of num. "
        "experiments in all the folders with configurations to be compared",
    )

    return parser


def _load_run_results(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    # X = np.loadtxt(run_dir / "X.csv")
    Y = np.loadtxt(run_dir / "Y.csv")
    times = np.loadtxt(run_dir / "times.csv")
    return Y, times


def _get_unique_plot_suffix(folders: List[Path]) -> str:
    return "__".join("_".join(folder.parts[-2:]) for folder in folders)[:220]


def main(args: argparse.Namespace):
    #  Get experiment configs:
    experiments_results: List[ExperimentResult] = []
    for experiment_dir in args.folders:
        runs: List[Path] = list(experiment_dir.glob("seed*"))
        config_file = runs[0] / "config.json"
        config_dict = json.load(config_file.open("r"))
        experiment_method_name = repr(BatchMethod(config_dict["batch_method"]))
        batch_size = int(config_dict["batch_size"])

        #  Get an iterable of tuples of the form (Y, times) for each experiment run
        run_results_tuples = (_load_run_results(run_dir) for run_dir in runs)
        #  Transform into three lists of Y, and times respectively
        Y_list, times_list = list(zip(*run_results_tuples))

        exp_result = ExperimentResult(
            observed_outputs=Y_list,
            observed_times=times_list,
            method_name=experiment_method_name,
            batch_size=batch_size,
        )
        experiments_results.append(exp_result)
    assert (
        len({exp_results.batch_size for exp_results in experiments_results}) == 1
    ), "All experiments should have the same batch size"

    if args.truncate_to_shortest_num_experiments:
        shortest_num_exps = min((exp_result.num_experiments for exp_result in experiments_results))
        print(f"Plotting results using {shortest_num_exps} runs per experiment")
        for exp_result in experiments_results:
            exp_result.observed_outputs = exp_result.observed_outputs[:shortest_num_exps]
            exp_result.observed_times = exp_result.observed_times[:shortest_num_exps]

    #  Make the plots
    with sns.axes_style("whitegrid"):
        colors = sns.color_palette("husl", len(experiments_results))

        # Make the distribution of cum. minimum comparison
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_running_best_traces_dist_comparison(
            Y_runs_list=[exp_results.observed_outputs for exp_results in experiments_results],
            names=[exp_results.method_name for exp_results in experiments_results],
            colors=colors,
            ax=ax,
            batch_size=batch_size,
            confidence_interval=args.ci,
            metric=args.metric,
            plot_estimator_ci=args.plot_metric_ci,
        )
        ax.set_yscale(args.yscale)
        fig.savefig(
            args.save_dir / f"convergence_dist__{_get_unique_plot_suffix(args.folders)}.png",
            bbox_inches="tight",
        )
        plt.close()

        # Make the plot showing samples of cumulative minimum (one per run of each experiment)
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_running_best_traces_samples_comparison(
            Y_runs_list=[exp_results.observed_outputs for exp_results in experiments_results],
            names=[exp_results.method_name for exp_results in experiments_results],
            colors=colors,
            ax=ax,
            batch_size=batch_size,
            metric=args.metric,
        )
        ax.set_yscale(args.yscale)
        fig.savefig(args.save_dir / f"samples_dist__{_get_unique_plot_suffix(args.folders)}.png", bbox_inches="tight")

        # Plot the ranks of each algo
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_ranking_comparison(
            Y_runs_list=[exp_results.observed_outputs for exp_results in experiments_results],
            names=[exp_results.method_name for exp_results in experiments_results],
            colors=colors,
            ax=ax,
            batch_size=batch_size,
        )
        fig.savefig(args.save_dir / f"ranks_per_step__{_get_unique_plot_suffix(args.folders)}.png", bbox_inches="tight")

        # Plot the convergence against time
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_running_best_against_time_dist_comparison(
            Y_runs_list=[exp_results.observed_outputs for exp_results in experiments_results],
            times=[exp_results.observed_times for exp_results in experiments_results],
            names=[exp_results.method_name for exp_results in experiments_results],
            colors=colors,
            ax=ax,
            batch_size=batch_size,
            confidence_interval=args.ci,
            plot_estimator_ci=args.plot_metric_ci,
        )
        fig.savefig(
            args.save_dir / f"convergence_vs_time__{_get_unique_plot_suffix(args.folders)}.png", bbox_inches="tight"
        )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args.save_dir.mkdir(exist_ok=True, parents=True)

    main(args)
