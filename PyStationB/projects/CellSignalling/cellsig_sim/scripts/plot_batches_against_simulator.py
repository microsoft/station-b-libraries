# TODO: A docstring is missing. What does this script do and how does one execute it?
# TODO: It seems that this script is specific to the cell signalling strand. Should it be moved to
#  ``CellSignalling/scripts``?
# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import argparse
import enum
from pathlib import Path

import abex.settings
import numpy as np
from cellsig_sim.simulations import FourInputCell, ThreeInputCell
from abex.plotting.composite_core import plot_multidimensional_function_slices
from abex.simulations import SimulatedDataGenerator

from abex.scripts.plot_convergence import RunResultsConfig, load_batches_with_run_and_batch_names

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

BATCH_COLUMN = "Batch Number"
# RUN_NAME_COLUMN = "Run Name"


@enum.unique
class SimulatorType(enum.Enum):
    FOUR_INPUT_CELL_NO_GROWTH = enum.auto()
    FOUR_INPUT_CELL_WITH_GROWTH = enum.auto()
    THREE_INPUT_CELL_NO_GROWTH = enum.auto()
    THREE_INPUT_CELL_WITH_GROWTH = enum.auto()

    @classmethod
    def get_data_generator(cls, sim_type: "SimulatorType"):  # pragma: no cover
        if sim_type is cls.FOUR_INPUT_CELL_NO_GROWTH:
            return SimulatedDataGenerator(FourInputCell(use_growth_in_objective=False))
        elif sim_type is cls.FOUR_INPUT_CELL_WITH_GROWTH:
            return SimulatedDataGenerator(FourInputCell(use_growth_in_objective=True))
        elif sim_type is cls.THREE_INPUT_CELL_NO_GROWTH:
            return SimulatedDataGenerator(ThreeInputCell(use_growth_in_objective=False))
        elif sim_type is cls.THREE_INPUT_CELL_WITH_GROWTH:
            return SimulatedDataGenerator(ThreeInputCell(use_growth_in_objective=True))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot slices from a simulator against the optimum in the batches collected. "
        "Scatter the observed point in batches over the slices."
    )
    parser.add_argument(
        "--results_config_file",
        type=Path,
        required=True,
        help="OptimizerConfig file describing which run and batch different files correspond to.",
    )

    parser.add_argument(
        "--bayesopt_config_file",
        type=Path,
        required=True,
        help="OptimizerConfig file for Bayesopt describing, inter alia, which parameters are in log-space.",
    )

    parser.add_argument("--simulator", type=str, default=None, help="The title for the plot.")

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


def load(file_path: Path) -> RunResultsConfig:  # pragma: no cover
    config = RunResultsConfig()
    # TODO: this method does not appear to exist.
    config.load(file_path)  # type: ignore
    return config


def main(args):  # pragma: no cover
    config = load(args.results_config_file)
    abex_config = abex.settings.load(args.abex_config_file)
    simulator_type = SimulatorType[str(args.simulator).upper()]
    data_generator = SimulatorType.get_data_generator(simulator_type)

    assert isinstance(
        data_generator, SimulatedDataGenerator
    ), "DataGenerator must be a simulator if slices from it are to be plotted"
    run_df = load_batches_with_run_and_batch_names(config)

    input_names = data_generator.parameter_space.parameter_names
    bounds_array = np.array(data_generator.parameter_space.get_bounds())
    # Get the best point observed in data
    max_idx = run_df[config.objective_column].argmax()
    slice_point = run_df[input_names].iloc[max_idx].values
    # Map the data in dataframes to a list of arrays (one per collected batch)
    obs_points = [
        run_df[run_df[BATCH_COLUMN] == batch_num][input_names].values  # type: ignore # auto
        for batch_num in range(run_df[BATCH_COLUMN].max() + 1)  # type: ignore # auto
    ]

    # Plot simulated objective function
    fig, _ = plot_multidimensional_function_slices(
        func=data_generator.simulate_experiment,
        slice_loc=slice_point,
        bounds=bounds_array,
        input_names=input_names,
        obs_points=obs_points,  # type: ignore # auto
        input_scales=[abex_config.data.input_plotting_scales[name] for name in input_names],  # type: ignore # auto
        output_scale=abex_config.data.output_settings.plotting_scale,
    )
    assert fig is not None
    # Possibly add title
    if args.title:
        fig.suptitle(args.title)
    # Get output_path:
    if args.output_path:
        output_path = args.output_path
    else:
        filename = f"simulated_slices_plot_{config.name}.png"
        output_path = args.results_dir / filename
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    args = create_parser().parse_args()
    main(args)
