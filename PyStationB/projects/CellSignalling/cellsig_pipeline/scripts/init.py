# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Initializes the experimental tracks.

Namely, this creates the required directory structure and creates config templates (to be adjusted by the user).

Usage:
    python scripts/init.py --names Experiment1 Experiment2 ...

Given names of the experiment can be arbitrary. Created experiments should be visible in the Experiments/ directory
"""
import argparse
from typing import Set

from abex.data_settings import InputParameterConfig
from abex.settings import OptimizerConfig
import cellsig_pipeline
import cellsig_pipeline.settings as settings
import cellsig_pipeline.structure as ps


def parse_args() -> Set[str]:
    """Returns the names of the experiments to be created."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--names", nargs="+", help="Names of the experiments to be initialized.", required=True)

    args = parser.parse_args()

    return set(args.names)


def prepare_data_config(name: str) -> None:
    """Prepares a data config with the default four tags we use in our workflow."""
    default_tags = [
        settings.Tag(name=name, value=value, level=settings.TagLevel.EXPERIMENT)
        for name, value in [
            ("TrackName", name),
            ("Batch", -1),
            ("TrackNumber", 1),
            ("Project", "CellSignalling"),
        ]
    ]

    default_settings = settings.DataSettings(initial_experiment_id="Fill this.", tags=default_tags)

    settings.dump_settings(ps.config_data_path(name), default_settings)


def prepare_abex_config(name: str) -> None:
    """Creates a default ABEX config with the inputs used in our project and with the data folder matching
    our structure."""
    # TODO: We create a config and then modify it. It would be cleaner to construct it in a principled way.
    config = OptimizerConfig()
    config.data.folder = ps.data_abex_input_dir(name)
    config.results_dir = "This value is overwritten at each iteration. You don't need to modify it."  # type: ignore

    # TODO: Hard-coded values. Shall we store them in another configuration file?
    config.data.inputs = {  # type: ignore # auto
        "Arabinose": InputParameterConfig(lower_bound=0.01, upper_bound=200, unit="mM", log_transform=True),
        "ATC": InputParameterConfig(lower_bound=0.25, upper_bound=5000, unit="ng/ml", log_transform=True),
        "Con": InputParameterConfig(lower_bound=1, upper_bound=2e4, unit="nM", log_transform=True),
    }

    config.data.output_column = "Objective"
    config.data.output_settings.log_transform = True

    config.model.add_bias = True

    settings.dump_settings(ps.config_abex_path(name), config)


def create_experiment(name: str) -> bool:
    """Creates the directory structure for experiment ``name``.

    Args:
        name: experiment name. It should uniquely identify the experiment.

    Returns:
        bool, True if the operation was successful. Otherwise, False.
    """
    # If the experiment already exists, we do nothing -- we don't want to overwrite it!
    if ps.experiment_dir(name).exists():
        return False

    ps.experiment_dir(name).mkdir(parents=True, exist_ok=False)

    prepare_abex_config(name)
    prepare_data_config(name)

    return True


if __name__ == "__main__":
    experiment_names: Set[str] = parse_args()
    cellsig_pipeline.logger.info(f"Asked to initialize the following experiments: {experiment_names}")

    successful: Set[str] = set()

    for experiment_name in experiment_names:
        success: bool = create_experiment(experiment_name)
        if success:
            successful.add(experiment_name)

    if successful:
        cellsig_pipeline.logger.info(
            f"Experiments {successful} were created successfully. " f"Remember to modify the configuration files."
        )

    if experiment_names != successful:
        cellsig_pipeline.logger.warning(
            f"The following experiments could not be created " f"{experiment_names.difference(successful)}."
        )

    cellsig_pipeline.logger.info("Initialization complete.")
