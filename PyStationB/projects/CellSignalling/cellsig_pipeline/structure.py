# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Global variables, specifying the structure of the created directories.

The data storage consists of different layers: we keep an experimental track, that is a series of related experiments,
tagged appropriately and kept in BCKG.
Experimental tracks are kept in `EXPERIMENTS_DIR` directory (see below) and are differentiated by ABEX_CONFIG (Bayesian
Optimization settings) and DATA_CONFIG (initial experiment to run ABEX and tags to be used in subsequent experiments
associated to the same track). Each experimental track directory contains these two configs, as well as data retrieved
from BCKG, pre-processed (characterization and combinatorics) and then processed by ABEX.

Apart from these two configuration files, some settings are shared for all experimental tracks (e.g. what signals and
what inputs we are using). For these "global" files see CONFIGS_DIR and related (static) paths.

Todo:
    these values should be used as the *defaults* of arguments to different steps. However, in many places these
    steps don't accept configuration objects and simply refer to these.
    From the perspective of time, I consider it a bad design.
"""
from pathlib import Path

EXPERIMENTS_DIR = Path("Experiments")  # Where we keep all the experimental tracks (in separate directories)

# Data structure
DATA_DIR = "Data"  # Where we keep the downloaded data

# Global configuration files (shared by all experimental tracks)
CONFIGS_DIR = Path("Configs")
CHARACTERIZATION_CONFIG_PATH: Path = CONFIGS_DIR / "characterization.yml"
COMBINATORICS_CONFIG_PATH: Path = CONFIGS_DIR / "combinatorics.yml"
ANTHA_CONFIG_PATH: Path = CONFIGS_DIR / "antha.yml"

# TODO: Replace all references to this with `combinatorics.CombinatoricsConfig.pairing`.
OBSERVATION_ID: str = "ObservationID"  # Each sample we observe corresponds to two wells, linked by ObservationID


# Local configuration files (separate for each experimental track)
ABEX_CONFIG = "abex_config.yml"
DATA_CONFIG = "data_config.yml"


def experiment_dir(experiment_name: str) -> Path:  # pragma: no cover
    """Each experimental track is identified by a unique name and has its own directory."""
    return EXPERIMENTS_DIR / experiment_name


def config_abex_path(experiment_name: str) -> Path:  # pragma: no cover
    """Path to the ABEX config for a given experimental track."""
    return experiment_dir(experiment_name) / ABEX_CONFIG


def config_data_path(experiment_name: str) -> Path:  # pragma: no cover
    """Path to the config with data retrieval and preprocessing settings, for a given experimental track."""
    return experiment_dir(experiment_name) / DATA_CONFIG


def data_characterization_dir(experiment_name: str) -> Path:  # pragma: no cover
    """Directory where we store the data after characterization, for a given experimental track."""
    return experiment_dir(experiment_name) / DATA_DIR / "Characterization"


def data_abex_input_dir(experiment_name: str) -> Path:  # pragma: no cover
    """Path with the data ready to be fed into ABEX, for a given experimental track."""
    return experiment_dir(experiment_name) / DATA_DIR / "ABEX-Inputs"


def iteration_name(iteration: int) -> str:  # pragma: no cover
    """In each directory we index iteration-specific files by the step number. We pad the number with zeros, so the
    order is lexicographical.

    Raises:
        ValueError, if ``step`` is not between 0 and 99.
    """
    if iteration < 0 or iteration > 99:
        raise ValueError(f"Step {iteration} out of bounds [0, 99].")

    return "iteration-{:02}".format(iteration)


def data_characterization_path(experiment_name: str, iteration: int) -> Path:  # pragma: no cover
    """Path with characterized data."""
    iteration_csv: str = f"{iteration_name(iteration)}.csv"
    return data_characterization_dir(experiment_name) / iteration_csv


def data_abex_input_path(experiment_name: str, iteration: int) -> Path:  # pragma: no cover
    """Path with data after the combinatorics (ready to be used by ABEX)."""
    iteration_csv: str = f"{iteration_name(iteration)}.csv"
    return data_abex_input_dir(experiment_name) / iteration_csv


def data_abex_results_dir(experiment_name: str) -> Path:  # pragma: no cover
    """Where the ABEX results are stored (all iterations)."""
    return experiment_dir(experiment_name) / "Results"


def data_abex_results_iteration_dir(experiment_name: str, iteration: int) -> Path:  # pragma: no cover
    """Where the ABEX results are stored (for a specific iteration)."""
    return data_abex_results_dir(experiment_name) / iteration_name(iteration)
