# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Data preprocessing utilities. These are high-level utilities, wrapping around ``combinatorics`` and ``pybckg``
submodules.

Exports:
    prepare_data_for_abex, a high-level function doing all necessary preprocessing


Todo:
    The signatures of many of these functions could be extended -- they should use the constants defined in
    ``structure`` submodule as default values, not explicitly calling for them.
"""
import pathlib
import time
from typing import List, Optional

import cellsig_pipeline.combinatorics as combinatorics
import cellsig_pipeline.pybckg as pybckg
import cellsig_pipeline.settings as settings
import cellsig_pipeline.structure as structure
import pandas as pd


def _load_data_settings(experiment_name: str) -> settings.DataSettings:  # pragma: no cover
    """Loads data settings associated to an experimental track."""
    path: pathlib.Path = structure.config_data_path(experiment_name)
    return settings.load_settings(path, class_factory=settings.DataSettings)


def _download_characterized_csv(
    experiment_id: str,
    output_path: pathlib.Path,
    connection: pybckg.WetLabAzureConnection,
    config: Optional[combinatorics.CombinatoricsConfig] = None,
) -> None:  # pragma: no cover
    """Saves a CSV with characterization results (see ``pipeline.pybckg.experiment_to_dataframe``).

    Args:
        experiment_id: BCKG ID of an experiment
        output_path: where the CSV should be saved
        connection: a connection object
        config: combinatorics config specifying the inputs. If not provided, the global one is used
            (see ``pipeline.structure.COMBINATORICS_CONFIG_PATH``)

    Todo:
        Consider making this function public and making ``connection`` an optional argument
    """
    config2: combinatorics.CombinatoricsConfig = config or settings.load_settings(
        structure.COMBINATORICS_CONFIG_PATH, class_factory=combinatorics.CombinatoricsConfig
    )
    df = pybckg.experiment_to_dataframe(experiment_id, connection, config=config2)
    df.to_csv(output_path, index=False)
    time.sleep(0.5)  # Give the Disk I/O some time to save the data.


def _download_initial_batch(experiment_name: str, connection: pybckg.WetLabAzureConnection) -> None:  # pragma: no cover
    """This method pulls the initial batch, as specified in the config, if it doesn't already exist.

    This batch is named as if it had been collected at ``iteration = 0``.

    See:
        _download_characterized_csv, _download_later_batch
    """
    # If the file already exists, we don't need to pull it
    expected_location: pathlib.Path = structure.data_characterization_path(experiment_name=experiment_name, iteration=0)
    # Otherwise, we check the ID from the config and download the file.
    if not expected_location.exists():
        config = _load_data_settings(experiment_name)
        experiment_id: str = config.initial_experiment_id
        _download_characterized_csv(experiment_id=experiment_id, output_path=expected_location, connection=connection)


def _download_later_batch(
    experiment_name: str, iteration: int, connection: pybckg.WetLabAzureConnection
) -> None:  # pragma: no cover
    """This method pulls the batch collected at iterations 1, 2, ...

    Args:
        experiment_name: experiment track name
        iteration: iteration, must be at least 2 (for ``iteration=1``, use ``_download_initial_batch``)
        connection: a connection object
    """
    assert iteration >= 2, "If iteration=1, use _download_initial_batch()."
    # We want to retrieve the batch ``iteration-1``.
    expected_location: pathlib.Path = structure.data_characterization_path(
        experiment_name=experiment_name, iteration=iteration - 1
    )

    if not expected_location.exists():
        config = _load_data_settings(experiment_name)

        # Get the list of experiments corresponding to these tags, in chronological order
        experiment_ids: List[str] = connection.experiments_ids_in_a_track(config.set_of_tags)

        if len(experiment_ids) < 1:
            raise ValueError(
                f"There are not enough experiment with these tags. For a set of tags {config.set_of_tags}"
                f" the following experiments were found: {experiment_ids}. Check the tag names."
            )

        # For iteration = 1, we take the initial batch. For iteration = 2, we want to download the CSV corresponding
        # to iteration = 1. As Python lists are indexed from 0, we need to take iteration-2.
        experiment_id = experiment_ids[iteration - 2]
        _download_characterized_csv(experiment_id=experiment_id, output_path=expected_location, connection=connection)


def _download_characterized_data(
    experiment_name: str, iteration: int, connection: pybckg.WetLabAzureConnection
) -> None:  # pragma: no cover
    """This is a higher-level wrapper around ``_download_initial_batch`` and ``_download_later_batch``.

    Args:
        experiment_name: experiment track name
        iteration: iteration number, should be at least 1
        connection: pyBCKG connection
    """
    # Create the directory for the characterization data.
    structure.data_characterization_dir(experiment_name).mkdir(parents=True, exist_ok=True)

    if iteration == 1:
        _download_initial_batch(experiment_name, connection=connection)
    elif iteration > 1:
        _download_later_batch(experiment_name, iteration=iteration, connection=connection)
    else:
        raise ValueError(f"Iteration {iteration} must be at least 1.")

    # Check if we have exactly as many files as we need at this iteration.
    n_files = len(list(structure.data_characterization_dir(experiment_name).glob("*.csv")))
    # TODO: Is this check actually useful?
    assert n_files == iteration, "The number of files in the characterization directory is wrong."


def _do_combinatorics(
    experiment_name: str, iteration: int, config: Optional[combinatorics.CombinatoricsConfig] = None
) -> None:  # pragma: no cover
    """Runs combinatorics on characterization data to get ABEX-ready data.

    Args:
        experiment_name: experimental track name
        iteration: iteration number, starting from 1
        config: combinatorics config

    See:
        ``combinatorics.from_expetiment_to_optimization``
    """
    # Initialize the directories
    structure.data_abex_input_dir(experiment_name).mkdir(parents=True, exist_ok=True)

    # Note that we process the file that was suggested at the last iteration, so it has lower iteration number
    expected_path: pathlib.Path = structure.data_abex_input_path(experiment_name, iteration - 1)

    # Read the combinatorics configuration file
    config2: combinatorics.CombinatoricsConfig = config or settings.load_settings(
        structure.COMBINATORICS_CONFIG_PATH, class_factory=combinatorics.CombinatoricsConfig
    )

    if not expected_path.exists():
        # Read the characterized data. Again, remember that we use the last experiment.
        data_pre_combinatorics = pd.read_csv(structure.data_characterization_path(experiment_name, iteration - 1))

        data_post_combinatorics: pd.DataFrame = combinatorics.from_experiment_to_optimization(
            data_pre_combinatorics, config2  # type: ignore # auto
        )

        # Save the data to the disk
        data_post_combinatorics.to_csv(expected_path, index=False)


def prepare_data_for_abex(experiment_name: str, iteration: int) -> None:  # pragma: no cover
    """Downloads the data, characterizes them and runs combinatorics.

    Args:
        experiment_name: experimental track name
        iteration: iteration number, starting at 1
    """

    # TODO: Export the possibility of giving a custom connection
    connection = pybckg.create_connection("BCKG_PRODUCTION_CONNECTION_STRING")

    # First, download a new file with characterization results.
    _download_characterized_data(experiment_name=experiment_name, iteration=iteration, connection=connection)

    # Now, do the combinatorics and save it, so it's ready to be used by ABEX.
    _do_combinatorics(experiment_name, iteration)
