# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""High-level steps forming the pipeline. Note that this works on the preprocessed data, which is done in
`cellsig_pipeline.data` and `cellsig_pipeline.pybckg`.

Exports:
    run_abex, running the whole ABEX
    abex_to_doe, converts ABEX batch to Antha-specific format
"""
import logging
from typing import Optional

import abex.settings as abset
import pandas as pd
from abex.optimizers.optimizer_base import OptimizerBase
from cellsig_pipeline import antha, combinatorics, settings, structure


def run_abex(experiment_name: str, iteration: int) -> pd.DataFrame:  # pragma: no cover
    """Runs ABEX on experimental track `experiment_name` at iteration `iteration`.

    Returns:
        a data frame with ABEX batch suggestions
    """
    # TODO: It seems that `simple_load_config` is deprecated. However, `simple_load_config` seems to have
    #  side effect. I'm unsure whether we should use it.
    config: abset.OptimizerConfig = abset.simple_load_config(structure.config_abex_path(experiment_name))

    n_files = len(list(structure.data_abex_input_dir(experiment_name).glob("*.csv")))
    assert n_files == iteration, f"Number of downloaded files must match the iteration: {n_files} != {iteration}"

    # Set the results directory
    config.results_dir = structure.data_abex_results_iteration_dir(experiment_name, iteration)
    exist_ok = len(list(config.results_dir.glob("*"))) == 0
    config.results_dir.mkdir(parents=True, exist_ok=exist_ok)

    # Generate a new batch
    optimizer = OptimizerBase.from_strategy(config, config.optimization_strategy)

    logging.info("steps.py:45: calling  optimizer.run")
    _, batch = optimizer.run()
    logging.info("steps.py:47: finished optimizer.run")

    assert batch is not None, "Batch size must be greater than 0."
    return batch


def _batch_to_bare_doe(batch: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
    """Runs the combinatorics, without adding any tags. Changes the column names to the Antha-specific conventions."""
    # Run the combinatorics
    config = settings.load_settings(structure.COMBINATORICS_CONFIG_PATH, combinatorics.CombinatoricsConfig)
    doe = combinatorics.from_optimization_to_experiment(batch, config)

    # Use the Antha-specific format
    antha_config = settings.load_settings(structure.ANTHA_CONFIG_PATH, antha.AnthaConfig)
    doe = antha.generate_antha_dataframe(doe, config=config, antha_naming=antha_config)

    return doe


def _add_tags(
    df: pd.DataFrame, experiment_name: str, antha_naming: Optional[antha.AnthaConfig] = None
) -> pd.DataFrame:  # pragma: no cover
    """Adds experiment-track-specific tags to the DoE."""
    antha_naming2: antha.AnthaConfig = antha_naming or settings.load_settings(
        structure.ANTHA_CONFIG_PATH, antha.AnthaConfig
    )

    config: settings.DataSettings = settings.load_settings(
        structure.config_data_path(experiment_name), settings.DataSettings
    )
    doe = df.copy()

    for tag in config.tags:
        antha_doe_column_name = antha_naming2.tag_column_format.format(tag.column_name)
        doe[antha_doe_column_name] = tag.encode()

    return doe


def abex_to_doe(batch: pd.DataFrame, experiment_name: str) -> pd.DataFrame:  # pragma: no cover
    """Converts ABEX-generated batch into a DoE.

    Args:
        batch: ABEX-suggested batch
        experiment_name: experiment name (so the tags can be retrieved)

    TODO:
        Is this a pure function?
    """
    return _add_tags(_batch_to_bare_doe(batch), experiment_name)
