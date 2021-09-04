# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This module provides high-level wrappers over pyBCKG. (And characterization, as characterization is temporarily
a part of pyBCKG).

In particular, pyBCKG should *not* be imported anywhere outside this module.

Exports:
    WetLabAzureConnection, the main class, extending the usual AzureConnection by useful features
    create_connection, a nice factory of WetLabAzureConnection
    experiment_to_dataframe, returns a data frame with characterized experiment
"""
from typing import Dict, Iterable, Optional, Set, Tuple, Union

import cellsig_pipeline.characterization as characterization
import cellsig_pipeline.combinatorics as combi
import cellsig_pipeline.pybckg.units as units
import cellsig_pipeline.settings as settings
import cellsig_pipeline.structure as structure
import pandas as pd
import pyBCKG.domain as domain
from cellsig_pipeline.pybckg.connection import WetLabAzureConnection


def _signal_to_common_name(signal: domain.Signal) -> str:  # pragma: no cover
    """Retrieve a human-readable name from filter settings.

    Todo:
        Consider creating a configuration file in CellSignalling where user can define human-readable names
        for different filter settings. (And a parser on PyBCKG side).
    """
    stationb_map: domain.SIGNAL_MAP = {
        (550.0, 10.0, 610.0, 20.0): "mRFP1",
        (430.0, 10.0, 480.0, 10.0): "ECFP",
        (500.0, 10.0, 530.0, None): "EYFP",
        (485.0, 12.0, 520.0, None): "GFP",
        (485.0, 12.0, 530.0, None): "GFP530",
        600.0: "OD",
        700.0: "OD700",
    }
    return signal.to_label(stationb_map)


def dataframe_from_sample(
    sample: domain.Sample, experiment: domain.Experiment, connection: WetLabAzureConnection
) -> pd.DataFrame:  # pragma: no cover
    """Retrieves a data frame associated to the given sample.
    The columns are renamed from GUIDs to human-readable names.

    Args:
        sample, the sample of which timeseries we try to retrieve
        experiment, experiment in which `sample` exists (used to retrieve the naming conventions)
        connection, a connection object, used to retrieve the data

    Returns:
        data frame with columns renamed
    """
    # Retrieve the human-readable names
    signal_map = {signal.guid: _signal_to_common_name(signal) for signal in experiment.signals}

    # Get the timeseries associated to a sample
    timeseries: pd.DataFrame = connection.get_timeseries_from_sample(sample=sample)
    return timeseries.rename(columns=signal_map)


def _characterize(
    df: pd.DataFrame, config: Optional[characterization.CharacterizationConfig] = None
) -> Dict[str, float]:  # pragma: no cover
    """An auxilary high-level function, characterizing the data frame using the integration-based characterization,
    as specified in the config.

    Returns:
        see `pipeline.characterization.integral_characterization`
    """
    # If `config` is not provided, use the global one.
    default_path = structure.CHARACTERIZATION_CONFIG_PATH
    config = config or settings.load_settings(default_path, class_factory=characterization.CharacterizationConfig)

    time_range: Tuple[float, float] = (config.integration.t_min, config.integration.t_max)

    return characterization.integral_characterization(
        df, signals=config.signals, time_column="time", time_range=time_range
    )


def _get_conditions(sample: domain.Sample, conditions: Iterable[combi.Input]) -> Dict[str, float]:  # pragma: no cover
    """Retrieves condition concentrations for all `conditions` from the values stored in `sample`.

    Note:
        If a particular condition is not present, it's value is set to 0.
        The units are converted to these defined in `conditions`.
    """
    # The default value is 0 for all required conditions.
    result = {condition.name: 0.0 for condition in conditions}

    # For each condition, we'll try to update this value by looking at the BCKG data.
    for condition in conditions:
        # Look up for the condition in `sample`
        for candidate in sample.conditions:
            if candidate.reagent.name == condition.name:
                # Change units
                new_concentration: domain.Concentration = units.change_units(
                    concentration=candidate.concentration, desired_unit=condition.units
                )
                result[condition.name] = new_concentration.value

    return result


def _get_pairing(sample: domain.Sample, connection: WetLabAzureConnection) -> Dict[str, str]:  # pragma: no cover
    """The Observation ID (to be renamed into Pairing ID) of a given sample.

    Returns:
        a dictionary with exactly one key: structure.OBSERVATION_ID

    Raises:
        ValueError, if the Observation ID tag is not associated to the sample

    Todo:
        This uses the global constant `structure.OBSERVATION_ID`. It should be replaced with `config.pairing`.
    """
    tags: Set[str] = connection.list_tags_associated_to_entity(sample.guid)

    for tag in tags:
        if f"{structure.OBSERVATION_ID}:" in tag:
            return {structure.OBSERVATION_ID: tag.split(":")[1]}

    raise ValueError(f"Sample {sample} has a set of tags {tags} which doesn't have a pairing ID in the right format.")


def _get_sample_information(sample: domain.Sample) -> Dict[str, str]:  # pragma: no cover
    """A dictionary retrieving the sample information, to be added into the final data frame with characterization
    results.

    Returns:
        dict, right now a dictionary with "SampleID" key
    """
    # TODO: This ideally should be replaced with a constant, rather than a hard-coded string.
    return {"SampleID": sample.guid}


def _process_sample(
    sample: domain.Sample,
    experiment: domain.Experiment,
    connection: WetLabAzureConnection,
    config: combi.CombinatoricsConfig,
) -> Dict[str, Union[float, str]]:  # pragma: no cover
    """A high-level function, retrieving the data frame from the sample, conditions, pairing ID and other information
    (see `_get_sample_information`) and returning it as a single dictionary.
    """
    df = dataframe_from_sample(sample=sample, experiment=experiment, connection=connection)

    return {
        **_get_sample_information(sample),
        **_get_pairing(sample, connection=connection),
        **_get_conditions(sample, conditions=config.all_conditions),
        **_characterize(df),
    }


def experiment_to_dataframe(
    experiment_guid: str, connection: WetLabAzureConnection, config: combi.CombinatoricsConfig
) -> pd.DataFrame:  # pragma: no cover
    """Retrieves the time series data from BCKG and characterizes them. Returns a data frame with characterized data.

    It's a very high-level function. Note that it wraps over `_process_sample`.
    """
    # Retrieve the experiment
    experiment: domain.Experiment = connection.get_experiment_by_id(experiment_guid)  # type: ignore # auto

    # Build the data frame gathering observation for each sample
    df = pd.DataFrame(
        [
            _process_sample(sample, experiment=experiment, connection=connection, config=config)
            for sample in experiment.samples
        ]
    )

    return df
