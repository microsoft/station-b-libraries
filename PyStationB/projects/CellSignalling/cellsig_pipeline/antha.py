# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Submodule for interactions with Synthace's Antha.

For now we can convert experiments to their DoE XLSX format. We consider integration with their *bundle* protocol, which
also contains information about the plates used and substances. (Currently they need to be specified in Antha's GUI).

Exports:
    generate_antha_dataframe, converts an experiment data frame to Antha-compatible data frame
"""
import cellsig_pipeline.combinatorics as comb
import cellsig_pipeline.settings as settings
import pandas as pd
import pydantic


class AnthaConfig(pydantic.BaseModel):
    """Antha DoE conventions."""

    concentration_column_format: str = "{} ({})"
    tag_column_format: str = "{}"


def generate_antha_dataframe(
    df: pd.DataFrame, config: comb.CombinatoricsConfig, antha_naming: AnthaConfig
) -> pd.DataFrame:  # pragma: no cover
    """Maps a data frame in experimental space, which uses `true_name`s, to the special input format to Antha (in
    particular parametrized by `antha_name`s.

    Args:
        df: pandas data frame with rows corresponding to experiments
            Ara    ATC    C6    C12
            7.6    8.7    0     5
            8.4    10.3   5.6   0
        config: configuration object specifying the inputs and units
        antha_naming: configuration object specifying Antha conventions for column names

    Returns:
        a data frame specifying an Antha experiment (rows correspond to wells), with column names inferred from input
        antha_name
    """
    # We construct an empty data frame. We will add columns manually in a for loop.
    output = pd.DataFrame()

    def add_reagent(reagent: comb.Input) -> None:
        # Names of columns added
        reagent_column: str = antha_naming.concentration_column_format.format(reagent.name, reagent.units)
        output[reagent_column] = df[reagent.name].values

    # Add signalling inputs
    add_reagent(config.signals.get_signal1)
    add_reagent(config.signals.get_signal2)

    # Add independent inputs
    for independent_input in config.inputs:
        add_reagent(independent_input)

    # Add pairing information
    def rename_pairing(v: str) -> str:
        tag = settings.Tag(name=config.pairing, value=v)
        return tag.encode()

    tag_column_name: str = settings.Tag(name=config.pairing, level=settings.TagLevel.SAMPLE).column_name
    antha_tag_column_name: str = antha_naming.tag_column_format.format(tag_column_name)
    output[antha_tag_column_name] = df[config.pairing].map(rename_pairing).values

    return output
