# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""A submodule for calculating the signal-to-crosstalk ratio.

The scenario is the following:

In theory, we have signal1 inducing ``output1`` and ``signal2`` inducing ``output2`` and some additional inputs,
controlling the strength of the response.

In practice, ``signal1`` induces ``output1`` and a bit of ``output2`` (similarly, ``signal2`` also induces a bit of
``output1``). Hence, we want to assess the performance of our device using the signal-to-crosstalk ratio:

.. math::

    output1(signal1, inputs) * output2(signal2, inputs) / ( output2(signal1, inputs) * output1(signal2, inputs) )

Each observation like that requires *two* real experiments -- one of them with ``signal1`` fixed to some value ``X``
and some fixed values of other inputs (but no ``signal2``).
The other one has ``signal2`` set to ``X`` and other inputs, but not ``signal1``.

We refer to this ``X`` value as ``signal_merged``.

Exports:
    CombinatoricsConfig, the configuration file specifying the inputs and outputs of the problem
    from_experiment_to_optimization, merges ``signal1`` and ``signal2`` into ``signal_merged``
    from_optimization_to_experiment, splits ``signal_merged`` into ``signal1`` and ``signal2``
"""
import uuid
import warnings
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydantic


class Input(pydantic.BaseModel):
    """Represents an input condition, alongside with a unit (to which the data will be normalized)."""

    name: str
    units: str


class SignalsConfig(pydantic.BaseModel):
    """
    Attributes:
        signal1 (str): the name of signal1, supposed to induce output1 only
        signal2 (str): the name of signal2, supposed to induce output2 only
        units (str): the units at which both signals are measured
        signal_merged (str): the name of a "glued" signal
    """

    signal1: str
    signal2: str
    units: str
    signal_merged: str

    @property
    def get_signal1(self) -> Input:  # pragma: no cover
        return Input(name=self.signal1, units=self.units)

    @property
    def get_signal2(self) -> Input:  # pragma: no cover
        return Input(name=self.signal2, units=self.units)


class OutputsConfig(pydantic.BaseModel):
    """
    Attributes:
        output1 (str): output to be induced by signal1
        output2 (str): output to be induced by signal2
    """

    output1: str
    output2: str


class CombinatoricsConfig(pydantic.BaseModel):
    """
    Attributes:
        signals (SignalsConfig): the signalling inputs
        inputs (list of Input): "independent" inputs, helping to convert signals to outputs
        outputs (OutputsConfig): the outputs, induced by signals
        pairing (str): name of the column which specifies which experiments correspond to one observation
        objective (str): name of the column where signal to crosstalk ratio is stored
    """

    signals: SignalsConfig
    inputs: List[Input]
    outputs: OutputsConfig

    pairing: str
    objective: str

    @property
    def all_conditions(self) -> List[Input]:  # pragma: no cover
        return self.inputs + [self.signals.get_signal1, self.signals.get_signal2]


def _check_columns_equal(
    df1: pd.DataFrame, df2: pd.DataFrame, column: str, column2: Optional[str] = None
) -> None:  # pragma: no cover
    """Check if values of ``column`` in ``df1`` match the values of ``column2`` in ``df2``.

    If ``column2`` is not provided, it's assumed to be equal to ``column``.
    """
    column2_s: str = column2 or column
    if not (df1[column].values == df2[column2_s].values).all():  # type: ignore # auto
        # TODO: Consider raising an error.
        warnings.warn(f"Mismatch for columns {column} in first data frame and {column2_s} in second data frame.")


def _check_inputs_equal(df1: pd.DataFrame, df2: pd.DataFrame, inputs: Iterable[Input]) -> None:  # pragma: no cover
    """Checks if the input columns agree in both data frames."""
    for input in inputs:
        _check_columns_equal(df1, df2, input.name)


def _check_complementary(df1: pd.DataFrame, df2: pd.DataFrame, config: CombinatoricsConfig) -> None:  # pragma: no cover
    """Check if `df1` and `df2` are complementary, i.e. `df1` contains the entries associated to `signal1` and
    `df2` contains the entries associated to `signal2`."""
    _check_inputs_equal(df1, df2, config.inputs)
    _check_columns_equal(df1, df2, config.signals.signal1, config.signals.signal2)


def _split_by_tags(
    df: pd.DataFrame, config: CombinatoricsConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:  # pragma: no cover
    """Splits a data frame into two -- the first one corresponding to ``signal1 > 0``, and the other corresponding to
    ``signal2 > 0``.

    Note:
        NaN values are replaced with 0.
    """
    # Sort by tagging
    df = df.fillna(value=0)  # type: ignore # auto
    df = df.sort_values([config.pairing, config.signals.signal1]).reset_index(drop=True)

    signal1_on = df[df[config.signals.signal1] > 0].reset_index(drop=True)  # type: ignore
    signal2_on = df[df[config.signals.signal2] > 0].reset_index(drop=True)  # type: ignore

    return signal1_on, signal2_on  # type: ignore # auto


def from_experiment_to_optimization(df: pd.DataFrame, config: CombinatoricsConfig) -> pd.DataFrame:  # pragma: no cover
    """Merges the ``signal1`` and ``signal2`` into ``signal_merged``.

    Args:
        df, a data frame with observations
        config, a combinatorics config, specifying the naming conventions (see below)

    Returns:
        data frame with columns with names of ``config.inputs``, ``config.objective``, ``config.signal_merged``
    """
    signal1_on, signal2_on = _split_by_tags(df, config)

    # Run the checks
    _check_complementary(signal1_on, signal2_on, config)

    def term(dataframe: pd.DataFrame, output: str) -> np.ndarray:
        return dataframe[output].values  # type: ignore # auto

    # Signal to crosstalk ratio: output1(signal1) * output2(signal2) divided by output2(signal1) * output1(signal2)
    numerator: np.ndarray = term(signal1_on, config.outputs.output1) * term(signal2_on, config.outputs.output2)
    denominator: np.ndarray = term(signal1_on, config.outputs.output2) * term(signal2_on, config.outputs.output1)
    objective: np.ndarray = numerator / denominator

    # Save independent inputs
    independent_inputs = {inp.name: signal1_on[inp.name].values for inp in config.inputs}

    # Save the signal value and the objective
    rest = {
        config.signals.signal_merged: signal1_on[config.signals.signal1].values,
        config.objective: objective,
    }

    merged = {**independent_inputs, **rest}

    return pd.DataFrame(merged)


def from_optimization_to_experiment(df: pd.DataFrame, config: CombinatoricsConfig) -> pd.DataFrame:  # pragma: no cover
    """Splits a batch of samples generated by ABEX (with ``signal_merged`` and independent inputs) into ``signal1``,
    ``signal2`` (and independent inputs).

    Args:
        df, a data frame with observations
        config, a combinatorics config, specifying the naming conventions (see below)

    Returns:
        data frame with columns with names of ``config.inputs``, ``config.objective``, ``config.signal1``
            and ``config.signal2``
    """
    output_list: List[dict] = []  # list of samples, converted to data frame at the end

    for entry_tuple in df.itertuples(index=False):
        entry: dict = entry_tuple._asdict()  # type: ignore # auto
        # We need to split each Bayesian Optimization entry into two experimental samples.
        # They have the same values of independent inputs and the same pairing identifier
        sample1 = {ind_input.name: entry[ind_input.name] for ind_input in config.inputs}
        sample1[config.pairing] = str(uuid.uuid4())
        sample2 = sample1.copy()

        # Now we should add the information about signal1 and signal2
        signal_on_value: float = entry[config.signals.signal_merged]

        sample1.update(
            {
                config.signals.signal1: signal_on_value,
                config.signals.signal2: 0.0,
            }
        )

        sample2.update(
            {
                config.signals.signal1: 0.0,
                config.signals.signal2: signal_on_value,
            }
        )

        # Append the samples to the list representing data frame.
        output_list.extend([sample1, sample2])

    # Convert list of samples to a data frame.
    output = pd.DataFrame(output_list)
    return output
