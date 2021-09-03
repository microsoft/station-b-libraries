# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------

"""
This module contains the DataSettings class (used for the "data" field of OptimizerConfig) and the helper
classes it uses.
"""
import collections
import enum
import logging
import os
from enum import Enum
from pathlib import Path
from typing import OrderedDict, List, Dict, Optional, Union, Literal, Tuple, Set

import numpy as np
import pandas as pd
import pydantic
from abex.constants import FILE

FloatIntStr = Union[float, int, str]


# Subclassing to str to make it json serializable
@enum.unique
class NormalisationType(str, Enum):
    """Enumeration of supported normalisation types"""

    MAX_ONLY = "MaxOnly"  # Divide by max
    FULL = "Full"  # Map to [0,1]
    NONE = "None"


PLOT_SCALE = Literal["linear", "symlog", "log"]


DATA = os.path.join("tests", "data", "Data")


class ParameterConfig(pydantic.BaseModel):
    """A class to collect settings for a generic continuous parameter (an input or output).

    Attributes:
        unit: The unit name to use on labels in plots
        log_transform: Whether to log this variable as a pre-processing step
        offset: If log-transform is True, this offset will be added to variable before applying the log.
            This is useful to possibly shift it into the range of the log-transform if some values are <= 0.
        rescale_column: If a column name given, values in this input will be divided by values of the rescale column
        final_offset:  A final shift (offset) that is added after possible log-transform, but before normalisation
        normalise: Whether and how to normalise this variable. The field can either be 'NONE', 'MAX_ONLY' or 'FULL'
    """

    unit: Optional[str] = None
    log_transform: bool = False
    offset: float = 0.0
    rescale_column: Optional[str] = None
    final_offset: float = 0.0
    normalise: NormalisationType = NormalisationType.MAX_ONLY  # ND Strongly recommend not turning this off

    @property
    def plotting_scale(self) -> PLOT_SCALE:  # pragma: no cover
        """Return the most appropriate plotting scale for this parameter ("log", "symlog" or "linear").

        Returns:
            str: Either "log", "symlog" or "linear" depending on the values of self.offset and self.log_transform.
        """
        if self.log_transform:
            if self.offset <= 0:
                return "log"
            else:
                # If an offset is applied, some of the pre-offset values might be non-positive -> log-scale unsuitable
                return "symlog"
        else:
            return "linear"


class InputParameterConfig(ParameterConfig):
    """A class to collect settings for a continuous input parameter.

    Attributes:
        lower_bound, upper_bound: The constraints on the inputs to be passed to the Bayesian Optimization procedure.
            Defined in the original input-space (these will be passed through pre-processing transforms). If not given
            the lower_bound and upper_bound will be inferred as the minimum and maximum of the values in the data
            provided.
        default_condition: NaN values for this input will be replaced with this value
        drop_if_nan: If set to True, if the value of the input is NaN, the entire row of data will be dropped
        drop_if_outside_bounds: Whether to drop a row of data if this input is outside the specified lower/upper bounds
    """

    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    default_condition: Optional[float] = None
    drop_if_nan: bool = False
    drop_if_outside_bounds: bool = False


def onehot(n: int, i: int) -> np.ndarray:  # pragma: no cover
    """Returns a one-hot vector of shape (n,) with 1 at the `i`th place."""
    v = np.zeros(n)
    v[i] = 1.0
    return v


def split_name_and_unit(s: str) -> Tuple[str, Optional[str]]:
    """For names of the form "ColumnName (unit)", separate out the name from the unit (by assuming the unit is within
    round brackets) and return a tuple ("ColumnName", "unit"). If no unit is is given (there are no round brackets),
    return ("ColumnName", None).

    Args:
        s: A string with a name and a possible unit specified in round brackets, i.e.:

    Returns:
        Tuple[str, Optional[str]]: A tuple with the first element being the name of the column extracted from the
            string, and the second being the string representing the unit extracted from the round brackets, or None
            if there are no brackets. I.e. the return values could be:

    Example:
        .. code-block ::

            >>> split_name_and_unit("ColumnName (unit)")
            ("ColumnName", "unit")
            >>> split_name_and_unit("ColumnName")
            ("ColumnName", None)
    """
    # If a unit is given:
    if "(" in s:  # pragma: no cover
        if s[-1] != ")":
            # The last symbol is expected to be ')'
            raise ValueError(
                f"If a unit is specified in the column name (is given in round brackets), the last element "
                f'must be the closing bracket for the unit: ")". The offending column name is:\n{s}'
            )
        name, unit = s.split("(", 1)
        # Remove the trailing closing bracket
        unit = unit[:-1]
        # Remove right whitespace after name
        name = name.rstrip()
        return name, unit
    # If no unit is given:
    else:
        return s, None


def is_upward_search_limit(path: Path) -> bool:  # pragma: no cover
    """
    Args:
        path:
            any Path
    Returns:
        True if any of several conditions are True (see the code)
    """
    path = path.absolute()
    if path.parent == path:
        return True  # top of file hierarchy
    if path.parent.name in ["libraries", "projects"] and (path / "environment.yml").is_file():
        return True  # subrepository directory
    if (path / "libraries").is_dir() and (path / "projects").is_dir():
        return True  # whole repository
    if path == Path.cwd():
        return True  # current directory
    home = os.getenv("HOME")
    if home is not None and path == Path(home):
        return True  # user's home directory
    return False


def has_required_files(folder: Path, data_adj: str, required: Set[str]) -> bool:
    """
    :param folder: a path to look for files in
    :param data_adj: an adjective to use in logging lines
    :param required: names of files to look for in folder
    :return: True if folder exists, and either (1) required is empty, or (2)
    folder contains one or more members of required. In the latter case, those members
    are removed from required.
    """
    if not folder.is_dir():
        logging.info(f"Not found: {data_adj} folder {folder}")
        return False
    files_found = [path.name for path in folder.glob("*.csv")]
    if required and not files_found:
        logging.info(f"Found, but empty: {data_adj} folder {folder}")
        return False
    found = required.intersection(files_found)
    logging.info(f"Found {len(found)} of {len(required)} files in: {data_adj} folder {folder}")
    if found or not required:
        required.difference_update(found)
        return True
    return False  # pragma: no cover


class DataSettings(pydantic.BaseModel):
    """
    A class to collect the configuration options for loading and pre-processing data

    Attributes:
        inputs: A dictionary of continuous inputs to use, mapping input names to an InputParameterConfig for that input.
            Specifies how to pre-process each input.
        categorical_inputs (List[str]): Specifies the categorical variables that will be used in the model
        devices (Dict[int, str]): Specifies the devices that are to be considered in the model. Datapoints with a GeneId
            not specified here will be filtered out from the dataset passed to the model
        config_file_location (Optional[Path]): Path from which the config file was read; should not be in config itself.
        files (List[str]): Specifies the CSV files to be loaded
        inputs (List[str]): Specifies the continuous-valued variables that will be used in the model
        categorical_filters (Dict[str, List[str]]): Filter for observations that belong to the contexts specified here
        output_column (str): The name of the output signal
        output_settings: ParameterConfig for the output. Describes which pre-processing to apply to the output.
        folder (Path): The folder where CSV data is loaded from
        default_conditions (Dict): Any missing values in columns given by this field will be filled in with the given
            value. Useful if filtering the data on columns that are not inputs with categorical_filters.
        zero (float): Map zeros of the output data to this value
    """

    inputs: OrderedDict[str, InputParameterConfig] = pydantic.Field(default_factory=collections.OrderedDict)
    categorical_inputs: List[str] = pydantic.Field(default_factory=list)
    devices: Dict[int, str] = pydantic.Field(default_factory=dict)
    config_file_location: Optional[Path] = None
    folder: Path = Path(DATA)  # We want to have the access to "folder" when validating "files", so it goes before that.
    files: Dict[str, str] = pydantic.Field(default_factory=dict)
    simulation_folder: Path = Path(DATA) / "fixed" / f"seed{0:06d}"
    categorical_filters: Dict[str, List[str]] = pydantic.Field(default_factory=dict)
    output_column: str = "Output"
    output_settings: ParameterConfig = pydantic.Field(default_factory=ParameterConfig)
    default_conditions: Dict[str, FloatIntStr] = pydantic.Field(default_factory=dict)
    zero: float = 1.0
    num_batches_left: Optional[int] = None

    @property
    def input_names(self) -> List[str]:
        """Return the names of the input variables"""
        return list(self.inputs.keys())

    @property
    def input_plotting_scales(self) -> Dict[str, PLOT_SCALE]:  # pragma: no cover
        """
        Returns a dictionary mapping from the input name to one of "log", "symlog", "linear" depending on
        what the most appriopriate plotting scale for that input is.
        """
        return {input_name: input_config.plotting_scale for input_name, input_config in self.inputs.items()}

    @pydantic.validator("files", pre=True, always=True)
    def files_not_provided(cls, v, *, values, **kwargs) -> Dict[str, str]:
        """If "files" is empty (e.g. not provided), we assume that all CSV files in the specified "folder" should be
        used."""
        if v:
            return v
        if values.get("folder", None) is not None:
            result = {
                file.stem: file.name for file in Path(values["folder"]).glob("*.csv") if file.stem != "_conditions.csv"
            }
            if result:
                return result
        return {}  # pragma: no cover
        # TODO: Consider raising an error:
        #  raise pydantic.ValidationError("You need to specify 'files' or 'folder'.")
        #  This however wouldn't be backwards-compatible, as TestDataSet seems to set neither "files" or "folder".

    @pydantic.validator("files")
    def duplicate_files(cls, v) -> Dict[str, str]:
        """If `files` contains non-unique files, raise an error."""
        if len(v.values()) != len(set(v.values())):
            raise pydantic.ValidationError("Some of the file names in config file are the same.")  # type: ignore
        return v

    def strip_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strips units from column names in the dataframe. Changes the names of the columns in place.

        Note:
            Changes dataframe in place.
        """
        for col in df.columns:  # type: ignore
            col_name, col_unit = split_name_and_unit(col)
            # If a unit was specified and the column is an input, check that it matches with the one given in config,
            # or update input parameter config units field if no unit was specified
            if col_name in self.inputs and col_unit is not None:  # pragma: no cover
                input_cfg = self.inputs[col_name]
                # If self.inputs[col_name] has a unit specified, assert it's the same as that in the DataFrame
                if input_cfg.unit:  # pragma: no cover
                    assert col_unit == input_cfg.unit
                # Otherwise, update the unit field in the config
                else:
                    input_cfg.unit = col_unit
            # Remove the units from the column name
            df.rename({col: col_name}, axis="columns", inplace=True)  # type: ignore # args allegedly mismatch
        return df

    def filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter-out irrelevant columns. This function is pure.

        Args:
            df: input data frame

        Returns:
            a view of `df` only with columns that represent inputs (continuous and categorical), file and output.
        """
        cols_to_keep: Set[str] = set(
            self.input_names
            + self.categorical_inputs
            + [FILE, self.output_column]
            + list(self.categorical_filters.keys())
        )

        return df[cols_to_keep]  # type: ignore

    def fill_in_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill-in missing values if a default value has been specified in config, alternatively, drop them if
        'drop_if_nan' has been set to True.

        Note:
            Changes dataframe in place.
        """
        # Get names of columns where missing values will be filled in, and the respective default conditions
        default_conditions: Dict[str, Union[str, float]] = {
            col: input_config.default_condition
            for col, input_config in self.inputs.items()
            if input_config.default_condition is not None
        }
        default_conditions = {
            **default_conditions,
            **self.default_conditions,
        }  # Add default values for non-input cols
        df.fillna(default_conditions, inplace=True)
        # Get names of columns for which, if the value is a nan, the row will be dropped
        drop_if_nan_cols = [col for col in self.input_names if self.inputs[col].drop_if_nan]
        if drop_if_nan_cols:  # pragma: no cover
            df.dropna(subset=drop_if_nan_cols, inplace=True)
        # Raise if all NaNs in input have not been removed or replaced
        if df[self.input_names].isnull().any().any():  # pragma: no cover
            offending_cols = df[self.input_names].columns[df[self.input_names].isnull().any()]
            raise ValueError(
                f"Some of the inputs in column {offending_cols} are NaN, and neither a default_value of drop_if_nan "
                f"have been set."
            )
        return df

    def drop_if_outside_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows of data if the inputs are outside of bounds specified, depending on the specification in the config
        for those inputs (i.e. depending on whether input_config.drop_if_outside_bounds is set to True).
        """
        res_df = df
        for input_name, input_config in self.inputs.items():
            # Filter-out only based on those inputs for which drop_if_outside_bounds set to True
            if input_config.drop_if_outside_bounds:
                # Filter-out entries outside the input lower and upper bounds as specified in config
                if input_config.lower_bound is not None:
                    res_df = res_df[(res_df[input_name] >= input_config.lower_bound)]  # type: ignore
                if input_config.upper_bound is not None:
                    res_df = res_df[(res_df[input_name] <= input_config.upper_bound)]  # type: ignore
        result = res_df.reset_index()  # type: ignore
        assert isinstance(result, pd.DataFrame)
        return result

    def load_dataframe(self) -> pd.DataFrame:
        """Load CSV data into a Pandas DataFrame"""
        df = self.load_dataframe_from_csv()

        # Apply categorical filters
        for parameter, values in self.categorical_filters.items():  # pragma: no cover
            valid_categ_ids = np.in1d(df[parameter].values, values)  # type: ignore
            if np.any(~valid_categ_ids):
                logging.warning(
                    f"{np.sum(~valid_categ_ids)} examples removed based on categorical filter on {parameter}"
                )
            df = df[valid_categ_ids]  # type: ignore
        assert isinstance(df, pd.DataFrame)
        return df

    def locate_config_folders(self) -> List[Path]:
        """
        Args:
            self object from a config
        Returns:
            one or more absolute paths for an existing self.folder and/or self.simulation_folder.
            Tries to find such paths relative to the current directory, and then relative to where the config
            was read from, its parent folder, etc, stopping when is_upward_search_limit is satisfied.
            Paths are included in the result if they contain one or more of the files mentioned in self.files.values();
            or, in the case that self.files is empty, the first such folder traversed that exists is returned,
            even if it is empty.
        Raises:
            FileNotFoundError if folders containing all the required files are not found.
        """
        required = set(self.files.values())
        results = []
        for folder, data_adj in [(self.folder, "real"), (self.simulation_folder, "simulated")]:
            if has_required_files(folder, data_adj, required):
                results.append(folder.absolute())
                if not required:
                    return results
            if folder.is_absolute() or self.config_file_location is None:
                continue  # pragma: no cover
            candidate_start_path = self.config_file_location.absolute().parent
            while True:
                candidate = candidate_start_path / folder
                if has_required_files(candidate, data_adj, required):
                    results.append(candidate)
                    if not required:
                        return results
                if is_upward_search_limit(candidate_start_path):
                    break
                candidate_start_path = candidate_start_path.parent
        raise FileNotFoundError("Cannot find folder or simulation_folder with required files")

    def load_dataframe_from_csv(self) -> pd.DataFrame:
        """
        Loads each csv file specified in self.files as a DataFrame, and concatenates them together.
        """
        frames = []
        folders = self.locate_config_folders()
        if not self.files:
            raise FileNotFoundError(f"No files defined for {self.folder}")  # pragma: no cover
        for label, f in self.files.items():
            csv_files = [folder / f for folder in folders if (folder / f).exists()]
            if not csv_files:
                raise FileNotFoundError(f"Cannot load dataframe from: {f}")  # pragma: no cover
            frame: pd.DataFrame = pd.read_csv(csv_files[0])  # type: ignore # auto
            frame[FILE] = label
            frames.append(frame)
        df = pd.concat(frames, join="outer")

        df = self.strip_units(df)
        df = self.filter_columns(df)
        df = self.fill_in_missing(df)
        df = self.drop_if_outside_bounds(df)
        return df

    def get_bounds_from_config_and_data(
        self, df: Optional[pd.DataFrame] = None
    ) -> OrderedDict[str, Tuple[float, float]]:
        """
        Get the parameter bounds from the config, and where not available, try to infer them from the data DataFrame
        (as the minimum and the maximum value that each parameter takes) if the data DataFrame is given.
        """
        # Get parameter bounds in the original space
        bounds = collections.OrderedDict()
        for input_name, input_settings in self.inputs.items():
            if input_settings.lower_bound is None:
                if df is None:
                    raise ValueError(  # pragma: no cover
                        "The parameter bounds must either be specified in the config, "
                        "or a DataFrame must be given as argument."
                    )
                # If no bounds specified in config, infer from data
                bounds[input_name] = (df[input_name].min(), df[input_name].max())
            else:
                assert input_settings.rescale_column is None, "Rescaled columns can't be given bounds"
                assert input_settings.upper_bound is not None, "Both bounds must be specified"
                bounds[input_name] = (input_settings.lower_bound, input_settings.upper_bound)
        return bounds
