# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Each optimization strategy is a whole pipeline: it loads the data, builds models, and generates new batches of
experiments. Steps that are used by all of them are collected in this submodule.
"""
import abc
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from abex.dataset import Dataset
from abex import transforms
from abex.settings import OptimizerConfig


class OptimizerBase(abc.ABC):
    def __init__(self, config: OptimizerConfig):
        """
        Args:
            config: configuration, which should not be modified once assigned here.
        """
        self.config = config.copy(deep=True)

    # Each subclass of OptimizerBase should have a distinct non-None value for this attribute, which should
    # be the value of one of the members of OptimizationStrategy.
    strategy_name: Optional[str] = None

    @abc.abstractmethod
    def run(self) -> Tuple[Optional[Path], Optional[pd.DataFrame]]:
        """
        Must be implemented in each subclass.
        Returns:
            path to the file with suggested samples. If batch size is 0, None is returned instead
            data frame with suggested samples. If batch size is 0, None is returned instead
        """
        pass  # pragma: no cover

    def run_with_adjusted_config(self, extension: str, file_dict: Dict[str, str]):
        sub_config = self.config.with_adjustments(extension, file_dict)
        return self.__class__(sub_config).run()

    def construct_dataset(self) -> Dataset:
        """Constructs data step, building the data set from the config.

        Returns:
            dataset
        """
        logging.info("-----------------")
        logging.info("Preparing data")

        logging.info("- Loading DataFrame and converting to a Dataset")
        dataset = Dataset.from_data_settings(self.config.data)

        return dataset

    @staticmethod
    def suggestions_to_original_space(dataset: Dataset, new_samples: np.ndarray) -> pd.DataFrame:
        """Applies postprocessing to suggestions of new samples (i.e. transforms the points backwards to the original
        space and clips them to the bounds).
        Returns:
            data frame with samples in the original space
        """
        # Array to data frame
        expt = pd.DataFrame(np.atleast_2d(new_samples), columns=dataset.transformed_input_names)
        # Move categorical inputs to the original space
        # TODO: Test if this works as we expect.
        expt = dataset.onehot_transform.backward(expt) if dataset.onehot_transform else expt

        # Move continuous inputs into the original space
        if not isinstance(dataset.preprocessing_transform, transforms.InvertibleTransform):
            raise AttributeError("The preprocessing must be invertible to generate a batch.")  # pragma: no cover
        expt_original_space: pd.DataFrame = dataset.preprocessing_transform.backward(expt)

        # Clip the experiments to the bounds
        _clip_to_parameter_bounds(expt_original_space, dataset.pretransform_cont_param_bounds, tol=1e-3)

        return expt_original_space

    @classmethod
    def from_strategy(cls, config: OptimizerConfig, strategy: Optional[str] = None) -> "OptimizerBase":
        """Returns the run function of an instance of the subclass with the matching strategy name"""

        if strategy is None:
            strategy = config.optimization_strategy
        for sub in cls.__subclasses__():
            assert sub.strategy_name is not None
            if sub.strategy_name.lower() == strategy.lower():
                return sub(config)
        raise ValueError(f"Optimization strategy {strategy} not recognized.")  # pragma: no cover


def _clip_to_parameter_bounds(
    df: pd.DataFrame, continuous_param_bounds: Dict[str, Tuple[float, float]], tol: float = 1e-3
) -> None:
    """Clips the continuous inputs in bounds to parameter bounds, as long as they are outside of the parameter bounds
    only by the specified tolerance (tol * parameter_bounds_range). This is needed as, for inputs at the boundary,
    the pre-processing can introduce slight numerical errors.

    Performs the clipping in place.

    Args:
        df: The DataFrame with (a batch of) inputs to perform clipping on
        continuous_param_bounds: Dictionary from continuous column
        tol (optional): The percentage of the input parameter range that the parameter is allowed to deviate
            outside of said range. Defaults to 1e-3.

    Raises:
        ValueError: If the input parameters go outside bound given by an amount larger than specified by
            `tol` parameter.
    """
    for input_column, (lower_bound, upper_bound) in continuous_param_bounds.items():
        param_range = upper_bound - lower_bound

        # Calculate the range of the values of this input
        input_min: float = df[input_column].min()  # type: ignore # auto
        input_max: float = df[input_column].max()  # type: ignore # auto

        # If any values are out of bounds by more than tol * param_range, raise an exception
        if input_min < lower_bound - tol * param_range:
            raise ValueError(  # pragma: no cover
                f"Value {input_min} in column {input_column} in batch is outside of the "
                f"parameter's bounds: {(lower_bound, upper_bound)}. This is outside the specified "
                f"tolerance of {tol}, so the value can't be clipped to range."
            )
        if input_max > upper_bound + tol * param_range:
            raise ValueError(  # pragma: no cover
                f"Value {input_max} in column {input_column} in batch is outside of the "
                f"parameter's bounds: {(lower_bound, upper_bound)}. This is outside the specified "
                f"tolerance of {tol}, so the value can't be clipped to range."
            )
        # Otherwise, clip to range
        # Log a warning if the input is being clipped:
        if (input_min < lower_bound) or (input_max > upper_bound):
            logging.debug(  # pragma: no cover
                f"Clipped some values in column {input_column} in batch. The original range was "
                f"{(input_min, input_max)}, and the points are being clipped to: "
                f"{(lower_bound, upper_bound)}"
            )
        # Perform clipping in place
        df[input_column].clip(lower=lower_bound, upper=upper_bound, inplace=True)
