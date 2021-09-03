# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""A submodule implementing "zooming in" (Biological) optimization strategy.

This optimization strategy has a single hyperparameter :math:`s`, called the *shrinking factor*.
It consists of of the following steps:

  1. The optimization space is a hypercuboid

    .. math::

        C = [a_1, b_1] \\times [a_2, b_2] \\times \\cdots \\times [a_n, b_n].

  2. Find the optimum :math:`x=(x_1, x_2, \\dots, x_n)` among the already collected samples.

  3. Construct a new hypercuboid :math:`D` centered at :math:`x`. If this is the :math:`N`th optimization step, the
     volume of :math:`D` is given by

     .. math::

        \\mathrm{vol}\\, D = s^N \\cdot \\mathrm{vol}\\, C

    Step :math:`N` is either provided in the configuration file or is estimated as ``n_samples/batch_size``.

  4. If :math:`D` is not a subset of :math:`C`, we translate it by a vector.

  5. To suggest a new batch we sample the hypercuboid :math:`D`. Many different sampling methods are available, see
     :ref:`abex.sample_designs` for this. For example, we can construct a grid, sample in a random way or use Latin
     or Sobol sampling.
"""
from pathlib import Path
from typing import List, Tuple

import abex.optimizers.optimizer_base as base
import numpy as np
import pandas as pd
from abex import space_designs as designs
from abex.dataset import Dataset
from abex.settings import OptimizationStrategy, ZoomOptSettings
from emukit.core import ContinuousParameter, ParameterSpace

Interval = Tuple[float, float]  # Endpoints of an interval
Hypercuboid = List[Interval]  # Optimization space is represented by a rectangular box


class ZoomOptimizer(base.OptimizerBase):

    strategy_name = OptimizationStrategy.ZOOM.value

    def run(self) -> Tuple[Path, pd.DataFrame]:
        """
        Optimizes function using "zooming in" strategy -- around observed maximum a new "shrunk" space is selected. We
        sample this space (e.g. using grid sampling or random sampling) to suggest new observations.

        Note:
            This method should not work well with very noisy functions or functions having a non-unique maximum. A more
            robust alternative (as Bayes optimization) should be preferred. On the other hand, this method is much
            faster to compute.

        Returns:
            path to the CSV with locations of new samples to be collected
            data frame with locations of new samples to be collected

        Raises:
            ValueError, if batch size is less than 1
        """

        # Construct the data set
        dataset: Dataset = self.construct_dataset()

        assert (
            self.config.zoomopt is not None
        ), "You need to set the 'zoomopt' field in the config to use Zoom optimizer."
        batch_transformed_space: np.ndarray = _suggest_samples(dataset=dataset, settings=self.config.zoomopt)

        # Transform the batch back to original space
        batch_original_space: pd.DataFrame = self.suggestions_to_original_space(
            dataset=dataset, new_samples=batch_transformed_space
        )

        # Save the batch to the disk and return it
        batch_original_space.to_csv(self.config.experiment_batch_path, index=False)
        # Save the inferred optimum
        optimum = evaluate_optimum(dataset)
        optimum.to_csv(self.config.results_dir / "optima.csv", index=False)
        return self.config.experiment_batch_path, batch_original_space


def evaluate_optimum(dataset: Dataset) -> pd.DataFrame:
    """
    Return the optimum as inferred by the Zoom Opt. algorithm. The inferred optimum is taken as the location
    of the observed sample with highest observed objective.

    Args:
        dataset (dataset.Dataset): Dataset with the data observed so-far.

    Returns:
        pd.DataFrame: A DataFrame with a single row: the inputs at the inferred optimum
    """
    # Get the index of data point with highest observed objective
    optimum_idx = dataset.pretransform_df[dataset.pretransform_output_name].argmax()
    # Get the inputs of the data point with highest observed objective
    optimum_loc = dataset.pretransform_df[dataset.pretransform_input_names].iloc[[optimum_idx]]
    return optimum_loc


def _suggest_samples(dataset: Dataset, settings: ZoomOptSettings) -> np.ndarray:
    """Suggests a new batch of samples.

    Currently this method doesn't allow categorical inputs.

    Returns:
        a batch of suggestions. Shape (batch_size, n_inputs).

    Raises:
        ValueError, if batch size is less than 1
        NotImplementedError, if any categorical inputs are present
    """

    if settings.batch < 1:
        raise ValueError(f"Use batch size at least 1. (Was {settings.batch}).")  # pragma: no cover

    continuous_dict, categorical_dict = dataset.parameter_space

    # If any categorical variable is present, we raise an exception. In theory they should be represented by one-hot
    # encodings, but I'm not sure how to retrieve the bounds of this space and do optimization within it (the
    # best way is probably to optimize it in an unconstrained space and map it to one-hot vectors using softmax).
    # Moreover, in BayesOpt there is iteration over contexts.
    if categorical_dict:
        raise NotImplementedError("This method doesn't work with categorical inputs right now.")  # pragma: no cover

    # It seems that continuous_dict.values() contains pandas series instead of tuples, so we need to map over it
    # to retrieve the parameter space
    original_space: Hypercuboid = [(a, b) for a, b in continuous_dict.values()]

    # Find the location of the optimum. We will shrink the space around it
    optimum: np.ndarray = _get_optimum_location(dataset)

    # Estimate how many optimization iterations were performed.
    step_number: int = settings.n_step or _estimate_step_number(
        n_points=len(dataset.output_array), batch_size=settings.batch
    )

    # Convert to per-batch shrinking factor if a per-iteration shrinking factor supplied
    per_batch_shrinking_factor = (
        settings.shrinking_factor ** settings.batch if settings.shrink_per_iter else settings.shrinking_factor
    )

    # Calculate by what factor each dimension of the hypercube should be shrunk
    shrinking_factor_per_dim: float = _calculate_shrinking_factor(
        initial_shrinking_factor=per_batch_shrinking_factor, step_number=step_number, n_dim=len(original_space)
    )

    # Shrink the space
    new_space: Hypercuboid = [
        shrink_interval(
            shrinking_factor=shrinking_factor_per_dim, interval=interval, shrinking_anchor=optimum_coordinate
        )
        for interval, optimum_coordinate in zip(original_space, optimum)
    ]

    # The shrunk space may be out of the original bounds (e.g. if the maximum was close to the boundary).
    # Translate it.
    new_space = _move_to_original_bounds(new_space=new_space, original_space=original_space)

    # Sample the new space to get a batch of new suggestions.
    parameter_space = ParameterSpace([ContinuousParameter(f"x{i}", low, upp) for i, (low, upp) in enumerate(new_space)])

    return designs.suggest_samples(
        parameter_space=parameter_space, design_type=settings.design, point_count=settings.batch
    )


def _estimate_step_number(n_points: int, batch_size: int) -> int:
    """Estimates which step this is (or rather how many steps were collected previously, basing on the ratio
    of number of points collected and the batch size).

    Note that this method is provisional and may be replaced with a parameter in the config.

    Raises:
        ValueError if ``n_points`` or ``batch_size`` is less than 1
    """
    if min(n_points, batch_size) < 1:
        raise ValueError(
            f"Both n_points={n_points} and batch_size={batch_size} must be at least 1."
        )  # pragma: no cover

    return n_points // batch_size


def _calculate_shrinking_factor(initial_shrinking_factor: float, step_number: int, n_dim: int) -> float:
    """The length of each in interval bounding the parameter space needs to be multiplied by this number.

    Args:
        initial_shrinking_factor: in each step the total volume is shrunk by this amount
        step_number: optimization step -- if we collected only an initial batch, this step is 1
        n_dim: number of dimensions

    Example:
        Assume that ``initial_shrinking_factor=0.5`` and ``step_number=1``. This means that the total volume should
        be multiplied by :math:`1/2`. Hence, if there are :math:`N` dimensions (``n_dim``), the length of each
        bounding interval should be multiplied by :math:`1/2^{1/N}`.
        However, if ``step_number=3``, each dimension should be shrunk three times, i.e. we need to multiply it by
        :math:`1/2^{3/N}`.


    Returns:
        the shrinking factor for each dimension
    """
    assert 0 < initial_shrinking_factor < 1, (
        f"Shrinking factor must be between 0 and 1. " f"(Was {initial_shrinking_factor})."
    )
    assert step_number >= 1 and n_dim >= 1, (
        f"Step number and number of dimensions must be greater than 0. "
        f"(Where step_number={step_number}, n_dim={n_dim})."
    )

    return initial_shrinking_factor ** (step_number / n_dim)


def _get_optimum_location(dataset: Dataset) -> np.ndarray:
    """Returns the position (in the transformed space) of the maximum. Shape (n_inputs,)."""

    # Retrieve the observations
    X, Y = dataset.inputs_array, dataset.output_array

    # Return the location of the maximum
    best_index = int(np.argmax(Y))

    return X[best_index, :]


def shrink_interval(shrinking_factor: float, interval: Interval, shrinking_anchor: float) -> Interval:
    """Shrinks a one-dimensional interval around the ``shrinking_anchor``. The new interval
    is centered around the optimum.

    Note:
        the shrunk interval may not be contained in the initial one. (E.g. if the shrinking anchor is near the
        boundary).

    Args:
        shrinking_factor: by this amount the length interval is multiplied. Expected to be between 0 and 1
        interval: endpoints of the interval
        shrinking_anchor: point around which the interval will be shrunk

    Returns:
        endpoints of the shrunk interval
    """
    neighborhood = shrinking_factor * (interval[1] - interval[0])
    return shrinking_anchor - neighborhood / 2, shrinking_anchor + neighborhood / 2


def _validate_interval(interval: Interval) -> None:
    """Validates whether an interval is non-empty.

    Note:
        one-point interval :math:`[a, a]` is allowed

    Raises:
          ValueError: if the end of the interval is less than its origin
    """
    origin, end = interval

    if end < origin:
        raise ValueError(f"Interval [{origin}, {end}] is not a proper one.")  # pragma: no cover


def interval_length(interval: Interval) -> float:
    """Returns interval length."""
    _validate_interval(interval)
    return interval[1] - interval[0]


def shift_to_within_parameter_bounds(new_interval: Interval, old_interval: Interval) -> Interval:
    """Translates ``new_interval`` to ``old_interval``, without changing its volume.

    Raises:
        ValueError: if translation is not possible.
    """
    if interval_length(new_interval) > interval_length(old_interval):
        raise ValueError(  # pragma: no cover
            f"Translation is not possible. New interval {new_interval} is longer "
            f"than the original one {old_interval}."
        )

    new_min, new_max = new_interval
    old_min, old_max = old_interval

    if old_min <= new_min and new_max <= old_max:  # In this case we don't need to translate the interval
        return new_interval
    else:
        if new_min < old_min:  # Figure out the direction of the translation
            translation = old_min - new_min
        else:
            translation = old_max - new_max

        return new_min + translation, new_max + translation


def _move_to_original_bounds(new_space: Hypercuboid, original_space: Hypercuboid) -> Hypercuboid:
    """Translates ``new_space`` to be a subset of the ``original_space``, without affecting its volume."""
    moved_bounds: Hypercuboid = []

    for new_interval, old_interval in zip(new_space, original_space):
        moved_bounds.append(shift_to_within_parameter_bounds(new_interval=new_interval, old_interval=old_interval))

    return moved_bounds
