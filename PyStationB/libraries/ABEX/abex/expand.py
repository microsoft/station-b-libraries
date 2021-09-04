# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Config expansion
"""
import logging
import math
import random
import re
from functools import reduce
from operator import mul
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from scipy.stats import norm  # type: ignore # norm does exist!


class FixedSeed:
    """
    Context manager to generate random numbers in a predictable way, restoring the random state afterwards.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        random.setstate(self.state)


def expand_structure(
    struc: Any, max_resolutions: Optional[int] = None, random_seed: int = 0
) -> Generator[Any, None, None]:
    """
    Yields a sequence of distinct resolutions, in random order, of the provided structure. Structures should be
    the results of loading yaml files, so consists of lists, dictionaries and atomic values (strings and numbers).

    A resolution is defined as follows:
    * A choice list is a Python list of more than two elements, whose first element is a list starting with a string
    whose first character is "@" and is followed by one of a-z and zero or more occurrences of a-z or underscore.
    Note that in YAML, you write such a list in the form ["@x", foo, bar], i.e. the "@x" has to be in quotes because
    "@" is not a legal first character for a YAML value.
    * A resolution of a choice list is a resolution of one of its non-initial elements.
    * A resolution of any other list is a list of resolutions of its elements.
    * A resolution of a dict is another dictionary, with each value replaced by one of its resolutions.
    * The (single) resolution of any other object - anything that is not a list or a dict - is the object itself.
    (We could extend the code to treat other complex objects, but that's not needed for YAML structures).

    If two choice lists have the same initial element all the resolutions are constrained to have the corresponding
    elements selected. All such choice lists must be the same length.

    Examples:
        "foo" has a single resolution, "foo"
        ["@x", "a", 2] has two resolutions, "a" and 2
        {"p": ["@x", "a", 2], "q": ["@y", "b", 3]} has the same four resolutions: "@x" and "@y" are distinct so
            all 2x2=4 combinations are valid
        {"p": ["@x", "a", 2], "q": ["@x", "b", 3]} has two resolutions, because corresponding elements are selected:
            {"p": "a", "q": "b"}, {"p": 2, "q": 3}

    You can think of each "@" string as defining a dimension, and each resolution as representing a point in the
    space defined by those dimensions. Two occurrences of the same "@" string refer to the same dimension.
    """
    for expansion, _ in expand_structure_with_resolutions(struc, max_resolutions, random_seed):
        yield expansion


def expand_structure_with_resolutions(
    struc: Any, max_resolutions: Optional[int] = None, random_seed: int = 0
) -> Generator[Tuple[Any, str], None, None]:
    """
    Alternative to expand_structure where what is yielded each time is not just a resolution but a pair
    (resolution, selection_dictionary) where selection_dictionary maps variable names (starting with "@") to
    the indices (from 1) at which values are selected.
    """
    dim_dict: Dict[str, int] = {}
    get_dimensions(struc, dim_dict)
    # Short cut, for the case where there are no choice lists.
    if not dim_dict:
        yield struc, ""
        return
    dim_list = sorted(dim_dict.items())
    dim_names = [name for name, _ in dim_list]
    dim_sizes = tuple(size for _, size in dim_list)
    # Number of distinct resolutions is the product of all the distinct dimensions, capped by max_resolutions if that
    # is not None. At present the product will be finite but could be very large. In future, we might draw values from
    # distributions with infinitely many values.
    n_resolutions = reduce(mul, dim_dict.values(), 1)
    if max_resolutions is not None and n_resolutions <= max_resolutions:
        max_resolutions = None  # pragma: no cover
    for index, res in enumerate(generate_resolutions(dim_sizes, random_seed, max_resolutions), 1):
        logging.info(f"Running configuration version {index}")
        # String of "fooNNN" items from dimension names "foo" to index NNN in choice list - from 1 to
        # number of choices, because the zeroth element of a choice list is the "@" symbol with the name.
        # There are no separator characters like "," or "=" because the names consist of a-z_ and the
        # indices are digits, so the sequence of characters can be interpreted uniquely and is suitable for
        # use in filenames. Thus selecting "z" in ["@foo", "x", "y", "z"] would give a spec of "foo3".
        res_dict = dict(zip(dim_names, res))
        res_spec = "".join(f"{name}{choice_index}" for name, choice_index in sorted(res_dict.items()))
        yield apply_selection(res_dict, struc), res_spec


def generate_resolutions(
    dim_sizes: Tuple[int, ...], random_seed: int = 0, max_resolutions: Optional[int] = None
) -> Generator[Tuple[int, ...], None, None]:
    if max_resolutions is None or math.prod(dim_sizes) <= max_resolutions:
        # Generate resolutions exhaustively and in canonical (sorted) order.
        next_res: Optional[List[int]] = [1 for _ in dim_sizes]
        while next_res is not None:
            yield tuple(next_res)
            next_res = increment_resolution(next_res, dim_sizes)
    else:
        # Generate max_resolutions randomly chosen (but distinct, and reproducible) resolutions.
        resolutions: Set[Tuple[int, ...]] = set()
        with FixedSeed(random_seed):
            while len(resolutions) < max_resolutions:
                # Generate a random resolution from dim_sizes
                random_res = tuple(random.randint(1, size) for size in dim_sizes)
                # If it's already been generated, we try again.
                if random_res in resolutions:
                    continue  # pragma: no cover
                resolutions.add(random_res)
                yield random_res


def increment_resolution(resolution: List[int], dim_sizes: Tuple[int, ...]) -> Optional[List[int]]:
    """
    Arguments:
        resolution: a list of the same length as dim_sizes. The value in each position should be
          between 1 and the corresponding value in dim_sizes inclusive.
        dim_sizes: a tuple of positive int values.
    Return:
        the "next" value of resolution: look for the last value in resolution that is strictly less than
        the corresponding value in dim_sizes. Return a variant of "resolution" in which that value is
        incremented and all later values (if any) are replaced by 1. If there is no strictly-less position,
        return None.
    """
    # Find last index at which value in resolution is less than value in dim_sizes
    index = len(dim_sizes) - 1
    while resolution[index] == dim_sizes[index]:
        if index == 0:
            return None
        index -= 1
    # Return a tuple which is identical to resolution before index, is incremented at index,
    # and has ones thereafter. The "noqa: E203" here prevents the space before the last colon triggering a
    # flake8 error. The space is there because "black" (incorrectly) insists on inserting it and cannot be
    # told not to.
    return list(resolution[:index]) + [resolution[index] + 1] + [1 for _ in resolution[index + 1 :]]  # noqa: E203


def is_choice_list(struc):
    """
    Returns whether struc is a valid choice list: a list of two or more elements, whose first element is
    a string consisting of "@" followed by a character a-z and optionally further occurrences of a-z and
    underscore (no digits allowed).
    """
    if isinstance(struc, list) and len(struc) > 2 and isinstance(struc[0], str) and struc[0].startswith("@"):
        if re.match("^@[a-z][a-z_]*$", struc[0]):
            return True
        logging.warning(f"Invalid choice-list variable name '{struc[0]}'")  # pragma: no cover
        logging.warning("Names must have an a-z character after '@' then one or more of a-z and _")  # pragma: no cover
    return False


def sample_from_distribution(spec: List[Any]) -> List[float]:
    """
    spec: list, whose first element should be a valid distribution name (see sample_at_point)
    and whose subsequent elements should be a number of samples (N) and the parameters of the
    distribution. For example: ["#normal#", 10, 1.5, 0.3] means "10 samples from a normal
    distribution with mean 1.5 and standard deviation 0.3".

    Returns: a list of N samples from the distribution, from cumulative probability values
    0.5/N, 1.5/N, ..., (N-0.5)/N. These samples are evenly spaced in the underlying probability
    space, rather than random. We do not sample the extremes (probabilities 0 and 1) because
    for the normal distribution and some others, these correspond to infinite values.
    """
    name = spec[0].lower()
    n_samples = int(spec[1])
    params = [float(v) for v in spec[2:]]
    return [sample_at_point(name, params, (0.5 + k) / n_samples) for k in range(n_samples)]


def sample_at_point(name: str, params: List[float], prob: float) -> float:
    """
    Returns the value of the cumulative density function defined by "name" and "params"
    at the probability "prob".
    """
    if name == "#uniform#":
        # Uniform distribution between low and high
        if len(params) != 2:
            raise ValueError(f"Distribution {name} needs exactly two parameters, not {params}")
        low, high = params
        return low + (high - low) * prob
    if name == "#normal#":
        # Normal distribution with mean mu and standard deviation sigma
        if len(params) != 2:
            raise ValueError(f"Distribution {name} needs exactly two parameters, not {params}")  # pragma: no cover
        mu, sigma = params
        z = norm.ppf(prob)
        return mu + z * sigma
    # Other distributions could be added here
    raise ValueError(f"Unknown distribution name {name}")


def get_dimensions(struc: Any, dim_dict: Dict[str, int]) -> None:
    """
    struc: any structure
    dim_dict: a dictionary from choice-list variable names (without the initial "@") to the number of choices
    (length of the rest of the list) for that name.

    Recurse down into "struc" through lists and dicts only, looking for choice lists. When a choice list is found,
    if the first element is a string that is already in dim_dict, a check is made that the length of the rest of the
    list is the same as the value in dim_dict. Otherwise, the name is added to dim_dict with that length as the value.
    """
    if is_choice_list(struc):
        head = struc[0]
        # Determine the dimension name: the part of the string after "@".
        dim_name = head[1:]
        if isinstance(struc[1], str) and struc[1].startswith("#") and struc[1].endswith("#"):
            dim_size = int(struc[2])
            if dim_size < 1:
                raise ValueError(f"Bad dimension size {dim_size} in {struc}")  # pragma: no cover
            # Replace distribution spec with samples from the distribution
            for index, value in enumerate(sample_from_distribution(struc[1:]), 1):
                if index < len(struc):
                    struc[index] = value
                else:
                    struc.append(value)
        else:
            dim_size = len(struc[1:])
        if dim_name in dim_dict:
            # Must be the same number of choices every time it is mentioned
            if dim_size != dim_dict[dim_name]:
                raise ValueError(
                    f"Length mismatch for @{dim_name}: {dim_size} in {struc}, {dim_dict[dim_name]} elsewhere"
                )
        else:
            dim_dict[dim_name] = dim_size
    elif isinstance(struc, list):
        # Recurse through (other) lists
        for sub in struc:
            get_dimensions(sub, dim_dict)
    elif isinstance(struc, dict):
        # Recurse through dicts - but not through any other object
        for sub in struc.values():
            get_dimensions(sub, dim_dict)


def apply_selection(sel_dict: Dict[str, int], struc: Any) -> Any:
    """
    sel_dict: dictionary from dimension names to index (from 1) in that dimension
    struc: a structure containing choice lists whose names are represented in sel_dict.
    """
    if is_choice_list(struc):
        # Select the element from struc denoted by sel_dict[struc[0][1:]]
        dim_name = struc[0][1:]
        selected = struc[sel_dict[dim_name]]
        # Allow for nested choice lists.
        return apply_selection(sel_dict, selected)
    elif isinstance(struc, list):
        # Recurse through lists
        return [apply_selection(sel_dict, sub) for sub in struc]  # pragma: no cover
    elif isinstance(struc, dict):
        # Recurse through dicts
        return dict((key, apply_selection(sel_dict, value)) for key, value in struc.items())
    else:
        # Do not recurse through anything else
        return struc
