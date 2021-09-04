# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This is a submodule used to retrieve the example data sets.

Its API may frequently change and it should *not* be used in production.

Exports:
    Dataset, which is essentially a dictionary of data frames
    load_dataframes_from_directory, a function reading all data frames in a directory into a dictionary
"""
import logging
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

CONDITION_KEY = "_conditions"


def missing_directory_message(path: Path) -> Optional[str]:  # pragma: no cover
    path = path.absolute()
    if path.is_dir():
        return None
    ancestor = path
    while ancestor.parent != ancestor:
        ancestor = ancestor.parent
        if ancestor.is_dir():
            break
    return f"Dataset directory {path} not found (only found {ancestor})"


class Dataset(Dict[str, pd.DataFrame]):
    """A class representing a set of data frames in a given directory.

    Methods:
        __getitem__, so that the data can be accessed using ``dataset[key]`` syntax
        items, so that one can iterate over pairs (key, data frame) as in ``dict.items()``
        get_a_frame, gives a data frame, what is useful for illustratory purposes
    """

    def __init__(self, path: PathLike) -> None:
        self._path = Path(path)
        assert self._path.is_dir(), missing_directory_message(self._path)
        all_csvs = self._path.glob("*.csv")
        frames = dict(map(load_dataframe, all_csvs))
        if CONDITION_KEY in frames:
            conditions = frames[CONDITION_KEY]
            self.conditions = {idx: row.to_dict() for idx, row in conditions.set_index("SampleID").iterrows()}
            frames.pop(CONDITION_KEY)
        else:
            self.conditions = {key: {} for key in frames.keys()}
        super().__init__(frames)
        self.check_conditions_coverage()

    def check_conditions_coverage(self):
        """
        Warn if the contents of the _conditions.csv file do not exactly match the data files in the folder.
        """
        condition_keys = set(self.conditions.keys())
        data_keys = set(self.keys())
        n_condition_only_keys = len(condition_keys.difference(data_keys))
        if n_condition_only_keys > 0:  # pragma: no cover
            logging.warning(
                f"{self._path} has {n_condition_only_keys} rows in {CONDITION_KEY}.csv with no corresponding data file"
            )
        n_data_only_keys = len(data_keys.difference(condition_keys))
        if n_data_only_keys > 0:  # pragma: no cover
            logging.warning(
                f"{self._path} has {n_data_only_keys} data files with no corresponding row in {CONDITION_KEY}.csv"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path='{self._path}')"

    def get_a_frame(self, index: int = 0) -> pd.DataFrame:
        """A utility function, returning a data frame at position `index` in lexicographical order of the keys."""
        keys = sorted(self.keys())
        key = keys[index]
        return self[key]

    def items_by_well(self) -> List[Tuple[Optional[str], str, pd.DataFrame]]:
        """
        Returns a sorted list of tuples of the form (well_id, sample_id, data_frame), where well_id is the value
        of the "Well" field in the conditions, or None if that is absent. The ordering is by well row (letter)
        and column (number) if there are well IDs, otherwise alphabetically by sample ID.
        """
        items = [(self.conditions[sample_id].get("Well", None), sample_id, value) for sample_id, value in self.items()]

        def ordering_tuple(tup: Tuple[Optional[str], str, Any]) -> Tuple[str, int]:
            well, sample_id, _ = tup
            try:
                return well[0], int(well[1:])  # type: ignore
            except (ValueError, IndexError, TypeError):
                return sample_id, 0

        return sorted(items, key=ordering_tuple)  # type: ignore

    def plate_layout(self) -> Optional[Tuple[List[str], List[int]]]:
        """
        Attempts to return the set of letters (row IDs) and numbers (column IDs) for the wells in the dataset,
        or None if that fails (most likely because there are no wells defined).
        """
        wells = set(self.conditions[sample_id].get("Well", None) for sample_id in self)
        try:  # pragma: no cover
            well_letters = sorted(set(w[0] for w in wells))
            well_numbers = sorted(set(int(w[1:]) for w in wells))
            return well_letters, well_numbers
        except (ValueError, IndexError, TypeError):
            return None


def load_dataframe(csv_path: PathLike) -> Tuple[str, pd.DataFrame]:
    """Returns a tuple (name, data frame). Used to construct a data set by `load_dataframes_from_directory`.

    See:
        load_dataframes_from_directory
        Dataset
    """
    return Path(csv_path).stem, pd.read_csv(csv_path)  # type: ignore # auto
