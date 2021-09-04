#!/usr/bin/env python
# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import staticchar
from psbutils.psblogging import logging_to_stdout
from staticchar.datasets import Dataset
from staticchar.plate import Plate

OUTPUT_DIRECTORY = Path("SC_RESULTS")
CONFIG_YML = "config.yml"


def new_run_index(config_name_pattern: str = "*") -> str:
    """
    Returns a 3-character string representing an integer, with leading zeros: "001", "002" etc.
    The value chosen is the first one for which no subdirectory of that name exists three levels
    under OUTPUT_DIRECTORY (the two intermediate levels being for the config and the dataset).
    """
    n = 1
    existing = set(path.name for path in OUTPUT_DIRECTORY.glob(f"{config_name_pattern}/*/[0-9][0-9][0-9]"))
    while True:
        index = f"{n:03d}"
        if index not in existing:
            return index
        n += 1  # pragma: no cover


def unique_name_dictionary(paths: Iterable[Path]) -> Dict[Path, str]:
    """
    Args:
        paths: a list of path names

    Returns:
        a dictionary mapping each path in paths to a unique name. By default
        default the name is path.name, but if multiple paths have the same path.name,
        the suffixes _1, _2 etc are used after the first one.
    """
    seen: Dict[str, int] = {}  # number of times each path.name has been seen so far
    result: Dict[Path, str] = {}
    for path in paths:
        base = path.name
        for suffix in [".yml", ".yaml"]:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        if base in seen:  # pragma: no cover
            result[path] = f"{base}_{seen[base]}"
            seen[base] += 1
        else:
            result[path] = base
            seen[base] = 1
    return result


def run(paths: Iterable[Path]) -> List[Path]:
    """
    Args:
        paths: an iterable that includes at least one .yml or .yaml file and at least one directory
               containing .csv files.

    Returns:
        a list of output directories created by the function, one for each combination of config file and .csv file,
        each containing .png and .txt files
    """
    logging_to_stdout()
    path_set = set(paths)
    config_paths = unique_name_dictionary(
        path for path in path_set if path.name.endswith(".yml") or path.name.endswith(".yaml")
    )
    data_dirs = unique_name_dictionary(path for path in path_set if path.is_dir() and list(path.glob("*.csv")))
    unaccounted = path_set.difference(config_paths.keys()).difference(data_dirs.keys())
    if unaccounted:  # pragma: no cover
        print(f"Warning: ignoring these argument(s): {','.join(str(u) for u in sorted(unaccounted))}")
    if not config_paths:  # pragma: no cover
        raise ValueError("Must supply at least one config file with suffix .yml or .yaml")
    if not data_dirs:  # pragma: no cover
        raise ValueError("Must supply at least one data directory containing .csv files")
    output_dirs = []

    for cfg_path, cfg_path_base in sorted(config_paths.items()):
        cfg_run_index = new_run_index(cfg_path_base)
        config = staticchar.config.load(cfg_path, staticchar.config.CharacterizationConfig)
        cfg_dir = OUTPUT_DIRECTORY / cfg_path_base
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for data_dir, data_dir_base in sorted(data_dirs.items()):
            output_dir = cfg_dir / data_dir_base / cfg_run_index
            output_dir.mkdir(parents=True)
            print(f"Writing files to {output_dir}")
            shutil.copyfile(cfg_path, output_dir / CONFIG_YML)
            dataset = Dataset(data_dir)
            plate = Plate(dataset, config)
            plate.characterize(output_dir)
            output_dirs.append(output_dir)
    return output_dirs


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) >= 3:
        run([Path(s) for s in sys.argv[1:]])
    else:
        print(f"Usage: {sys.argv[0]} config1.yml config2.yml ... data_dir1 data_dir2 ...")
