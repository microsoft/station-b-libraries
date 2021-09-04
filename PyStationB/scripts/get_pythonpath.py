#!/usr/bin/env python
# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------

import argparse
import os
from pathlib import Path
from typing import List, Optional

CATEGORY_NAMES = ["libraries", "projects"]


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Returns parsed arguments for the provided arg list or sys.argv[1:]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", action="store_true", help="Whether to output a line for VSCode settings.json")
    parser.add_argument("-s", "--subrepo", default="", help="Subrepository name")
    return parser.parse_args(args)


def get_pythonpath(arg_list: Optional[List[str]] = None) -> str:
    """
    Print PYTHONPATH for the whole repo (if args.subrepo missing) or the specific subrepo.
    Produces different results when called from WSL/Linux and Windows!
    """
    args = parse_args(arg_list)
    root_dir = Path(__file__).parent.parent.absolute()
    for category in CATEGORY_NAMES:
        if not (root_dir / category).is_dir():  # pragma: no cover
            raise FileNotFoundError(f"Does not exist, or is not a directory: {root_dir / category}")
    is_windows = os.name == "nt"
    subrepo_dirs = get_subrepo_directories(args.subrepo, root_dir)
    pythonpath_dirs = [str(subrepo_dir) for subrepo_dir in subrepo_dirs] + ["."]
    if args.json:
        paths = ", ".join(f'"{p}"' for p in pythonpath_dirs).replace("\\", "\\\\")
        result = f'    "python.analysis.extraPaths": [{paths}]'
    else:
        separator = ";" if is_windows else ":"
        result = separator.join(pythonpath_dirs)
    print(result)
    return result


def get_subrepo_directories(subrepo_name: Optional[str], root_dir: Path) -> List[Path]:
    """
    Returns a list of subrepository paths, either for the whole repo (all libraries and projects,
    plus the top level) or for a specific subrepo name (that subrepo plus contents of internal_requirements.txt
    if present).
    """
    subrepo_dirs: List[Path] = []
    if subrepo_name:
        subrepo_names = [subrepo_name]
        int_reqs = list(root_dir.glob(f"libraries/{subrepo_name}/internal_requirements.txt")) + list(
            root_dir.glob(f"projects/{subrepo_name}/internal_requirements.txt")
        )
        if int_reqs:
            with int_reqs[0].open() as fh:
                subrepo_names.extend([line.strip() for line in fh])
        subrepo_dirs = []
        for subrepo_name in subrepo_names:
            subrepo_dirs += list(root_dir.glob(f"libraries/{subrepo_name}")) + list(
                root_dir.glob(f"projects/{subrepo_name}")
            )
    else:
        for category in CATEGORY_NAMES:
            for subrepo in (root_dir / category).glob("[A-Z][A-Za-z]*"):
                if subrepo.is_dir():
                    subrepo_dirs.append(subrepo)
        subrepo_dirs.append(root_dir)
    return subrepo_dirs


if __name__ == "__main__":  # pragma: no cover
    get_pythonpath()
