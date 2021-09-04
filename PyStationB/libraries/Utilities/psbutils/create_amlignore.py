# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import os
import sys
from pathlib import Path
from typing import List, Set

from psbutils.misc import ROOT_DIR
from psbutils.type_annotations import PathOrString

AMLIGNORE = ROOT_DIR / ".amlignore"
AMLIGNORE_FIXED = ROOT_DIR / ".amlignore_fixed"


def locate_subrepo(subrepo_name_or_path: PathOrString) -> Path:
    """
    Args:
        subrepo_name_or_path: string or Path that can be
        (1) a subrepository name, e.g. "ABEX";
        (2) the path to a subrepository relative to the repository root, e.g. "libraries/ABEX"; or
        (3) the path to a file or directory within a subrepository, e.g. "libraries/ABEX/abex/foo.py".
    Returns:
        the Path to the subrepository relative to the repository root, e.g. Path("libraries/ABEX").
    Raises a ValueError if the result cannot be calculated.
    """
    name_path = Path(subrepo_name_or_path)
    if name_path.exists():
        # Check for name_path being inside a subrepository.
        candidate = name_path.absolute()
        while True:
            parent = candidate.parent
            if parent.name in ["libraries", "projects"]:
                return candidate.relative_to(ROOT_DIR)
            if parent == candidate:
                break  # pragma: no cover
            candidate = parent
    for top in [ROOT_DIR, ROOT_DIR / "libraries", ROOT_DIR / "projects"]:  # pragma: no cover
        subrepo = top / subrepo_name_or_path
        if subrepo.is_dir():
            return subrepo.absolute().relative_to(ROOT_DIR)
    raise ValueError(f"Cannot find a subrepo named {subrepo_name_or_path}")  # pragma: no cover


def create_amlignore(subrepo_name: PathOrString):  # pragma: no cover
    """
    Given a subrepository name or path (e.g. "ABEX" or "libraries/ABEX" or something underneath it, as in
    locate_subrepo), creates .amlignore to contain the contents of .amlignore_fixed plus any subrepositories
    NOT required for the one provided (according to its internal_requirements.txt).
    """
    with AMLIGNORE.open("w") as fp:
        for line in amlignore_lines(subrepo_name):
            fp.write(line + "\n")


def amlignore_lines(subrepo_name_or_path: PathOrString) -> List[str]:
    """
    Given a subrepository name or path (as in locate_subrepo), returns lines to write to .amlignore to contain
    the contents of .amlignore_fixed plus any subrepositories NOT required for the one provided (according to
    its internal_requirements.txt). We use "/" as separator even if os.sep is something else.
    """
    ignore: Set[str] = set()
    if AMLIGNORE_FIXED.exists():
        with AMLIGNORE_FIXED.open() as fp:
            for line in fp:
                ignore.add(line.strip())
    subrepo = locate_subrepo(subrepo_name_or_path)
    required_subrepos = {subrepo}
    int_req = ROOT_DIR / subrepo / "internal_requirements.txt"
    if int_req.exists():
        with int_req.open() as fp:  # pragma: no cover
            for line in fp:
                required_subrepos.add(locate_subrepo(line.strip()))
    all_subrepos_abs = set((ROOT_DIR / "libraries").glob("[A-Z][A-Za-z0-9]*")).union(
        (ROOT_DIR / "projects").glob("[A-Z][A-Za-z0-9]*")
    )
    all_subrepos = set(sr.relative_to(ROOT_DIR) for sr in all_subrepos_abs)
    ignored_subrepos = all_subrepos.difference(required_subrepos)
    ignore.update(str(sr).replace(os.sep, "/") for sr in ignored_subrepos)
    return sorted(ignore)


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} subrepo_name_to_include")
    else:
        create_amlignore(sys.argv[1])
