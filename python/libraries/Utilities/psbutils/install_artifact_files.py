#!/usr/bin/env python
# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------

"""
This script can be run on one or more "drop.zip" files downloaded from builds, and will copy new figureNNN.png
files to an appropriate place, ready for inspection and possible "git add". See filecheck.py for instructions
on locating and downloading "drop.zip" files from a build.

This script should be run from a "figures" directory, e.g. your_repo/tests/data/figures.
"""

import sys
from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZipFile

from psbutils.filecheck import figure_found


def install_artifact_files(drop_file: Path):
    if not drop_file.exists():
        raise FileNotFoundError(f"File not found: {drop_file.absolute()}")  # pragma: no cover
    if not drop_file.name.endswith(".zip"):
        raise ValueError(f"Argument should be a .zip file, not {drop_file}")  # pragma: no cover
    if Path.cwd().name != "figures":
        raise ValueError(f"Run this script from a 'figures' directory, not {Path.cwd()}")  # pragma: no cover
    tmp_dir = mkdtemp()
    with ZipFile(drop_file, "r") as zip:
        zip.extractall(tmp_dir)
    tmp_dir_path = Path(tmp_dir)
    for fig_png in tmp_dir_path.rglob("figure[0-9][0-9][0-9].png"):
        fig_png_rel = fig_png.relative_to(tmp_dir_path)
        parts = fig_png_rel.parts
        if len(parts) >= 4 and parts[0] == "drop" and parts[2] == "figures":
            target_dir = Path(*parts[3:-1])
            if figure_found(fig_png, target_dir, Path.cwd()):
                print(f"Identical file already exists, so ignoring: {fig_png_rel}")
        else:
            print(f"WARNING: unexpected path in zip archive: {fig_png_rel}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    for arg in sys.argv[1:]:
        install_artifact_files(Path(arg))
