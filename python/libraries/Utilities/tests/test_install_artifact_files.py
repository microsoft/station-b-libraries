# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp

from psbutils.install_artifact_files import install_artifact_files
from psbutils.misc import find_subrepo_directory


def test_install_artifact_files():
    """
    Check that install_artifact_files unzips a file into the right place. For the context
    for this operation, see the documentation in filecheck.py.
    """
    # Temporary directory to unzip the drop file into.
    tmp_dir = Path(mkdtemp())
    fig_dir = tmp_dir / "figures"
    fig_dir.mkdir()
    # cwd should be the parent directory of the tests/ directory.
    subrepo_dir = find_subrepo_directory()
    cwd = Path.cwd()
    os.chdir(fig_dir)
    # An example drop.zip file, originally created by downloading a build artifact following the
    # instructions in filecheck.py. It contains one .png file.
    drop_file = subrepo_dir / "tests" / "data" / "drop.zip"
    # Install the contents of the drop file.
    install_artifact_files(drop_file)
    # Check that exactly one .png file was installed.
    assert len(list(Path(".").rglob("*.png"))) == 1
    os.chdir(cwd)
    # Clean up.
    shutil.rmtree(tmp_dir)
