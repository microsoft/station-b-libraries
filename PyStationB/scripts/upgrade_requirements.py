#!/usr/bin/env python
# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------

"""
Create updated versions of requirements files by creating a conda environment from existing requirements
files but without version constraints. If the resulting environment behaves well, the modified environment
and requirements files can be committed.

Note that upgrading versions may result in new (probably spurious) pyright and/or mypy errors being triggered,
and in slightly different png files being created in tests, requiring additional "figureNNN.png" files to be
added. See Utilities/psbutils/filecheck.py for the latter procedure.
"""
import os
import subprocess
from pathlib import Path
from typing import Dict, Set

REPO_NAME = "PyStationB"
ENV_NAME = "PyStationB"
ENV_YML = "environment.yml"
REQ_DEV_TXT = "requirements_dev.txt"
REQ_TXT = "requirements.txt"
FIXED_REQ_TXT = "fixed_requirements.txt"

FILE_DCT = {
    ENV_YML: "environment_upgraded.yml",
    REQ_TXT: "requirements_upgraded.txt",
    REQ_DEV_TXT: "requirements_upgraded_dev.txt",
}

NAME_DCT = FILE_DCT.copy()
NAME_DCT.update({ENV_NAME: "PyStationBUpgraded"})

# Start of line in "conda env create" output indicating what packages and versions were installed.
SUCCESS_KEY = "Successfully installed "
# Requirements file names that should be modified with new versions but not used to collect required packaged.
ADDITIONAL_NAMES = ["additional_requirements.txt", "additional_requirements_dev.txt"]


def get_fixed_requirements() -> Dict[str, str]:  # pragma: no cover
    """
    :return: a dictionary of entries key: val from the lines key==val in FIXED_REQ_TXT if that exists;
    otherwise an empty dictionary.
    """
    result = {}
    path = Path(FIXED_REQ_TXT)
    if path.is_file():
        with path.open() as fp:
            for line in fp:
                fields = line.strip().split("==")
                if len(fields) == 2:
                    result[fields[0]] = fields[1]
    return result


def upgrade_packages():  # pragma: no cover
    """
    Create updated versions of requirements files by creating a conda environment from existing requirements
    files but without version constraints.
    """
    # Change to the PyStationB directory. We assume this file lives in the scripts/ directory.
    root_dir = Path(__file__).absolute().parent.parent
    assert root_dir.name == REPO_NAME, f"root_dir is {root_dir}, should have basename {REPO_NAME}"
    os.chdir(str(root_dir))
    # Read fixed requirements if any.
    fixed_reqs = get_fixed_requirements()
    # For each name1, name2 pair, read the file named name1 and create file name2 with version constraints
    # removed, and with file references suitably suffixed.
    for name1, name2 in FILE_DCT.items():
        edit_environment_name_and_remove_version_constraints(name1, name2, fixed_reqs)
    cmd = f"conda env create --force --file {NAME_DCT[ENV_YML]}"
    print(f"Running conda env create to create environment {NAME_DCT[ENV_NAME]}. This could take several minutes...")
    process = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Print error messages if any
    for line in process.stderr.split("\n"):
        print(line)
    # Go through stdout lines, looking for package version line which should be of form
    # "Successfully installed foo-1.2 bar-baz-0.4.6 ...". If it's not found, we return, having (we hope)
    # already clarified what's gone wrong by printing the stderr lines.
    selected_versions = get_selected_versions(process.stdout)
    if not selected_versions:
        print("Command appears to have FAILED")
        return
    print(f"Command appears to have worked, found {len(selected_versions)} package versions.")
    print(f"To try out the environment: conda activate {NAME_DCT[ENV_NAME]}")
    print("To see the changes: git diff")
    for name in FILE_DCT.values():
        Path(name).unlink()
    reported: Set[str] = set()
    for name in [REQ_TXT, REQ_DEV_TXT]:
        update_with_versions(name, selected_versions, True, reported)
    for name in ADDITIONAL_NAMES:
        update_with_versions(name, selected_versions, False, reported)


def edit_environment_name_and_remove_version_constraints(
    name1: str, name2: str, fixed_reqs: Dict[str, str]
) -> None:  # pragma: no cover
    """
    Creates file name2 as an edited version of name1, for generating a new conda environment.
    :param name1: name of a file to read from
    :param name2: name of a file to write to
    """
    with Path(name1).open("r") as fp:
        with Path(name2).open("w") as gp:
            for line in fp:
                line = edit_requirements_line(line, fixed_reqs)
                gp.write(line + "\n")


def edit_requirements_line(line: str, fixed_reqs: Dict[str, str]) -> str:
    """
    :param line: a line from an environment or requirements file
    :return: modified version of line for the "upgraded" counterpart of the file.
    """
    line = line.rstrip()
    # Look for a line ending with a space and some member of NAME_DCT - either an environment
    # name or a file name. If the former is found, replace it with the latter.
    for key, val in NAME_DCT.items():
        if line.endswith(" " + key):
            line = line[: -len(key)] + val
            break
    # If the line contains "==" and does not start with a space, discard the "==" and the following
    # text, which is assumed to be a version number.
    if not line.startswith(" "):
        pos = line.find("==")
        if pos > 0:
            line = line[:pos]
            if line in fixed_reqs:
                line = f"{line}=={fixed_reqs[line]}"
    return line


def get_selected_versions(output: str) -> Dict[str, str]:
    """
    :param output: a string, typically representing the output of a process
    :return: a dictionary from package names to versions; empty if no line started with SUCCESS_KEY.
    """
    selected_versions = {}
    for line in output.split("\n"):
        if line.startswith(SUCCESS_KEY):
            # Assume everything following SUCCESS_KEY will be of the form "package-version"
            versions = line[len(SUCCESS_KEY) :].split()  # noqa: E203
            for version in versions:
                fields = version.split("-")
                selected_versions["-".join(fields[:-1])] = fields[-1]
            break
    return selected_versions


def update_with_versions(
    name: str, selected_versions: Dict[str, str], with_subrepos: bool, reported: Set[str]
) -> None:  # pragma: no cover
    """
    Rewrites file(s) named "name" with package versions specified in "selected_versions".
    :param name: name of a file to look for in current directory (root of main repo) and possible all subrepos
    :param selected_versions: dictionary from package names to versions
    :param with_subrepos: whether to look for files in subrepos as well as at top level
    :param reported: set of packages names for which changes have been reported.
    """
    paths = [Path(name)]
    if with_subrepos:
        paths += sorted(Path(".").glob(f"*/*/{name}"))
    for path in paths:
        with path.open("r") as fp:
            lines = fp.readlines()
        upgraded_lines = [attach_specified_version(line, selected_versions, reported) for line in lines]
        if upgraded_lines != lines:
            print(f"Updating {path}")
            with path.open("w") as gp:
                for line in upgraded_lines:
                    gp.write(line + "\n")


def attach_specified_version(line: str, selected_versions: Dict[str, str], reported: Set[str]) -> str:
    """
    :param line: a line from a requirements file
    :param selected_versions: desired version of each package
    :param reported: set of packages names for which changes have been reported.
    :return: version of line with relevant selected version (if present; otherwise line is unchanged).
    """
    line = line.rstrip()
    fields = line.split("==")
    if len(fields) == 2 and fields[0] in selected_versions:
        updated = f"{fields[0]}=={selected_versions[fields[0]]}"
        if updated != line:
            line = updated
            print(f"Upgrading {fields[0]} from {fields[1]} to {selected_versions[fields[0]]}")
            reported.add(fields[0])
    return line


if __name__ == "__main__":  # pragma: no cover
    upgrade_packages()
