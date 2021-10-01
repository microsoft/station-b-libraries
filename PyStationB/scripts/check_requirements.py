#!/usr/bin/env python
# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import collections
import sys
from pathlib import Path
from typing import List

from pkg_resources import parse_requirements

ROOT_DIR = Path(__file__).absolute().parent.parent
EXCLUSIONS = ["GlobalPenalisation"]


def read_requirements(path: Path):
    lines = []
    extras = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line and line[0].isalpha():
                lines.append(line)
            else:
                extras.append(line)
        return list(parse_requirements(lines)), extras


def get_subrepo_requirements(basename: str) -> List[str]:
    reqs = collections.defaultdict(set)
    parse_failures = []
    additional = ROOT_DIR / f"additional_{basename}"
    if additional.exists():
        a_reqs = read_requirements(additional)[0]
        for req in a_reqs:
            reqs[req.name].update(req.specs)  # type: ignore # auto
    for category in ["libraries", "projects"]:
        for subrepo in (ROOT_DIR / category).glob("*"):
            if subrepo.name in EXCLUSIONS: continue
            req_file = subrepo / basename
            if req_file.exists():
                try:
                    f_reqs, _ = read_requirements(req_file)
                    for req in f_reqs:
                        reqs[req.name].update(req.specs)  # type: ignore # auto
                except ValueError:  # pragma: no cover
                    parse_failures.append(req_file)
    if parse_failures:
        raise ValueError(
            f"Failed to parse requirements file(s) {' '.join(map(str, parse_failures))}"
        )  # pragma: no cover
    req_list = []
    for name, specs in reqs.items():
        req_str = name
        if len(specs) > 1:
            raise ValueError(f"Possible clash between specs for {name}: {specs}")  # pragma: no cover
        elif len(specs) == 1:
            rel, value = list(specs)[0]
            req_str += rel + value
        req_list.append(req_str)
    return sorted(req_list)


def get_non_empty_lines_stripping_comments(path: Path):
    """
    :param path: file to read
    :return: Lines in the file, stripping off any thing from "#" onwards, then discarding whitespace-only lines.
    """
    with path.open() as fh:
        env_lines = [line.split("#", 1)[0].rstrip() for line in fh]
        return [line for line in env_lines if line]


def process_environment_files(force: bool) -> bool:
    """
    :param force: whether to write out environment files.

    If force, write each subrepo's environment file as a copy of the top-level one, changing only the name
    to be that of the subrepo; and return True.

    Otherwise, check that this is already the case, and return whether it is.
    """
    env_lines = get_non_empty_lines_stripping_comments(ROOT_DIR / "environment.yml")
    name_lines = [line for line in env_lines if line.startswith("name: ")]
    assert len(name_lines) == 1
    env_lines = name_lines + [line for line in env_lines if line != name_lines[0]]
    ok = True
    for category in ["libraries", "projects"]:
        for subrepo in (ROOT_DIR / category).glob("*"):
            sub_env_file = subrepo / "environment.yml"
            expected = ["name: " + subrepo.name] + env_lines[1:]
            if force:
                if (subrepo / "requirements_dev.txt").exists():  # pragma: no cover
                    with sub_env_file.open("w") as gh:
                        for line in expected:
                            gh.write(f"{line}\n")
            elif sub_env_file.exists():
                sub_env_lines = get_non_empty_lines_stripping_comments(sub_env_file)
                if sub_env_lines != expected:  # pragma: no cover
                    sys.stderr.write(f"# Unexpected content in {sub_env_file}:\n")
                    sys.stderr.write("# Expected:\n")
                    for line in expected:
                        sys.stderr.write(f"{line}\n")
                    sys.stderr.write("# Found:\n")
                    for line in sub_env_lines:
                        sys.stderr.write(f"{line}\n")
                    ok = False
    return ok


def check_requirements(args: List[str]) -> bool:
    """
    :param args: argument list. Should be either [] or ["-f"].

    If args is ["-f"], writes top-level requirements.txt and requirements_dev.txt as the union of
    lower-level ones, and lower-level environment.yml files as a close copy (only the name is changed) of the
    top-level ones, and returns True.

    If args is [], checks that the above is already the case, printing error messages if not. Returns True
    if no discrepancies were found, otherwise False.

    Thus the usual mode of use would be:
        python scripts/check_requirements.py     # to see what problems there are
        python scripts/check_requirements.py -f  # if it's OK to propagate changes
    """
    if args == ["-f"]:
        force = True  # pragma: no cover
    elif not args:
        force = False
    else:
        raise ValueError(f"Usage: {sys.argv[0]} [-f]")  # pragma: no cover
    is_consistent = process_environment_files(force)
    for basename in ["requirements_dev.txt", "requirements.txt"]:
        subrepo_reqs = get_subrepo_requirements(basename)
        target = ROOT_DIR / basename
        if force:
            write_top_level_requirement_file(basename, subrepo_reqs, target)  # pragma: no cover
        elif not check_top_level_requirement_file(basename, subrepo_reqs, target):
            is_consistent = False  # pragma: no cover
    if is_consistent and not force: # pragma: no cover
        sys.stderr.write("# Environment and requirements files all look OK.\n")
    return is_consistent # pragma: no cover


def check_top_level_requirement_file(basename: str, subrepo_reqs: List[str], target: Path) -> bool:
    try:
        target_reqs1, other_lines = read_requirements(target)
        target_reqs = sorted(str(req) for req in target_reqs1)
    except ValueError:  # pragma: no cover
        raise ValueError(f"# Failed to parse requirements file {target}")
    is_consistent = target_reqs == subrepo_reqs
    if not is_consistent:  # pragma: no cover
        sys.stderr.write(f"# Mismatch between top-level {basename} and value calculated from subrepositories\n")
        for missing in sorted(set(subrepo_reqs).difference(target_reqs)):
            sys.stderr.write(f"# In subrepositories only: {missing}\n")
        for missing in sorted(set(target_reqs).difference(subrepo_reqs)):
            sys.stderr.write(f"# In top level only: {missing}\n")
        is_consistent = False
    return is_consistent


def write_top_level_requirement_file(basename, subrepo_reqs, target):  # pragma: no cover
    if target.exists():
        _, other_lines = read_requirements(target)
    else:
        other_lines = []
    with target.open("w") as gh:
        additional = ROOT_DIR / f"additional_{basename}"
        to_write = other_lines + subrepo_reqs
        if additional.exists():
            with additional.open() as fh:
                for line in fh:
                    if line.strip() not in to_write:
                        gh.write(line)
        for line in other_lines + subrepo_reqs:
            gh.write(f"{line}\n")


if __name__ == "__main__":
    check_requirements(sys.argv[1:])  # pragma: no cover
