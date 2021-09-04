# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path

INT_REQ_NAME = "internal_requirements.txt"


def test_monorepo_structure():
    with_init = defaultdict(list)
    without_init = defaultdict(list)
    depends_on = defaultdict(set)
    families = [Path("libraries"), Path("projects")]
    library_subrepo_names = set(path.name for path in Path("libraries").glob("*"))
    for family in families:
        # libraries/ and projects/ must both be present
        assert family.is_dir(), f"{family} is missing or not a directory"
        for subrepo_dir in family.glob("*"):
            # Every member of libraries/ and projects/ must be a directory...
            assert subrepo_dir.is_dir(), f"{subrepo_dir} must be a directory"
            subrepo = subrepo_dir.name
            # ...and have a name starting in upper case.
            assert subrepo[0].isupper(), f"{subrepo} (under {family}) should start with an upper case letter"
            # We skip further checks if .git exists (so it's presumably a submodule) or if
            # setup.py does not exist (so it isn't a proper subrepository yet).
            if (subrepo_dir / ".git").exists() or not (subrepo_dir / "setup.py").exists():
                continue
            # These files must exist if setup.py exists.
            for name in ["requirements.txt", "requirements_dev.txt", INT_REQ_NAME]:
                assert (
                    subrepo_dir / name
                ).is_file(), f"{subrepo_dir / name} must exist since {subrepo_dir / 'setup.py'} does."
            # internal_requirements.txt must consist only of library subrepo names.
            int_req_path = subrepo_dir / INT_REQ_NAME
            with int_req_path.open() as fh:
                for line in fh.readlines():
                    line = line.strip()
                    assert (
                        line in library_subrepo_names
                    ), f"Invalid dependency name {line} (not a subrepo under libaries/) in {int_req_path}"
                    assert line != subrepo, f"Subrepo {subrepo} should not depend on itself (in {int_req_path})"
                    depends_on[line].add(subrepo)
            # Record the package directories with and without __init__.py.
            for subdir in subrepo_dir.glob("*"):
                if subdir.is_dir():
                    if (subdir / "__init__.py").exists():
                        with_init[subdir.name].append(subrepo)
                    else:
                        without_init[subdir.name].append(subrepo)
            # If there are tests, check the .coveragerc file is present and looks OK.
            if (subrepo_dir / "tests").exists():
                check_coveragerc(subrepo_dir)
    # If any top level package name occurs with an __init__.py, there must be no other packages of the same
    # name, with or without __init__.py, or their contents will not be importable.
    for name in with_init:
        assert (
            len(with_init[name] + without_init[name]) == 1
        ), f"Multiple packages named {name}, at least one with an __init__.py file"
    # Check dependencies are complete and do not have circularities.
    for name1 in depends_on:
        for name2 in depends_on[name1]:
            if name2 in depends_on:
                for name3 in depends_on[name2]:
                    assert name3 in depends_on[name1], (
                        f"{name1} depends on {name2} which depends on {name3}, "
                        f"but {name3} is not in {name1}'s {INT_REQ_NAME} file"
                    )
                    assert (
                        name3 != name1
                    ), f"Mutual dependency in {INT_REQ_NAME} files: {name1} on {name2} and vice versa"


def check_coveragerc(subrepo_dir: Path) -> None:
    covrc_file = subrepo_dir / ".coveragerc"
    assert covrc_file.exists(), f"Must be present because {subrepo_dir}/tests is present: {covrc_file}"
    config = ConfigParser()
    config.read(covrc_file)
    assert "run" in config
    assert "source" in config["run"]
    for name in config["run"]["source"].split("\n"):
        name = name.strip()
        required = subrepo_dir / name
        assert required.is_dir(), f"{covrc_file} assumes non-existent directory {required}"
    assert "report" in config
    assert "fail_under" in config["report"], f"{covrc_file} must have a 'fail_under' line in its 'report' section"
