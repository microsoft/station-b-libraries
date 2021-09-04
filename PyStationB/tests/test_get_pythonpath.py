# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from collections import defaultdict
from pathlib import Path

from scripts.get_pythonpath import get_pythonpath, get_subrepo_directories


def test_get_pythonpath():
    msg1 = get_pythonpath([])
    assert len(msg1.split(":")) >= 11
    msg2 = get_pythonpath(["-j"])
    assert msg2 != msg1
    msg3 = get_pythonpath(["-s", "ABEX"])
    assert len(msg3.split(":")) < len(msg1.split(":"))


def test_get_subrepo_directories_general():
    root_dir = Path(__file__).parent.parent
    get_subrepo_directories(None, root_dir)
    category_counts = defaultdict(int)
    subrepo_dirs = get_subrepo_directories(None, root_dir)
    for d in subrepo_dirs:
        category_counts[d.parent.name] += 1
    assert category_counts["libraries"] >= 4
    assert category_counts["projects"] >= 3
    # "+1" is for the root directory
    assert len(subrepo_dirs) == category_counts["libraries"] + category_counts["projects"] + 1


def test_get_subrepo_directories_specific():
    root_dir = Path(__file__).parent.parent
    get_subrepo_directories(None, root_dir)
    category_counts = defaultdict(int)
    subrepo_dirs = get_subrepo_directories("ABEX", root_dir)
    for d in subrepo_dirs:
        category_counts[d.parent.name] += 1
    # ABEX depends on at least some other libraries:
    assert category_counts["libraries"] >= 2
    assert "projects" not in category_counts
    # Only ABEX and other libraries, no projects or top level:
    assert len(subrepo_dirs) == category_counts["libraries"]
