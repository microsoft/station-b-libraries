# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import shutil
from pathlib import Path

import pytest
from abex.plotting.bayesopt_plotting import get_logged_img_title
from psbutils.misc import ROOT_DIR

REL_PATH_TO_ROOT_DIR = Path(__file__).relative_to(ROOT_DIR)


@pytest.fixture(scope="module")
def dummy_path_1():
    dummy_dir = ROOT_DIR / "foo" / "bar"
    dummy_dir.mkdir(exist_ok=True, parents=True)
    dummy_path = dummy_dir / "baz.yml"
    dummy_path.touch()
    yield dummy_path
    shutil.rmtree(ROOT_DIR / "foo")


@pytest.fixture(scope="module")
def dummy_path_2():
    dummy_dir = ROOT_DIR / "abc" / "def" / "ghi"
    dummy_dir.mkdir(exist_ok=True, parents=True)
    dummy_path = dummy_dir / "iter1_seed00000.png"
    dummy_path.touch()
    yield dummy_path
    shutil.rmtree(ROOT_DIR / "abc")


def test_get_logged_img_title(dummy_path_1: Path, dummy_path_2: Path):
    """
    If plot title is defined, that should be the first item in the logged image title. If a
    path to the stored plot is provided, take all directories from root down

    Returns:

    """
    assert get_logged_img_title() == "plot"
    assert get_logged_img_title(title="plot123") == "plot123"
    assert get_logged_img_title(fname=dummy_path_1) == "plot_baz"
    assert get_logged_img_title(title="plot_123", fname=dummy_path_1) == "plot_123_baz"

    # assert that even with a longer path we only get the parent and grandparent directories
    assert get_logged_img_title(fname=dummy_path_2) == "plot_iter1_seed00000"
    assert get_logged_img_title(title="plot_123", fname=dummy_path_2) == "plot_123_iter1_seed00000"
