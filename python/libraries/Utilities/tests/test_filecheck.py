# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pytest
from psbutils.filecheck import FIGURE_DATA_PATH, compare_files_in_directories, figure_found, MANIFEST_FILE_BASE
from psbutils.misc import find_subrepo_directory


def test_figure_found():
    figure_data_path = find_subrepo_directory() / FIGURE_DATA_PATH
    figure_data_path.mkdir(parents=True, exist_ok=True)
    name = "foo"
    subdir = figure_data_path / name
    assert not subdir.exists()
    png0 = subdir / "figure000.png"
    assert not png0.exists()
    ax = plt.axes()
    assert not figure_found(ax, name)
    assert png0.exists()
    assert figure_found(ax, name)
    hashfile_path = subdir / MANIFEST_FILE_BASE
    assert hashfile_path.exists()
    with hashfile_path.open() as inp:
        lines = inp.readlines()
        assert len(lines) == 1
        assert lines[0].split()[0] == "figure000.png"
    hashfile_path.unlink()
    png0.unlink()
    subdir.rmdir()


@pytest.mark.timeout(20)
def test_compare_files_in_directories():
    comparison_dir = f"{find_subrepo_directory()}/tests/data/figures/test_compare_files_in_directories"
    lines = list(compare_files_in_directories([comparison_dir]))
    assert len(lines) >= 7
    expected = [
        "figure000.png vs figure001.png: identical content",
        "figure000.png vs figure002.png: difference proportion 0.006445, SD ratio 0.060168",
        "figure000.png vs figure003.png: different size: (640, 480) (figure000.png) vs (900, 900) (figure003.png)",
        "figure001.png vs figure002.png: difference proportion 0.006445, SD ratio 0.060168",
        "figure001.png vs figure003.png: different size: (640, 480) (figure001.png) vs (900, 900) (figure003.png)",
        "figure002.png vs figure003.png: different size: (640, 480) (figure002.png) vs (900, 900) (figure003.png)",
    ]
    assert [line.strip() for line in lines[-6:]] == expected
