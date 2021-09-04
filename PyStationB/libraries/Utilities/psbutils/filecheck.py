# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import collections
import hashlib
import inspect
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import SubplotBase
from PIL import Image, UnidentifiedImageError
from psbutils.misc import find_subrepo_directory


FIGURE_DATA_PATH = Path("tests") / "data" / "figures"
MANIFEST_FILE_BASE = "hashes.txt"

Plottable = Union[plt.Figure, plt.Axes, SubplotBase, Path]


def figure_found(
    fig_or_ax: Optional[Plottable], subdir_name: Union[Path, str], root_path: Path = FIGURE_DATA_PATH
) -> bool:
    """
    Arguments:
        fig_or_ax: pyplot Figure or Axes into which something has presumably been plotted;
           or a .png file that must exist. If None, the value of pyplot.gcf() is used.
        subdir_name: subdirectory of tests/data/figures in which to find a file matching ax
    Return:
        whether any existing figureNNN.png file in tests/data/figures/subdir_name matches ax. If not, a new
        figureNNN.png file is written, using the first unused value of NNN.

    This function should be called from tests of plotting code. If it triggers a test failure, read on; it
    is designed to make these tests easy to fix.

    How to use this function: in your test, add the line

       assert figure_found(fig_or_ax, "subdir_name")

    where "subdir_name" is unique in your test suite - e.g. a string starting with the name of the test in
    which it occurs. The first time you run your test, it will fail, because the subdirectory does not exist yet.
    However, it will be created and a file figure000.png will be written, either by saving it if it is a
    matplotlib object, or simply by copying it if it is an existing file.

    You will see a warning to check that file. You should inspect it and decide whether it looks OK. If it does, you
    should git add both the file itself and the hashes.txt file that should have been created in the subdirectory.
    If it does not, you should delete it and fix the code or the test. (Tip: if you are making several calls to
    figure_found in one test, you can omit the "assert"s initially to prevent the test failing, then reinstate them
    once the .png files are in place.)

    In future, if exactly the same "fig_or_ax" object is checked, the test should succeed. If it fails, you should again
    look at the new file. If it looks valid (most likely it is different at the binary level from the original one
    but not visibly different) you should git add it and all should be well. Otherwise you should remove it and
    fix the problem. You should see some lines associated with the test failure that give information on how
    similar the new figure is to existing ones; if it is identical at the pixel level, or there are statistics
    indicating it is extremely similar (see "image_similarity" below for details), you may decide to git add it
    without inspection.

    The hashes.txt mechanism is used to avoid having to recalculate hex digests for the files in the subdirectory
    every time tests are run. It is possible for hashes.txt to get out of step with the contents of the directory,
    for example if you delete an existing axNNN.png file (not the one created by figure_found in a failing test but
    installed after a successful run) and do not remove it from hashes.txt. It is always safe to delete hashes.txt;
    figure_found will automatically recalculate it if it is not present.

    We allow for multiple figureNNN.png files in a directory because several alternatives may be valid - e.g.
    matplotlib may yield trivially different results in Linux and Windows, or from different versions of matplotlib.

    There may also be different versions of these files created in Azure DevOps builds. If your tests calling
    figure_found succeed on desktop but fail in the build, do this:
      * From the Summary tab of your build, look in the "Related" column for a line saying "N published" for some N.
      * Click on that line and expand "drop". If any files were published, you should see them under the "Linux/figures"
        and/or "Windows_NT/figures" directories.
      * Download the "drop" folder to your local machine (three vertical dots at far right), cd to the
        tests/data/figures directory in your subrepo, and then run install_artifact_files.py (in this directory) or
        equivalently install_artifact_files.sh (in the top-level scripts/ directory) on the drop.zip file.
     *  Inspect the new files. If they look good, git add and commit them, and retry the build.
     *  If during this process, there are changes to any hashes.txt files, they should also be added and committed.
    """
    # Directory to look for axNNN.png files in. We assume figure_found has been called from a test file.
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    assert module is not None
    dir_path = find_subrepo_directory(module.__file__) / root_path / subdir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    # Update the manifest, creating it if necessary, and return an up to date basename-to-hexdigest dictionary.
    hex_hashes = _update_manifest(dir_path)
    # Choose the next available figureNNN.png name for the figure we are testing, save figure to that file, and
    # get its hexdigest.
    current_base = _new_ax_png_name(hex_hashes)
    # figureNNN.png path where we will (perhaps temporarily) store the image we've been handed.
    current_path = dir_path / current_base
    if fig_or_ax is None:
        fig_or_ax = plt.gcf()  # pragma: no cover
    if isinstance(fig_or_ax, Path):
        shutil.copyfile(fig_or_ax, current_path)
    else:
        fig = fig_or_ax.get_figure() if isinstance(fig_or_ax, plt.Axes) else fig_or_ax
        fig.savefig(current_path)  # type: ignore # auto
    with current_path.open(mode="rb") as f:
        contents = f.read()
        current_hashes = hashlib.sha256(contents).hexdigest()
    # See if any of the existing files has the hexdigest of the current one. If so, we assume the files are equal.
    # We remove the current figureNNN.png and return success.
    if current_hashes in hex_hashes.values():
        current_path.unlink()
        return True
    # If no existing file matches the current one, we leave the current one in place and return failure.
    # We do not add a line to the manifest file, in case the user forgets to delete it; it will be added the next
    # time the test is run (and then hashes.txt will also have to be checked in).
    sys.stderr.write(f"CHECK THIS FILE, THEN GIT-ADD OR DELETE IT: {current_path.absolute()}\n")
    for line in image_similarity_advice(current_path, sorted(hex_hashes)):
        sys.stderr.write(f"  {line}\n")  # pragma: no cover
    # If this variable is defined, treat it as a directory to which the png file should also be copied. This
    # is useful for publishing these files as artifacts of a build.
    additional = os.getenv("ADDITIONAL_FIGURE_FILE_DIRECTORY")
    if additional is not None:  # pragma: no cover
        additional_path = Path(additional) / subdir_name
        additional_path.mkdir(parents=True, exist_ok=True)
        sys.stderr.write(f"Copying {current_path} to directory {additional_path}\n")
        shutil.copyfile(current_path, additional_path / current_base)
    return False


def _new_ax_png_name(dct: Dict[str, str]) -> str:
    """
    Returns the first string of the form "axNNN.png" not present in dct, for NNN=000, 001, 002, ...
    """
    idx = 0
    while True:
        base = f"figure{idx:03d}.png"
        if base not in dct:
            return base
        idx += 1


def _update_manifest(dir_path: Path) -> Dict[str, str]:
    """
    Ensure the manifest file "hashes.txt" is up to date: that it has a line of the form "basename hexdigest"
    for every file in dir_path whose basename matches "figure???.png", where hexdigest is the MD5 hex digest of the
    binary contents of the file. Prunes out any lines in hashes.txt that do not correspond to files.

    Return: a dictionary from basename to hexdigest.
    """
    manifest_path = Path(dir_path / MANIFEST_FILE_BASE)
    hex_hashes = {}
    all_present = True
    if manifest_path.exists():
        for line in manifest_path.open():  # pragma: no cover
            name, hash = line.rstrip().split()
            if (Path(dir_path) / name).exists():
                hex_hashes[name] = hash
            else:
                all_present = False
    if not all_present:  # pragma: no cover
        # Rewrite the manifest file to omit any missing figure files
        with manifest_path.open("w") as f:
            for name, hash in sorted(hex_hashes.items()):
                f.write(f"{name} {hash}\n")
    # Find any figure???.png files not mentioned in the manifest, and append lines for them.
    unaccounted = set(path.name for path in dir_path.glob("figure???.png") if path.name not in hex_hashes)
    if unaccounted:
        with manifest_path.open("a") as f:
            for base in unaccounted:
                with (dir_path / base).open(mode="rb") as b:
                    contents = b.read()
                digest = hashlib.sha256(contents).hexdigest()
                f.write(f"{base} {digest}\n")
                hex_hashes[base] = digest
    return hex_hashes


def image_similarity_advice(path: Path, comparison_basenames: Iterable[str]) -> List[str]:  # pragma: no cover
    """
    Given a reference path, which should be a graphics file, and a list of basenames in the same directory
    which should also be graphics files, return a list of strings describing the relationship of
    the first file to the most similar of the comparison ones. If any of the comparison paths are
    judged identical (in content) to the reference file, only those comparisons are printed.
    """
    dir_path = path.parent
    found_identical = False
    lines = []
    for name in comparison_basenames:
        is_identical, comparison = image_similarity(path, dir_path / name)
        line = f"{name}: {comparison}"
        if is_identical == found_identical:
            lines.append(line)
        elif is_identical:
            lines = [line]
            found_identical = True
    return lines


def image_similarity(
    path1: Path, path2: Path, max_proportion_different: float = 0.0, max_sd_ratio: float = 0.0
) -> Tuple[bool, str]:
    """
    Given two paths which should be graphics files readable by PIL.Image, returns whether they are "similar"
    (defined below), and a string describing any differences.

    When the two images have the same dimensions but different content, they are not similar.
    If they have the same dimensions and are pixel-by-pixel identical, they are similar.
    Otherwise (same dimensions but different content), two statistics intended to quantify how different the images are,
    are calculated and returned.
      (1) Proportion different: the fraction of values (over all pixels and channels) that are unequal.
      (2) SD ratio: the ratio of the standard deviation of the difference between the images to the standard deviation
          of their sum.
    In the last case, the images are "similar" if neither statistic exceeds the corresponding threshold.
    Note that the statistics are fairly simple minded; two images might look indistinguishable but have high statistic
    values, e.g. if they are shifted by one pixel compared to each other. But low statistic values should mean they look
    similar.
    """
    try:
        image1 = Image.open(path1)
    except UnidentifiedImageError:  # pragma: no cover
        return False, "cannot decode first file"
    except FileNotFoundError:  # pragma: no cover
        return False, "cannot find first file"
    try:
        image2 = Image.open(path2)
    except UnidentifiedImageError:  # pragma: no cover
        return False, "cannot decode"
    except FileNotFoundError:  # pragma: no cover
        return False, "cannot find"
    if image1.size != image2.size:
        return False, f"different size: {image1.size} ({path1.name}) vs {image2.size} ({path2.name})"
    if image1.size[0] * image1.size[1] > 10000000:
        return False, "cannot compare, too big"  # pragma: no cover
    data1 = image1.getdata()
    array1 = np.asarray(data1, dtype=np.int16)
    data2 = image2.getdata()
    array2 = np.asarray(data2, dtype=np.int16)
    if np.all(array1 == array2):
        return True, "identical content"
    diff = array1 - array2  # type: ignore
    sum = array1 + array2  # type: ignore
    proportion_different = np.count_nonzero(diff) / array1.size  # type: ignore
    sd_ratio = np.std(diff) / np.std(sum)  # type: ignore
    is_similar = proportion_different <= max_proportion_different and sd_ratio <= max_sd_ratio
    return is_similar, f"difference proportion {proportion_different:8.6f}, SD ratio {sd_ratio:8.6f}"


def compare_files_in_directories(dir_names: List[str]) -> Generator[str, None, None]:
    """
    Given a list of directories, look for all files with names matching "figure???.png" under each one.
    Return the results of comparing all pairs of such files in the same subdirectory; they are plots
    that are intended to look the same but may have been generated under different conditions (OS's,
    library versions etc).
    """
    for dir_name in dir_names:
        dir_path = Path(dir_name)
        if not dir_path.exists():  # pragma: no cover
            yield ""
            yield f"Directory not found: {dir_path}"
            continue
        figure_file_dct = collections.defaultdict(list)
        for path in sorted(dir_path.rglob("figure???.png")):
            figure_file_dct[path.parent].append(path)
        if not figure_file_dct:  # pragma: no cover
            yield ""
            yield f"No figure files found under {dir_path}"
            continue
        for figdir_path in sorted(figure_file_dct):
            figure_files = figure_file_dct[figdir_path]
            if len(figure_files) < 2:
                continue  # pragma: no cover
            yield ""
            yield f"In directory {figdir_path}:"
            for i, file1 in enumerate(figure_files):
                for file2 in figure_files[i + 1 :]:  # noqa: E203
                    comparison = image_similarity(file1, file2)[1]
                    yield f"  {file1.name} vs {file2.name}: {comparison}"


if __name__ == "__main__":  # pragma: no cover
    for line in compare_files_in_directories(sys.argv[1:] or ["."]):
        print(line)
        sys.stdout.flush()
