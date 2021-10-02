# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import inspect
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, List, Optional

from psbutils.type_annotations import PathOrString


def check_is_any_of(message: str, actual: Optional[str], valid: Iterable[Optional[str]]) -> None:
    """
    Raises an exception if 'actual' is not any of the given valid values.
    :param message: The prefix for the error message.
    :param actual: The actual value.
    :param valid: The set of valid strings that 'actual' is allowed to take on.
    :return:
    """
    if actual not in valid:
        all_valid = ", ".join(["<None>" if v is None else v for v in valid])
        raise ValueError("{} must be one of [{}], but got: {}".format(message, all_valid, actual))


def delete_and_remake_directory(folder: PathOrString) -> None:
    """
    Delete the folder if it exists, and remakes it.
    """
    folder = str(folder)

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)


def is_windows() -> bool:  # pragma: no cover
    """
    Returns True if the host operating system is Windows.
    """
    return os.name == "nt"


def is_linux() -> bool:  # pragma: no cover
    """
    Returns True if the host operating system is a flavour of Linux.
    """
    return os.name == "posix"


def remove_file_or_directory(pth: Path) -> None:
    """
    Remove a directory and its contents, or a file.
    """
    if pth.is_dir():
        for child in pth.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                remove_file_or_directory(child)
        pth.rmdir()
    elif pth.exists():
        pth.unlink()


def flatten_list(lst: List[List[Any]]) -> List[Any]:  # pragma: no cover
    return [item for sublist in lst for item in sublist]


def find_root_directory() -> Path:  # pragma: no cover
    path = Path.cwd()
    while True:
        if (path / "libraries").is_dir() and (path / "projects").is_dir():
            return path.resolve()
        if path.parent == path:
            raise FileNotFoundError(f"Cannot find root directory above {Path(__file__).parent}")
        path = path.parent


ROOT_DIR = find_root_directory()


def find_subrepo_directory(path_name: Optional[str] = None) -> Path:
    """
    Returns the first absolute path at or above "path_name" whose parent's basename is "libraries" or "projects".
    If path_name is missing or None, the caller's __file__ is used.
    """
    if path_name is None:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        assert module is not None
        path_name = module.__file__
    parent = Path(path_name).absolute()
    while True:
        if parent.parent.name in ["libraries", "projects"]:
            return parent
        if parent.parent == parent:
            raise FileNotFoundError(f"Cannot find subrepo directory above {path_name}")  # pragma: no cover
        parent = parent.parent
