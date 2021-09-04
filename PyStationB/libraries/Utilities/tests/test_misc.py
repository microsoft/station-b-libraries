# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import tempfile
from pathlib import Path

import pytest
from psbutils.misc import check_is_any_of, delete_and_remake_directory, remove_file_or_directory


def test_is_any_of() -> None:
    """
    Tests for check_is_any_of: checks if a string is any of the strings in a valid set.
    """
    check_is_any_of("prefix", "foo", ["foo"])
    check_is_any_of("prefix", "foo", ["bar", "foo"])
    check_is_any_of("prefix", None, ["bar", "foo", None])
    # When the value is not found, an error message with the valid values should be printed
    with pytest.raises(ValueError) as ex:
        check_is_any_of("prefix", None, ["bar", "foo"])
    assert "bar" in ex.value.args[0]
    assert "foo" in ex.value.args[0]
    assert "prefix" in ex.value.args[0]
    # The error message should also work when one of the valid values is None
    with pytest.raises(ValueError) as ex:
        check_is_any_of("prefix", "baz", ["bar", None])
    assert "bar" in ex.value.args[0]
    assert "<None>" in ex.value.args[0]
    assert "prefix" in ex.value.args[0]
    assert "baz" in ex.value.args[0]


def test_delete_and_remake_directory():
    root = Path(tempfile.mkdtemp())
    path1 = root / "a"
    delete_and_remake_directory(path1)
    assert path1.is_dir()  # assert path1 exists after not previously existing
    delete_and_remake_directory(path1)
    assert path1.is_dir()  # assert path1 still exists when it did previously exist
    path2 = root / "b"
    path3 = path2 / "c"
    delete_and_remake_directory(path3)
    assert path3.is_dir()
    delete_and_remake_directory(path2)
    assert path2.is_dir()
    assert not path3.is_dir()
    # If this does not remove the contents of root, the rmdir will fail.
    delete_and_remake_directory(root)
    root.rmdir()


def test_remove_file_or_directory():
    root = Path(tempfile.mkdtemp())
    path1 = root / "a"
    path2 = path1 / "b"
    path2.mkdir(parents=True)
    path3 = path1 / "c"
    path3.touch()
    assert path2.is_dir()
    assert path3.is_file()
    remove_file_or_directory(path2)
    assert not path2.exists()
    remove_file_or_directory(path3)
    assert not path3.exists()
    path3.touch()  # ensure there is a file (as well as a directory) under root
    remove_file_or_directory(root)
    assert not root.exists()
