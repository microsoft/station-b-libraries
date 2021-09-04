# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path
from tempfile import mkdtemp

from scripts import check_copyright_notices as ccn


def test_full_notice():
    full = ccn.construct_full_notice()
    assert len(full) == len(ccn.NOTICE) + 2
    assert all(map(lambda line: line.startswith("#"), full))


def test_add_copyright_notice():
    tmp_dir = Path(mkdtemp())
    empty = tmp_dir / "empty.py"
    with open(empty, "w"):
        pass
    with_notice = tmp_dir / "with.py"
    with open(with_notice, "w") as out:
        for line in ccn.FULL_NOTICE:
            out.write(line)
    without_notice = tmp_dir / "without.py"
    with open(without_notice, "w") as out:
        out.write("def foo():\n  pass\n")
    # Is empty:
    assert ccn.has_copyright_notice_or_is_empty(empty)
    # Has copyright notice:
    assert ccn.has_copyright_notice_or_is_empty(with_notice)
    # Non empty and no copyright notice:
    assert not ccn.has_copyright_notice_or_is_empty(without_notice)
    # Running the check causes at least one file to be modified:
    assert ccn.check_copyright_notices_on_files([empty, with_notice, without_notice])
    # It was the one that used not to have a copyright notice:
    assert ccn.has_copyright_notice_or_is_empty(without_notice)
    # Running again does not cause any more to be added.
    assert not ccn.check_copyright_notices_on_files([empty, with_notice, without_notice])
    for path in [empty, with_notice, without_notice]:
        path.unlink()
    tmp_dir.rmdir()
