# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path
from tempfile import NamedTemporaryFile

from scripts.dos2unix import dos2unix


def test_dos2unix():
    temp = NamedTemporaryFile(delete=False)
    temp.close()
    name = temp.name
    with open(name, "w") as gh:
        gh.write("Hello\n")
    dos2unix([name])
    with open(name, "rb") as fh:
        data = fh.read()
    Path(name).unlink()
    assert data == b"Hello\n"
    assert len(data) == 6
