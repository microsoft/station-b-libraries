# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import sys
from pathlib import Path
from typing import List


def dos2unix(files: List[str], encoding: str = "utf-8") -> None:
    """
    Reads every file in the list as utf-8 text, and writes it back using just the newline character
    as line separator.
    """
    for file in files:
        with Path(file).open(encoding=encoding) as fh:
            lines = fh.readlines()
        with Path(file).open("wb") as gh:
            for line in lines:
                gh.write(line.rstrip("\r\n").encode(encoding))
                gh.write("\n".encode(encoding))


if __name__ == "__main__":
    dos2unix(sys.argv[1:])  # pragma: no cover
