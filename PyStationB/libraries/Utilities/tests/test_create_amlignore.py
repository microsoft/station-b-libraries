# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from psbutils.create_amlignore import amlignore_lines


def test_amlignore_lines():
    lines = amlignore_lines(__file__)
    assert ".git" in lines
    assert "libraries/Utilities" not in lines
    assert any(map(lambda line: line.startswith("libraries/"), lines))
    assert any(map(lambda line: line.startswith("projects/"), lines))
