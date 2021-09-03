# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from abex.scripts.initial_design import create_parser


def test_create_parser():
    # Just to get test coverage:
    p = create_parser()
    assert p is not None
