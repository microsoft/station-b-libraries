# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from cellsig_sim.scripts.run_cell_signalling_loop import main
from cellsig_sim.optconfig import CellSignallingOptimizerConfig


def test_trivial():
    """
    Trivial test, just to load the files. TODO: test quickly but properly.
    """
    CellSignallingOptimizerConfig()
    assert main is not None
