# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from cellsig_sim.scripts.run_cell_simulations_in_pipeline import main as main1
from cellsig_sim.scripts.plot_cellsig_predicted_optimum_convergence import main as main2
from cellsig_sim.scripts.run_cell_simulator_pipeline import main as main3


# Stub tests to provide coverage. The bulk of the code is tested in the ABEX subrepo.
def test_load_wrapper_scripts():
    assert main1 is not None
    assert main2 is not None
    assert main3 is not None
