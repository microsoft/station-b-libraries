# TODO: Move this code (and dependencies) into abex library
# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Creates and runs an Azure ML pipeline to run an entire workflow of 1) running the simulator (with 1 node per config
expansion), 2) plotting the results. Results can be viewed in the AML portal, or directly in the Datastore
named 'workspaceblobstore' which will be created in the Workspace specified in azureml-args.yml
"""
from pathlib import Path
from typing import List, Optional

from abex.simulations.run_simulator_pipeline import run_simulator_pipeline
from azureml.core import Run
from cellsig_sim.optconfig import CellSignallingOptimizerConfig

CELLSIG_SCRIPT_DIR = Path(__file__).parent


def main(arg_list: Optional[List[str]] = None) -> Optional[Run]:  # pragma: no cover
    """
    Submit an Azure pipeline experiment consisting of running the simulator (involving multiple runs if
     multiple Spec expansions), and plotting the results
    Args:
        arg_list:

    Returns:

    """
    return run_simulator_pipeline(
        arg_list,
        CELLSIG_SCRIPT_DIR / "run_cell_simulations_in_pipeline.py",
        CELLSIG_SCRIPT_DIR / "plot_cellsig_predicted_optimum_convergence.py",
        CellSignallingOptimizerConfig,
    )


if __name__ == "__main__":
    main()  # pragma: no cover
