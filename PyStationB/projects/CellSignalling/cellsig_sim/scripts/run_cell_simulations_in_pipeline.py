# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Wrapper script which runs simulator specifically for the Azure ML pipeline specified in run_simulator_pipeline.
Whilst running this code locally will work, it is recommended to invoke run_cell_signalling_loop.py
instead for non-"AML pipeline" runs, for more control over multiprocessing, number of runs to instantiate etc.
"""
import os
import sys
from typing import List, Optional


try:
    from abex.simulations.run_simulator_pipeline import run_simulations_in_pipeline
    from cellsig_sim.optconfig import CellSignallingOptimizerConfig
except ImportError as e:  # pragma: no cover
    # Print diagnostic information if there were problems with imports, probably inside AzureML.
    print("ImportError raised...")
    current = os.getcwd()
    print(f"Current directory: {current}")
    print("Contents of current directory:")
    for item in sorted(os.listdir(current)):
        print(f"  {item}")
    print(f"PYTHONPATH: {os.getenv('PYTHONPATH')}")
    print("sys.path:")
    for component in sys.path:
        existence = "EXISTS" if os.path.exists(component) else "DOES NOT EXIST"
        print(f"  {component} - {existence}")
    raise e


def main(arg_list: Optional[List[str]] = None):  # pragma: no cover
    """
    Wrapper script which runs simulator specifically for the Azure ML pipeline specified in run_simulator_pipeline.
    Args:
        arg_list:

    Returns:

    """
    run_simulations_in_pipeline(arg_list, CellSignallingOptimizerConfig)


if __name__ == "__main__":  # pragma: no cover
    main(arg_list=sys.argv[1:])
