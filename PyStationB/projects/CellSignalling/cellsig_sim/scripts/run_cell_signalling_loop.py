# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import List, Optional
from azureml.core import Run
import matplotlib
from cellsig_sim.optconfig import CellSignallingOptimizerConfig

matplotlib.use("Agg")
from psbutils.psblogging import logging_to_stdout  # noqa: E402
from abex.simulations.submission import submit_aml_runs_or_do_work  # noqa: E402


def main(arg_list: Optional[List[str]] = None) -> List[Run]:  # pragma: no cover
    """Returns: list of AzureML Runs, if any were submitted"""
    logging_to_stdout()
    return submit_aml_runs_or_do_work(arg_list, CellSignallingOptimizerConfig)


if __name__ == "__main__":  # pragma: no cover
    main()
