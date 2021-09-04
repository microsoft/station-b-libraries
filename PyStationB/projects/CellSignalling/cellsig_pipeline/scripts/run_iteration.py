# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Runs the iteration of the experiment. Namely,
  - downloads the data from the last iteration from pyBCKG,
  - runs ABEX on them,
  - merges ABEX suggestions together, to a DoE which is ready to be uploaded into Antha

Usage:
    python scripts/run_iteration.py --iteration N --names Experiment1 Experiment2 ...

Arguments:
    iteration: starts as 1. It is the time you are applying this script to a particular experimental track.
    names: names of the experiments (as they are listed in the Experiments/ directory) which will be processed
    output (optional): the name of the XLSX file with DoE. If not set, it defaults to "{current date}.xlsx"
"""
import argparse
import datetime
import pathlib
import time
from typing import List, Optional, Set

import cellsig_pipeline
import cellsig_pipeline.structure as ps
import pandas as pd
import pydantic
from psbutils.psblogging import logging_to_stdout


class Arguments(pydantic.BaseModel):
    iteration: pydantic.PositiveInt
    experiment_names: Set[str]
    skip_data_retrieval: bool
    skip_doe: bool
    output: Optional[str] = None

    @pydantic.validator("output")
    def validate_output(cls, v):  # pragma: no cover
        """Validates that output ends has .xlsx extension. If it is None, sets it to the current date."""
        if v is None:  # Set it to today's date, if not defined.
            return "{}-doe.xlsx".format(datetime.date.today())
        elif pathlib.Path(v).suffix != ".xlsx":  # Check if this is an XLSX file.
            raise pydantic.ValidationError("Output must be an XLSX file.")  # type: ignore # auto
        else:  # The name is valid.
            return v


def parse_args(arg_list: Optional[List[str]]) -> Arguments:
    """Returns the names of the experiments to be created."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, help="The iteration number. (Starts at 1).", required=True)
    parser.add_argument("-n", "--names", nargs="+", help="Names of the experiments to be run.", required=True)
    parser.add_argument("--output", type=str, help="XLSX file to which the Antha DoE will be dumped.", default=None)
    parser.add_argument("--skip_doe", action="store_true", help="Skip Antha DOE generation")
    parser.add_argument(
        "--skip_data_retrieval",
        action="store_true",
        help="If the data has been retrieved before, set this flag to skip the download.",
    )

    args = parser.parse_args(arg_list)

    return Arguments(
        iteration=args.iteration,
        experiment_names=args.names,
        skip_data_retrieval=args.skip_data_retrieval,
        skip_doe=args.skip_doe,
        output=args.output,
    )


def get_batch(experiment_name: str, iteration: int, skip_data_retrieval: bool) -> pd.DataFrame:
    """Gets a new batch for one experimental track.

    Args:
        experiment_name: experimental track name
        iteration: iteration number, should start at 1
        skip_data_retrieval: if True, we don't download the data from BCKG (and assume the post-combinatorics data
            are ready to be used by ABEX)

    Returns:
        a DoE for this experimental track
    """
    if iteration < 1:
        raise ValueError("Iteration number must be at least 1.")  # pragma: no cover

    if not ps.experiment_dir(experiment_name).is_dir():
        raise NameError(f"Experiment {experiment_name} needs to be initialized.")  # pragma: no cover

    # Retrieve the data
    if not skip_data_retrieval:
        cellsig_pipeline.data.prepare_data_for_abex(experiment_name, iteration)  # pragma: no cover
    # Give the OS time to save them to the disk
    time.sleep(0.5)

    new_batch: pd.DataFrame = cellsig_pipeline.steps.run_abex(experiment_name, iteration)
    return new_batch


def validate_doe(doe: pd.DataFrame) -> None:  # pragma: no cover
    """Check if a DoE is valid."""
    if len(doe) != 60:  # TODO: This hard-coded number can be saved in a configuration file.
        cellsig_pipeline.logger.warning(f"The DoE has a wrong number of wells: {len(doe)} != 60.")


def main(arg_list: Optional[List[str]] = None) -> None:
    logging_to_stdout()
    arguments = parse_args(arg_list)
    cellsig_pipeline.logger.info(f"Asked to propose a new experiment with arguments {arguments}.")

    # Get a DoE for each experimental track
    dataframes: List[pd.DataFrame] = []
    for i, name in enumerate(arguments.experiment_names, 1):
        cellsig_pipeline.logger.info(f"Processing experiment {name} ({i}/{len(arguments.experiment_names)}).")

        new_batch: pd.DataFrame = get_batch(name, arguments.iteration, arguments.skip_data_retrieval)
        if not arguments.skip_doe:  # pragma: no cover
            doe: pd.DataFrame = cellsig_pipeline.steps.abex_to_doe(new_batch, name)
            dataframes.append(doe)

    if arguments.skip_doe:
        cellsig_pipeline.logger.info("Completed suggestions of new batches.")
    else:  # pragma: no cover
        cellsig_pipeline.logger.info("Completed suggestions of new batches. Merging them into one Antha DoE...")

        # Merge all the DoEs together into one DoE.
        doe = pd.concat(dataframes)
        validate_doe(doe)
        doe.to_excel(arguments.output, index=False)

        cellsig_pipeline.logger.info(f"The DoE {arguments.output} is ready.")


if __name__ == "__main__":  # pragma: no cover
    main()
