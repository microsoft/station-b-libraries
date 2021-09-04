# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Generates initial design for a Bayesian Optimization experiment.

Usage:
    python abex/scripts/initial_design.py spec_file.yml --output file.csv --seed 10 --design_type RANDOM
"""

import argparse
import pathlib
from typing import List

import numpy as np
import pandas as pd

# ABEX
from abex.settings import OptimizerConfig, load
from abex.space_designs import DesignType, suggest_samples
from emukit.core import ContinuousParameter, ParameterSpace


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initial designs for Bayesian Optimization.")
    parser.add_argument("yaml", type=str, help="Name of yaml spec file.")
    parser.add_argument("--min_seed", type=int, default=42, help="Minimum random seed (default: 42).", required=False)
    parser.add_argument("--max_seed", type=int, default=42, help="Maximum random seed (default: 42).", required=False)
    parser.add_argument("--output", type=str, help="Where the output CSV should be stored.", required=True)
    parser.add_argument(
        "--design_type",
        type=DesignType,
        choices=DesignType,
        required=True,
        help="Design type to use to generate the initial batch.",
    )
    return parser


def main():  # pragma: no cover
    args = create_parser().parse_args()

    path = pathlib.Path(args.yaml)
    config: OptimizerConfig = load(path, seed=args.base_seed)  # type: ignore # auto

    # Generate parameter space and record which parameters are in the log space
    rough_parameter_space: List[ContinuousParameter] = []
    is_parameter_log: List[bool] = []

    for name, parameter_config in config.data.inputs.items():
        bounds = np.array([parameter_config.lower_bound, parameter_config.upper_bound])

        # Check if we need to pass to the log space
        if parameter_config.log_transform:
            is_parameter_log.append(True)
            bounds = np.log10(bounds + parameter_config.offset)
        else:
            is_parameter_log.append(False)

        parameter = ContinuousParameter(name, bounds[0], bounds[1])
        rough_parameter_space.append(parameter)

    parameter_space = ParameterSpace(rough_parameter_space)

    # Fix the seed and generate samples
    np.random.seed(config.seed)
    samples = suggest_samples(
        parameter_space=parameter_space, design_type=args.design_type, point_count=config.bayesopt.batch
    )

    # Move to the standard space, if needed
    rough_dataframe: dict = {}
    for i, (is_log, name) in enumerate(zip(is_parameter_log, parameter_space.parameter_names)):
        x = samples[:, i]
        if is_log:
            x = 10 ** x
        rough_dataframe[name] = x

    # Export the output
    pd.DataFrame(rough_dataframe).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()  # pragma: no cover
