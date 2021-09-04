# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Convergence plotting script for comparing multiple experiments that allows for aggregating over multiple randomised
sub-runs (runs with same configurations, but a different random seed).

This plotting script plots the *distribution* of the objective (as inferred using the simulator) for the
optimum output by the optimization algorithm at each iteration. This should be a better indication
of the quality of the algorithm than just plotting the best sample observed so far (which could have been
a lucky sample in a noisy region of the simulator).

The plotting command compares multiple experiments. It expects the following hierarchy of result directories for *each*
experiment:

experiment_results_dir/
├── run_seed00/
│   ├── iter1
|   |   └── optima.csv
|   |   ...
│   └── iter10
|       └── optima.csv
├── run_seed01/
│   ├── iter1
|   |   └── optima.csv
|   |   ...
│   └── iter10
|       └── optima.csv
|  ...
└── config.yml

Each optima.csv file should contain at least one row of data, with column names corresponding to simulator inputs.
(more column names can be present, but at least those corresponding to simulator inputs must be given)


Example command:
python plot_convergence_multiple_runs.py --experiment_dirs /path/to/experiment-res-dir/one
/path/to/experiment-res-dir/two --experiment_labels "Experiment 1" "Experiment 2" --output_path "plot.png"

"""
from typing import List, Optional

from abex.simulations.plot_predicted_optimum_convergence import plot_predicted_optimum_covergence
from cellsig_sim.optconfig import CellSignallingOptimizerConfig


# OptimizerConfig fields specific to CellSignalling simulator settings
# TODO : For maintainability, this should be inferred from config class in looping.py


def main(arg_list: Optional[List[str]] = None) -> None:  # pragma: no cover
    plot_predicted_optimum_covergence(arg_list, CellSignallingOptimizerConfig)


if __name__ == "__main__":  # pragma: no cover
    main()
