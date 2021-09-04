# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path

import pytest
from abex.plotting.expected_basenames import expected_basenames_2d
from cellsig_sim.scripts.run_cell_signalling_loop import main
from psbutils.misc import find_subrepo_directory


# @pytest.mark.timeout(1800)
@pytest.mark.skip("Can cause ADO timeout on all platforms")
def test_tutorial_wetlab_simulation():
    # For real use, we would want something like --num_iter 15 --num_runs 100, but to test everything is working,
    # smaller values are sufficient, and reduce the compute time from hours to minutes.
    num_iter = 2
    num_runs = 3
    subrepo_dir = find_subrepo_directory()
    main(
        [
            "--spec_file",
            f"{subrepo_dir}/tests/data/Specs/tutorial-wetlab-sim.yml",
            "--num_iter",
            str(num_iter),
            "--num_runs",
            str(num_runs),
            "--enable_multiprocessing",
            "--plot_simulated_slices",
        ]
    )
    results_dir = Path("Results") / "tutorial-wetlab-sim"
    assert (results_dir / "config.yml").is_file()
    for i_run in range(num_runs):
        run_dir = results_dir / "fixed" / f"seed{i_run}"
        assert (run_dir / "init_batch.csv").is_file()
        for i_iter in range(1, num_iter + 1):
            iter_dir = run_dir / f"iter{i_iter}"
            assert iter_dir.is_dir()
            basenames = [f.name for f in sorted(iter_dir.iterdir())]
            assert basenames == expected_basenames_2d(4, variant=2)
