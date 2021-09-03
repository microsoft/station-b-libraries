# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import List, Optional

import pytest
from abex.azure_config import AzureConfig
from abex.scripts.run import RunConfig
from abex.settings import OptimizerConfig
from abex.simulations.submission import submit_aml_runs_or_do_work
from azureml.core.run import Run
from psbutils.misc import find_subrepo_directory


@pytest.mark.timeout(60)
def test_submit_aml_runs_or_do_work():
    def submitter(_: AzureConfig, run_config: RunConfig, parent_run: Optional[Run] = None):
        return run_config

    yml_path = find_subrepo_directory() / "tests/data/Specs/run_loop_test_config.yml"
    arg_str = f"--spec_file {yml_path} --num_iter 1 --num_runs 10 " "--plot_simulated_slices --submit_to_aml"
    arg_list = arg_str.split()
    result: List[RunConfig] = submit_aml_runs_or_do_work(arg_list, OptimizerConfig, submitter)
    assert len(result) == 2
    assert result[0].num_runs == 10
    arg_list += ["--num_runs_per_aml_run", "0"]
    result = submit_aml_runs_or_do_work(arg_list, OptimizerConfig, submitter)
    assert len(result) == 2
    assert result[0].num_runs == 10
    assert result[1].num_runs == 10
    arg_list[-1] = "4"
    result = submit_aml_runs_or_do_work(arg_list, OptimizerConfig, submitter)
    assert len(result) == 6
    assert set(cfg.spec_file for cfg in result[:3]) == {
        "tmp_spec.strat1_0.json",
        "tmp_spec.strat1_4.json",
        "tmp_spec.strat1_8.json",
    }
    assert set(cfg.spec_file for cfg in result[3:]) == {
        "tmp_spec.strat2_0.json",
        "tmp_spec.strat2_4.json",
        "tmp_spec.strat2_8.json",
    }
    assert [cfg.base_seed for cfg in result] == [0, 4, 8, 0, 4, 8]
    assert [cfg.num_runs for cfg in result] == [4, 4, 2, 4, 4, 2]
