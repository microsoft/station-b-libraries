# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import os
from pathlib import Path
import shutil
import pandas as pd
import pytest

from abex.settings import load
from cellsig_pipeline.scripts import run_iteration
from cellsig_pipeline.structure import ABEX_CONFIG


@pytest.mark.timeout(450)
def test_hmc_local_penalization():
    """
    Check that with HMC, suggested values in the batch are sufficiently diverse. Typical values of std() for
    each input with and without calling update_batch on each newly-minted Acquisition object
    (in IntegratedAcquisition.generate_acquisition_with_batch_update) are:

    Input    Without    With
    ATC         17       38
    Arabinose   0.6       1
    Con         10       20
    """
    proj_dir = Path(__file__).parent.parent.parent
    os.chdir(str(proj_dir))
    expt_name = "hmc_local_penalization"
    src_data_dir = Path("tests/data/Data") / expt_name
    expt_yml_base = expt_name + ".yml"
    src_config_file = Path("tests/data/Specs") / expt_yml_base
    config = load(str(src_config_file))
    tgt_data_folder = config.data.folder
    tgt_data_dir = tgt_data_folder.parent.parent
    if tgt_data_dir.exists():
        shutil.rmtree(tgt_data_dir)
    shutil.copytree(src_data_dir, tgt_data_folder)
    tgt_config_file = tgt_data_dir / ABEX_CONFIG
    shutil.copy(src_config_file, tgt_config_file)
    run_iteration.main(["-n", expt_name, "--iteration", "1", "--skip_data_retrieval", "--skip_doe"])
    batch_csv_path = tgt_data_dir / "Results" / "iteration-01" / "batch.csv"
    assert batch_csv_path.exists()
    batch_df = pd.read_csv(batch_csv_path)
    assert len(batch_df) == config.bayesopt.batch  # type: ignore
    assert batch_df["ATC"].std() >= 27  # type: ignore
    assert batch_df["Arabinose"].std() >= 0.85  # type: ignore
    assert batch_df["Con"].std() >= 15  # type: ignore
    shutil.rmtree(tgt_data_dir)
