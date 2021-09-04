# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path

import yaml
from abex.simulations.submission import create_temporary_azureml_environment_file


def test_create_azureml_environment_file():
    name = create_temporary_azureml_environment_file()
    path = Path(name)
    assert path.exists()
    with path.open() as fh:
        data = yaml.safe_load(fh)
    path.unlink()
    assert "name" in data
    assert "dependencies" in data
    pip_deps = None
    for dct in data["dependencies"]:
        if isinstance(dct, dict) and "pip" in dct:
            pip_deps = dct["pip"]
            break
    assert pip_deps is not None
