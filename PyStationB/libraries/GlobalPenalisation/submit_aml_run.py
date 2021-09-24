# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Submits a benchmarking run to Azure ML.

Usage:
    python submit_aml_run.py EXPERIMENT_NAME [ARGUMENTS TO BE PASSED to run_benchmark.py]

Example:
    python submit_aml_run.py TestForrester-Experiment forrester

will execute `run_benchmark.py` on the forrester objective in the Azure ML cloud.
"""
import dataclasses
import sys
from typing import Sequence, Union

from azureml import core

SOURCE_DIRECTORY = "."


@dataclasses.dataclass
class AMLSettings:
    """Settings used to connect to Azure ML."""

    subscription_id: str = ""
    resource_group: str = ""
    workspace_name: str = ""
    compute_target: str = ""


def _create_workspace(aml_settings: AMLSettings) -> core.Workspace:
    """Retrieves AML Workspace from given settings."""
    return core.Workspace(
        subscription_id=aml_settings.subscription_id,
        resource_group=aml_settings.resource_group,
        workspace_name=aml_settings.workspace_name,
    )


def _create_environment() -> core.Environment:
    env = core.Environment.from_conda_specification(name="GlobalPenalisationEnvironment", file_path="environment.yml")
    env.environment_variables.update({"PYTHONPATH": SOURCE_DIRECTORY})
    return env


def submit_run(
    aml_settings: AMLSettings,
    args_to_pass: Sequence[Union[str, int, float]],
    experiment_name: str,
) -> None:
    # Define workspace and experiment
    workspace = _create_workspace(aml_settings)
    exp = core.Experiment(workspace, experiment_name)

    # Define the run config
    environment = _create_environment()
    config = core.ScriptRunConfig(
        source_directory=SOURCE_DIRECTORY,
        script="run_benchmark.py",
        compute_target=aml_settings.compute_target,
        environment=environment,
        arguments=args_to_pass,
    )

    # Submit the run
    run = exp.submit(config)
    print(run)


def main() -> None:
    aml_settings = AMLSettings()
    # TODO: Consider passing the arguments in a more principled way.
    submit_run(aml_settings, args_to_pass=sys.argv[2:], experiment_name=sys.argv[1])


if __name__ == "__main__":
    main()
