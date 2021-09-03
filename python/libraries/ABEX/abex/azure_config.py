# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
""" This module is intended to separate arguments for Azure ML from other command line arguments, and store
them in an AzureConfig object """
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import yaml
from abex.common.generic_parsing import GenericConfig
from param import Boolean, String


class AzureConfig(GenericConfig):
    """OptimizerConfig to parse arguments related to Azure workspace."""

    subscription_id: str = String(None, doc="The subscription to use for Azure ML jobs")  # type: ignore
    blob_storage: str = String(None, doc="The name of the blob container to download from")  # type: ignore
    workspace_name: str = String(None, doc="The name of the AzureML workspace that should be used.")  # type: ignore
    resource_group: str = String(
        None, doc="The Azure resource group that contains the Azure ML workspace."
    )  # type: ignore
    aml_experiment: str = String(None, doc="The Experiment name to submit Azure ML runs to.")  # type: ignore
    submit_to_aml: bool = Boolean(False, doc="If True, will submit experiment to Azure ML")  # type: ignore
    compute_target: str = String(None, doc="The compute cluster to submit to")  # type: ignore

    def __init__(self, **params: Any) -> None:
        allowed = self.param.params()
        to_keep = dict(pair for pair in params.items() if pair[0] in allowed)
        super().__init__(**to_keep)


class ParserResult:
    """Class to store dictionary of arguments returned from parsing."""

    def __init__(self, args: Dict[str, Any]):
        self.args = args


def read_variables_from_yaml(yaml_file: Path) -> Dict[str, Any]:
    """Read all variables from given yaml file and return as a dictionary."""
    if not yaml_file.is_file():
        raise FileNotFoundError(f"No file found at: {yaml_file}")  # pragma: no cover
    yaml_contents = yaml.safe_load(open(str(yaml_file), "r"))
    v = "variables"
    if v in yaml_contents:
        return cast(Dict[str, Any], yaml_contents[v])
    else:
        raise KeyError("The Yaml file was expected to contain a section '{}'".format(v))  # pragma: no cover


def parse_results_and_add_azure_config_vars(parser: GenericConfig, args: Optional[List[str]] = None) -> ParserResult:
    """Combine args from argument parser with those stored in default 'azureml' args file and returns the parsed
    arguments.
    """
    known_args = vars(parser.parse_args(args))
    # We don't want to search upwards from Path(__file__), because the current file may be part of an installed
    # library that does not include azureml-args.yml.
    yaml_filepath = find_file_upwards(Path("."), "azureml-args.yml")
    additional_args = read_variables_from_yaml(yaml_filepath)
    total_args = {**known_args, **additional_args}
    return ParserResult(total_args)


def find_file_upwards(dirname: Path, basename: str) -> Path:
    """
    Returns the path of the first file with basename "basename", starting in "dirname" and looking upwards
    from there.
    """
    current = dirname.absolute()
    while True:
        file = current / basename
        if file.exists():
            return file
        if current.parent == current:
            raise FileNotFoundError(f"Cannot find {basename} in or above {dirname.absolute()}")  # pragma: no cover
        current = current.parent


def copy_relevant_script_args(args: GenericConfig) -> List[str]:
    """
    Copy the relevant arguments to be passed to the script in AML.
    Skips over the 'submit_to_aml' argument, and any True values arising from the initial argument
    parsing in the main method
    """
    script_args = []
    defaults = args.param.defaults()
    for (key, val) in sorted(args.param.get_param_values(onlychanged=True)):
        # If the key is submit_to_aml, we don't want it because we will do the submitting here and don't
        # want it repeated inside the AML job. If the key is not in "defaults", it is not a GenericConfig
        # one but an AzureConfig one, and we will likewise deal with it here during submission
        if key == "submit_to_aml" or key not in defaults:
            continue  # pragma: no cover
        else:
            # We want the string form of the value if it's an Enum
            val = val.value if isinstance(val, Enum) else val
            if defaults[key] is False and val is True:
                # This looks like a flag, so we don't want to add a value. GenericConfig.add_args will deal
                # with such switches correctly.
                script_args += [f"--{key}"]
            else:
                # Otherwise we do want the value as well.
                script_args += [f"--{key}", val]
    return script_args
