# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import copy
import json
import logging
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import yaml
from abex.azure_config import AzureConfig, copy_relevant_script_args, parse_results_and_add_azure_config_vars
from abex.constants import AZUREML_ENV_NAME
from abex.scripts.run import RunConfig
from abex.settings import CustomDumper, OptimizerConfig, load_config_from_path_or_name
from abex.simulations.looping import load_resolutions_from_command, run_multiple_seeds
from azureml.core import ComputeTarget, Environment, Experiment, Run, ScriptRunConfig, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.run import _OfflineRun
from psbutils.misc import ROOT_DIR


def create_temporary_azureml_environment_file() -> str:
    """
    Returns the name of a temporary file in which has been placed a self-contained Conda environment specification
    as required by AzureML. The contents of the file are those of the top-level environment.yml file, with the
    "-r" reference to requirements_dev.txt replaced by a list of the packages in requirements.txt (not the wider
    requirements_dev.txt, as we don't need development and testing packages in an AzureML job).
    """
    temp = NamedTemporaryFile(delete=False)
    temp.close()
    with (ROOT_DIR / "environment.yml").open() as fh:
        struc = yaml.safe_load(fh)
    target_dep = locate_or_create_pip_section_target(struc["dependencies"])
    with (ROOT_DIR / "requirements.txt").open() as fh:
        target_dep["pip"] = [line.strip() for line in fh]
    with Path(temp.name).open("w") as gh:
        yaml.dump(struc, gh, Dumper=CustomDumper)
    return temp.name


def locate_or_create_pip_section_target(deps: List[Any]) -> Dict[str, Any]:
    """
    :param deps: the "dependencies" section of the Conda YAML data
    :return: the (first) member of deps that is a dictionary; if there is none, one is appended.
    """
    target_dep = None
    for dep in deps:
        if isinstance(dep, dict):
            target_dep = dep
            break
    if target_dep is None:  # pragma: no cover
        target_dep = {}
        deps.append(target_dep)
    return target_dep


def get_pythonpath_relative_to_root_dir():  # pragma: no cover
    """
    Returns a (Unix) PYTHONPATH string consisting of all the projects and libraries under ROOT_DIR that
    have a setup.py file, relative to ROOT_DIR.
    """
    dirs = [str(spy.parent.relative_to(ROOT_DIR)) for spy in ROOT_DIR.glob("[pl]*s/*/setup.py")]
    return ":".join(["."] + dirs)


def submit_to_aml(
    azure_config: AzureConfig, args: RunConfig, parent_run: Optional[Run] = None
) -> Run:  # pragma: no cover
    """
    Submit the script given on the command line to Azure ML.
    :param azure_config: config containing the args we need to submit this
    :param args: the command line args that this script was initially called with
    :return: the submitted run.
    """
    ws = Workspace(azure_config.subscription_id, azure_config.resource_group, azure_config.workspace_name)
    script_path = Path(sys.argv[0])

    # copy all arguments except submit_to_aml
    script_args = copy_relevant_script_args(args)

    compute_target = ComputeTarget(ws, azure_config.compute_target)

    # set up the Python environment
    env = Environment(AZUREML_ENV_NAME)
    aml_env_file = create_temporary_azureml_environment_file()
    conda_deps = CondaDependencies(conda_dependencies_file_path=aml_env_file)
    env.python.conda_dependencies = conda_deps
    env.environment_variables = {"PYTHONPATH": get_pythonpath_relative_to_root_dir()}
    config = ScriptRunConfig(
        source_directory=ROOT_DIR,
        script=script_path.absolute().relative_to(ROOT_DIR),
        arguments=script_args,
        compute_target=compute_target,
        environment=env,
    )
    experiment = Experiment(ws, args.aml_experiment or azure_config.aml_experiment)
    if parent_run is not None:
        run = parent_run.submit_child(config)
    else:
        try:
            run = experiment.submit(config)
        except KeyError as e:
            logging.error(
                f"FAILED to submit run to experiment {experiment.name}; do you have Contributor-level access?"
            )
            raise e
        logging.info(f"Submitted run {experiment.name}:{run.number}")
        logging.info(f"Run URL is: {run.get_portal_url()}")
        Path(aml_env_file).unlink()
    return run


def spec_file_basename(res_spec: str, low_seed: int, suffix: str = "json") -> str:  # pragma: no cover
    """
    Returns the name of a temporary json file, unique per value of res_spec.
    """
    if res_spec:
        return f"tmp_spec.{res_spec}_{low_seed}.{suffix}"
    return f"tmp_spec_{low_seed}.{suffix}"


def launch_parent_run(azure_config: AzureConfig, args: RunConfig) -> Run:  # pragma: no cover
    """
    Create a Run that does nothing, other than act as a parent run to subsequent child runs
    :param azure_config: config containing the args we need to submit this
    :param args: the command line args that this script was initially called with
    :return: the submitted run
    """
    ws = Workspace(azure_config.subscription_id, azure_config.resource_group, azure_config.workspace_name)

    compute_target = ComputeTarget(ws, azure_config.compute_target)

    # set up the Python environment
    env = Environment(AZUREML_ENV_NAME)
    aml_env_file = create_temporary_azureml_environment_file()
    conda_deps = CondaDependencies(conda_dependencies_file_path=aml_env_file)
    env.python.conda_dependencies = conda_deps
    env.environment_variables = {"PYTHONPATH": get_pythonpath_relative_to_root_dir()}

    config = ScriptRunConfig(
        command=["true"],  # a command must be provided
        source_directory=ROOT_DIR,
        compute_target=compute_target,
        environment=env,
    )
    experiment = Experiment(ws, args.aml_experiment or azure_config.aml_experiment)
    try:
        run = experiment.submit(config)
    except KeyError as e:
        logging.error(f"FAILED to submit run to experiment {experiment.name}; do you have Contributor-level access?")
        raise e

    logging.info(f"Submitted parent run {experiment.name}:{run.number}")
    logging.info(f"Parent run URL is: {run.get_portal_url()}")
    Path(aml_env_file).unlink()
    return run


def submit_aml_runs_or_do_work(
    arg_list: Optional[List[str]],
    loop_config_class: Type[OptimizerConfig],
    submitter: Optional[Callable] = None,
) -> List[Any]:
    """
    Run multiple traces with different random seeds of Bayesian Optimization or Zoom Optimization on a simulator
    for a given number of batches (iterations). They will run as different processes if --enable_multiprocessing
    is specified in arg_list, and on AzureML if --submit_to_aml is specified.

    A calling script might have a "main" function that simply calls this function, allowing command-line usage
    such as

        python calling_script.py --spec_file config.yml --num_iter 5 --num_runs 10

    where the config.yml is a config file specifying all the settings required for ABEX (see OptimizerConfig in
    abex/settings.py) and there are additional simulator-specific fields detailed in OptimizerConfig.

    This function either submits Run(s) to AzureML to call run_multiple_seeds, or does the work locally.
    If AzureML is used, there may be a single AML Run, one AML Run per resolution of the config,
    or multiple AML Runs per resolution, each for a different range of runs.

    NOTE: the word "run" without qualification refers here to an optimization run. An AzureML
    Run is always referred to explicitly. Thus in particular, "num_runs_per_aml_run" should be read
    as "number of optimization runs per Azure ML Run".
    """
    parser = RunConfig()
    args = parser.parse_args(arg_list)
    if args.num_runs_per_aml_run >= 0:
        args.submit_to_aml = True
    azure_config: Optional[AzureConfig] = None
    actual_submitter = submitter or submit_to_aml
    if args.submit_to_aml:
        parser2_result = parse_results_and_add_azure_config_vars(parser, arg_list)
        azure_config = AzureConfig(**parser2_result.args)
        if args.num_runs_per_aml_run < 0:  # pragma: no cover
            # Multiple resolutions, if any, will be handled inside a single AML run.
            _, config_dct = load_config_from_path_or_name(args.spec_file)
            create_config_json_file(config_dct, args)
            logging.info("Submitting a single AzureML run")
            return [actual_submitter(azure_config, args)]
    # Expand config
    config_pair_lists = load_resolutions_from_command(args, loop_config_class)
    runs = []
    # num_runs_per_aml_run = args.num_runs_per_aml_run

    if not isinstance(Run.get_context(), _OfflineRun) and submitter is None:  # pragma: no cover
        assert azure_config is not None  # for mypy
        parent_run = launch_parent_run(azure_config, args)
    else:
        # We don't want to launch a parent run if "submitter" has been passed, as then
        # this is probably a test.
        parent_run = None

    for pair_list in config_pair_lists:
        if not args.submit_to_aml:
            run_multiple_seeds(args, pair_list)  # pragma: no cover
        else:
            # Version of args modified for a single AML run
            subargs_list = create_args_for_submission(args, pair_list)
            assert azure_config is not None  # for mypy
            for subargs in subargs_list:
                res_spec = pair_list[0][1].resolution_spec
                low_seed = subargs.base_seed
                logging.info(
                    f"Submitting an AzureML run for resolution spec {res_spec}, "
                    f"seeds {low_seed} "
                    f"to {low_seed + subargs.num_runs - 1} inclusive"
                )
                runs.append(actual_submitter(azure_config, subargs, parent_run=parent_run))
                Path(spec_file_basename(res_spec, low_seed)).unlink(missing_ok=True)
    return runs


def create_args_for_range_of_runs(base_seed, num_runs_per_aml_run, subargs):  # pragma: no cover
    subsubargs = copy.copy(subargs)
    subsubargs.base_seed = base_seed
    subsubargs.num_runs = min([num_runs_per_aml_run, subargs.base_seed + subargs.num_runs - base_seed])
    return subsubargs


def create_args_for_submission(
    args: RunConfig, pair_list: List[Tuple[Dict[str, Any], OptimizerConfig]]
) -> List[RunConfig]:  # pragma: no cover
    nr_per_sub = args.num_runs_per_aml_run
    if nr_per_sub <= 0:
        nr_per_sub = args.num_runs
    subargs_list = []
    for idx in range(0, len(pair_list), nr_per_sub):
        config_dct, config = pair_list[idx]
        subargs = copy.copy(args)
        subargs.base_seed = config.seed or 0
        subargs.num_runs = min(nr_per_sub, args.base_seed + args.num_runs - (config.seed or 0))
        subargs.submit_to_aml = False
        subargs.num_runs_per_aml_run = 0
        create_config_json_file(config_dct, subargs)
        subargs.resolution_spec = config_dct["resolution_spec"]
        subargs_list.append(subargs)
    return subargs_list


def paths_to_posix_strings(config: Any) -> Any:
    if isinstance(config, Path):
        return config.as_posix()  # pragma: no cover
    if isinstance(config, Dict):
        return dict((k, paths_to_posix_strings(v)) for k, v in config.items())
    if isinstance(config, List):
        return [paths_to_posix_strings(item) for item in config]  # pragma: no cover
    return config


def create_config_json_file(config: Dict[str, Any], args: RunConfig) -> None:
    spec_file = spec_file_basename(config["resolution_spec"], args.base_seed)
    with open(spec_file, "w") as f:
        config = paths_to_posix_strings(config)
        json.dump(config, f)
    args.spec_file = spec_file
