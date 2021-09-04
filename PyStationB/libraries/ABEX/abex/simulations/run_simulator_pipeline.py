# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import logging
import os
import sys
from pathlib import Path
from typing import Type, Tuple, List, Dict, Optional, Set

import yaml
from abex.azure_config import AzureConfig, parse_results_and_add_azure_config_vars
from abex.constants import AZUREML_ENV_NAME
from abex.dataset import Dataset
from abex.scripts.run import RunConfig
from abex.settings import OptimizerConfig, CustomDumper, load_config_from_path_or_name
from abex.simulations import SimulatedDataGenerator
from abex.simulations.looping import load_resolutions_from_command, run_multiple_seeds
from abex.simulations.submission import (
    create_temporary_azureml_environment_file,
    get_pythonpath_relative_to_root_dir,
    spec_file_basename,
    create_config_json_file,
)
from azureml.core import Workspace, ComputeTarget, Datastore, Environment, RunConfiguration, Experiment, Run
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData, StepSequence, Pipeline
from azureml.pipeline.steps import PythonScriptStep
from psbutils.create_amlignore import create_amlignore
from psbutils.misc import ROOT_DIR
from psbutils.psblogging import logging_to_stdout


class AMLResources:
    def __init__(self, azure_config: AzureConfig):
        if azure_config.compute_target in (None, "local"):
            raise ValueError(
                "AML pipelines don't run locally. Please specify a remote compute target"
            )  # pragma: no cover
        self.ws = Workspace(azure_config.subscription_id, azure_config.resource_group, azure_config.workspace_name)
        self.compute_target = ComputeTarget(self.ws, azure_config.compute_target)
        self.env = self.specify_conda_environment()
        self.datastore = Datastore(workspace=self.ws, name="workspaceblobstore")

    @staticmethod
    def specify_conda_environment():
        # set up the Python environment
        env = Environment(AZUREML_ENV_NAME)
        aml_env_file = create_temporary_azureml_environment_file()
        conda_deps = CondaDependencies(conda_dependencies_file_path=aml_env_file)
        env.python.conda_dependencies = conda_deps
        env.environment_variables = {"PYTHONPATH": get_pythonpath_relative_to_root_dir()}
        return env


def specify_run_step(
    args: RunConfig,
    aml_resources: AMLResources,
    run_script_path: Path,
    loop_config_class: Type[OptimizerConfig],
    check_consistency: bool = True,
) -> Tuple[List[PythonScriptStep], List[PipelineData], Dict[str, List[str]], List[str]]:
    """
    Create the pipeline step(s) to run the simulation.
    Args:
        aml_resources: an instance of AMLResources which contains the necessary information on
          AML resources to instantiate pipeline steps
        run_script_path: script that the run step should invoke
        loop_config_class: (subclass of) OptimizerConfig that should be instantiated
        check_consistency: whether to run data_and_simulation_are_consistent; normally we do, but
           this may be set to False for tests that check other parts of this functionality.

    Returns: A list of PythonScriptSteps, with one for each expansion, a list of output data locations in AML,
    a dictionary of styled subsets for plotting, and a list of the temporary spec files that have been created

    """

    # Expand config
    selections_and_configs = list(load_resolutions_from_command(args, loop_config_class))

    temp_spec_files = []
    parallel_steps = []
    all_run_outputs = []
    styled_subsets: Dict[str, List[str]] = {}

    # For each expansion, create a PythonScriptStep to run the simulator script.
    num_selections = len(selections_and_configs)
    for index, pair_list in enumerate(selections_and_configs, 1):
        config0 = pair_list[0][1]
        if (not check_consistency) or data_and_simulation_are_consistent(config0):
            logging.info(
                f"Config resolution {index} of {num_selections} will have {len(pair_list)} runs included in pipeline"
            )
        else:  # pragma: no cover
            logging.error(f"Dropping config resolution {index} of {num_selections} from pipeline")
            continue
        for config_dct, config in pair_list:
            batch_strategy = config_dct["bayesopt"]["batch_strategy"]
            acquisition = config_dct["bayesopt"]["acquisition"]
            experiment_label = f"{batch_strategy} - {acquisition}"

            # TODO: what about acquisition, optimization_strategy?
            if batch_strategy not in styled_subsets:
                styled_subsets[batch_strategy] = [experiment_label]
            else:
                styled_subsets[batch_strategy].append(experiment_label)  # pragma: no cover

            # set up the run configuration
            aml_run_config = RunConfiguration(_name=f"Parallel run combination {config.resolution_spec}.{config.seed}")
            aml_run_config.target = aml_resources.compute_target
            aml_run_config.environment = aml_resources.env  # type: ignore # auto

            # create different versions of args for each combination
            temp_config_path = spec_file_basename(config.resolution_spec, config.seed or 0, suffix="yml")
            temp_spec_files.append(temp_config_path)
            with Path(temp_config_path).open("w") as fp:
                yaml.dump(config_dct, fp, Dumper=CustomDumper)
            args.spec_file = temp_config_path

            original_arg_list = sys.argv[1:]
            simulator_args = original_arg_list
            spec_file_index = simulator_args.index("--spec_file")
            simulator_args[spec_file_index + 1] = temp_config_path
            num_runs_index = simulator_args.index("--num_runs")
            if isinstance(num_runs_index, int) and num_runs_index >= 0:
                simulator_args[num_runs_index + 1] = "1"  # pragma: no cover
            else:
                simulator_args += ["--num_runs", "1"]

            # create PipelineData to consume the output of this step in the next (plotting) step
            step_output = PipelineData(
                name=f"outputs_batch_{config.resolution_spec}_{config.seed}",
                output_name=f"outputs_batch_{config.resolution_spec}_{config.seed}",
                datastore=aml_resources.datastore,
                is_directory=True,
            )
            all_run_outputs += [step_output]
            simulator_args += ["--output_dir", step_output]

            step = PythonScriptStep(
                script_name=str(run_script_path.absolute().relative_to(ROOT_DIR)),
                source_directory=ROOT_DIR,
                arguments=simulator_args,
                outputs=[step_output],
                compute_target=aml_resources.compute_target,
                runconfig=aml_run_config,
            )
            parallel_steps.append(step)

    return parallel_steps, all_run_outputs, styled_subsets, temp_spec_files


def specify_plotting_step(
    styled_subsets: Dict[str, List[str]],
    experiment_labels: List[str],
    all_run_outputs: List[PipelineData],
    aml_resources: AMLResources,
    plot_script_path: Path,
) -> Tuple[PythonScriptStep, PipelineData]:
    """
    Create the pipeline steps to plot the results of the simulator.
    Args:
        styled_subsets:
        experiment_labels: A list of parameters to take from the Spec file to use as experiment
        labels. Currently only accepts 'acquisition' and 'batch_strategy'
        all_run_outputs: A list of PipelineData paths to directories where simulator results are stored
        aml_resources: an instance of AMLResources which contains the necessary information on
        AML resources to instantiate pipeline steps

    Returns: The PythonScriptStep for running the data, plus the PipelineData path to where the output will be stored

    """
    output_plot = PipelineData(
        name="plotting_output",
        output_name="plotting_output",
        datastore=aml_resources.datastore,
    )

    styled_subsets_list = []
    for subset_members in styled_subsets.values():
        styled_subsets_list += ["--styled_subset", " ".join([f'"{m}"' for m in subset_members])]  # pragma: no cover

    # assert len(all_run_outputs) == len(experiment_labels)
    experiment_dirs = []
    for run_output in all_run_outputs:
        experiment_dirs += ["--experiment_dirs", run_output]  # pragma: no cover
    plotting_args = (
        ["--output_dir", output_plot, "--num_simulator_samples_per_optimum", "100"]
        + experiment_dirs
        + experiment_labels
    )

    # Plotting
    plotting_run_config = RunConfiguration(_name="Plotting")

    plotting_run_config.target = aml_resources.compute_target
    plotting_run_config.environment = aml_resources.env  # type: ignore # auto

    plotting_step = PythonScriptStep(
        script_name=str(plot_script_path.absolute().relative_to(ROOT_DIR)),
        source_directory=ROOT_DIR,
        arguments=plotting_args,
        inputs=all_run_outputs,
        outputs=[output_plot],
        compute_target=aml_resources.compute_target,
        runconfig=plotting_run_config,
    )
    return plotting_step, output_plot


def data_and_simulation_are_consistent(config: OptimizerConfig) -> bool:  # pragma: no cover
    """
    Check that the input names in the data section of the config match those of the simulator specified in it,
    and likewise the output names. Also check that the data files themselves are consistent with the data
    settings in the config. These checks are also carried out elsewhere, but running them here avoids
    submitting an AML job that will fail as soon as it starts to run.
    """

    data_var_names = sorted(config.data.inputs.keys())
    dataset_ok = True
    missing_names: Set[str] = set()
    data_output_name: Optional[str] = None
    if config.data.folder.is_dir() or config.data.files:
        df = config.data.load_dataframe()
        df_col_names = sorted(df.columns.tolist())
        data_output_name = config.data.output_column
        missing_names = set(data_var_names).union([data_output_name]).difference(df_col_names)
        if missing_names:
            logging.error(
                "One or more columns expected by the config file are missing from the data: "
                + ", ".join(sorted(missing_names))
            )
        try:
            Dataset(df, config.data)
        except ValueError as e:
            logging.error(f"Constructing Dataset object raised a ValueError: {e}")
            dataset_ok = False
    simulator = config.get_simulator()
    data_generator = SimulatedDataGenerator(simulator)
    simulation_var_names = sorted(data_generator.parameter_space.parameter_names)
    input_names_consistent = data_var_names == simulation_var_names
    if not input_names_consistent:  # pragma: no cover
        logging.error("Inputs in the config file must match those of the data generator (simulator)")
        logging.error(f"Inputs in the config:             {', '.join(data_var_names)}")
        logging.error(f"Inputs allowed by data generator: {', '.join(simulation_var_names)}")
    simulation_output_name = data_generator.objective_col_name
    output_names_consistent = (data_output_name == simulation_output_name) or data_output_name is None
    if not output_names_consistent:
        logging.error("Output in the config file must match objective of the data generator (simulator)")
        logging.error(f"Output in the config:            {data_output_name}")
        logging.error(f"Objective of the data generator: {simulation_output_name}")
    return input_names_consistent and output_names_consistent and not missing_names and dataset_ok


def run_simulator_pipeline(
    arg_list: Optional[List[str]],
    run_script_path: Path,
    plot_script_path: Path,
    loop_config_class: Type[OptimizerConfig],
) -> Optional[Run]:  # pragma: no cover
    """
    Creates and runs an Azure ML pipeline to run an entire workflow of 1) running the simulator (with 1
    node per config expansion and seed), 2) plotting the results. Results can be viewed in the AML portal,
    or directly in the Datastore named 'workspaceblobstore' which will be created in the Workspace
    specified in azureml-args.yml
    """
    logging_to_stdout()
    # TODO: replace arg parse with method that expects experiment_labels, styled_subsets as args
    parser = RunConfig()
    args = parser.parse_args(arg_list)
    if not args.submit_to_aml:
        raise ValueError("This script doesn't support local runs. Please ensure --submit_to_aml flag is set ")

    logging.info("Creating .amlignore")
    create_amlignore(run_script_path)

    logging.info("Creating pipeline")
    parser2_result = parse_results_and_add_azure_config_vars(parser, arg_list)
    azure_config = AzureConfig(**parser2_result.args)
    aml_resources = AMLResources(azure_config)

    parallel_steps, all_run_outputs, styled_subsets, temp_spec_files = specify_run_step(
        args, aml_resources, run_script_path, loop_config_class
    )
    if not parallel_steps:
        logging.error("All config resolutions were dropped - bailing out")
        return None

    experiment_labels = [
        "--experiment_labels",
        "acquisition",
        "--experiment_labels",
        "batch_strategy",
        "--experiment_labels",
        "batch",
        "--experiment_labels",
        "hmc",
    ]
    plotting_step, plotting_datastore = specify_plotting_step(
        styled_subsets,
        experiment_labels,
        all_run_outputs,
        aml_resources,
        plot_script_path,
    )

    all_steps = StepSequence(steps=[parallel_steps, plotting_step])
    pipeline = Pipeline(workspace=aml_resources.ws, steps=all_steps)

    logging.info("Validating pipeline")
    pipeline.validate()
    logging.info("Creating experiment")
    expt = Experiment(aml_resources.ws, "simulator_pipeline")
    logging.info("Submitting pipeline run")
    pipeline_run = expt.submit(pipeline)  # noqa: F841

    # remove the temporary Spec file created earlier
    [os.remove(spec_file) for spec_file in temp_spec_files]  # type: ignore
    return pipeline_run


def run_simulations_in_pipeline(arg_list: Optional[List[str]], loop_config: Type[OptimizerConfig]):  # pragma: no cover
    logging_to_stdout()
    parser = RunConfig()
    args = parser.parse_args(arg_list)

    # Multiple resolutions, if any, will be handled inside a single AML run.
    _, config_dct = load_config_from_path_or_name(args.spec_file)
    rd_path = Path(config_dct["results_dir"])
    if rd_path.name.startswith("seed"):
        results_dir = f"{args.output_dir}/{rd_path.parent.name}/{rd_path.name}"
    else:
        results_dir = args.output_dir
    logging.info(f"Updating results dir from {config_dct['results_dir']} to {results_dir}")
    config_dct["results_dir"] = results_dir

    create_config_json_file(config_dct, args)

    # Expand config
    selections_and_configs = load_resolutions_from_command(args, loop_config)

    for pair_list in selections_and_configs:
        run_multiple_seeds(args, pair_list)
