# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Implements DataLoop, which allows running multiple steps of Bayesian Optimization."""
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from abex.optimizers.optimizer_base import OptimizerBase
from abex.plotting import plot_convergence
from abex.plotting.composite_core import plot_multidimensional_function_slices
from abex.settings import OptimizerConfig
from abex.data_settings import DataSettings
from abex.simulations.interfaces import DataGeneratorBase, SimulatedDataGenerator
from abex.space_designs import DesignType, suggest_samples
from emukit.core import ContinuousParameter, ParameterSpace


class DataLoop:
    """Manages the entire loop of generating simulated data and performing Bayes Opt. on it. I.e. it manages
    the following steps:

    0) (optional) Generate initial experiment data and save to csv (if no initial data files supplied in config)
    1) Run the ABEX Bayesian Optimisation procedure to generate the next batch of experimental inputs.
    2) Save evaluations, plots and the next batch from this iteration in a subdirectory of config.results
    3) Simulate experiment for the batch of inputs produced in step 2. Save it to a csv file.
    4) Repeat steps 1-3 for a given number of steps

    Methods:
        generate_init_batch, generation of an initial batch
        run_loop, main method, for running the optimization loop

    Fields:
        data_generator (DataGeneratorBase): data generator used to provide experimental observations
        config (OptimizerConfig): configuration object for the Bayesian Optimization
        num_init_points (int, optional): if given, random initialization batch with this number of points is generated

    Pre-defined fields:
        initial_batch_filename (str): filename of the initial batch of inputs
        initial_data_filename (str): filename of the experimental data corresponding to the initial batch
        initial_data_key (str): OptimizerConfig file for initial data retrieval
        data_filename_base (str): template name for generated batches
        data_key_base (str): OptimizerConfig key for batch generation
        iter_results_dir_base (str): template for naming generated directories corresponding to different loop steps
        convergence_plot_filename (str): filename of a generated convergence plot
        sim_visualisation_filename (str): filename for the plot with 1D and 2D slices of the objective

    """

    # Filename for initial batch of inputs
    initial_batch_filename: str = "init_batch.csv"
    # Key to be used to refer to a experiment data file in OptimizerConfig.
    # Will be formatted with an integer:
    data_key_base: str = "experiment_outcomes_batch_{:06d}"
    # Filename for experiment data corresponding to a given step of the loop.
    data_filename_base: str = data_key_base + ".csv"
    # Filename for experiment data corresponding to initial batch
    initial_data_filename: str = data_filename_base.format(0)
    # String key to be used to refer to the initial data file in the OptimizerConfig dictionary
    initial_data_key: str = "Initial Data"
    # Results for a given iteration of the loop will be saved in a subdirectory in the results_dir with this name:
    iter_results_dir_base: str = "iter{}"
    # Filename for the convergence plots
    convergence_plot_filename: str = "convergence_plot.png"
    # Filename for simulation slices visualisation plot
    sim_visualisation_filename: str = "simulation_slices_visualisation.png"

    def __init__(
        self,
        data_generator: DataGeneratorBase,
        data_settings: DataSettings,
        results_dir: Path,
        experiment_batch_path: Path,
        num_init_points: Optional[int] = None,
        design_type: DesignType = DesignType.RANDOM,
        seed: int = 0,
    ):
        """
        Args:
            data_generator: Class that generates csv-s of experimental outputs given a csv batch of
                next inputs. The data_generator can simulate the experiments, or generate the data in any other way
                (e.g. (in principle) send a request to perform a real experiment)
            data_settings: data config for the Bayesian Optimization procedure
            results_dir: directory to write results to
            experiment_batch_path: ...
            num_init_points: If no initial data files supplied in config, how many initial
                data points to generate from the data_generator for random inputs. Will be ignored if initial files
                are specified in config. Defaults to None.
            seed: The random seed to set for the experiments

        Raises:
            ValueError: If no initial files specified in config, and if num_init_points is not > 0
        """
        self.data_generator: DataGeneratorBase = data_generator
        self.data_settings: DataSettings = data_settings
        self.results_dir: Path = results_dir
        self.experiment_batch_path = experiment_batch_path
        self.num_init_points: Optional[int] = num_init_points
        self.design_type: DesignType = design_type
        # Validate correctness of the config fields
        self._validate_config_inputs()
        # Set the random seed
        self.seed = seed
        np.random.seed(seed)
        # Check if any initial data is given. If not, and num_init_points given, generate an initial batch
        if self.data_settings.files:
            logging.info(  # pragma: no cover
                f"Not generating initial batch, using {len (self.data_settings.files)} files "
                f"in {self.data_settings.folder} instead"
            )
        elif not num_init_points:
            raise ValueError(  # pragma: no cover
                "Either initial data must be provided in the config, or num_init_points must be " "greater than 0."
            )
        else:
            # If no initial files supplied, collect initial data (generate inputs through random design)
            self.generate_init_batch(num_init_points, design_type)

    @classmethod
    def from_config(
        cls,
        data_generator: DataGeneratorBase,
        config: OptimizerConfig,
        num_init_points: Optional[int] = None,
        design_type: DesignType = DesignType.RANDOM,
    ):
        """
        Args:
            data_generator: Class that generates csv-s of experimental outputs given a csv batch of
                next inputs. The data_generator can simulate the experiments, or generate the data in any other way
                (e.g. (in principle) send a request to perform a real experiment)
            config: config for the Bayesian Optimization procedure
            num_init_points: If no inital data files supplied in config, how many initial
                data points to generate from the data_generator for random inputs. Will be ignored if initial files
                are specified in config. Defaults to None.
            design_type: how to choose initial points
        Raises:
            ValueError: If no initial files specified in config, and if num_init_points is not > 0
        """
        # Prevent any modifications to parts of the config by this class (e.g. data.files) altering the config
        # that was passed in.
        config = config.copy(deep=True)
        return cls(
            data_generator,
            config.data,
            config.results_dir,
            config.experiment_batch_path,
            num_init_points,
            design_type,
            config.seed or 0,
        )

    def _validate_config_inputs(self) -> None:
        """Validate that config given is compatible with self.data_generator

        Raises:
            ValueError: If inputs in config do not match those of self.data_generator
        """
        # Assert that the inputs specified in the config file conform to the inputs taken by the DataGenerator
        if set(self.data_settings.inputs.keys()) != set(self.data_generator.parameter_space.parameter_names):
            raise ValueError(  # pragma: no cover
                "Inputs in the config file must match those of the data generator (experiment sim.). "
                f"Inputs in the config:\n{sorted(self.data_settings.input_names)}\n"
                f"Inputs allowed by data generator:\n{sorted(self.data_generator.parameter_space.parameter_names)}"
            )

    def generate_init_batch(self, num_init_points: int, design_type: DesignType) -> None:
        """Used when no initial data files have been supplied to generate an initial batch of experimental data.

        1) Generates an initial batch of inputs via random design and saves it to a csv, 2) runs the experiment
        and saves the results to a csv for those inputs, 3) updates config.data.files with the newly generated file.

        TODO: Once space_designs has been moved to be part of core abex functionality (rather than being
            located in CellSignalling), initial batch generation should invoke that functionality, rather than it being
            duplicated here.
        """
        logging.info(f"Generating initial batch with {num_init_points} data-points.")
        init_batch_inputs: np.ndarray = self._generate_init_batch_inputs(num_init_points, design_type)
        batch_inputs_path: Path = self._save_init_batch_to_csv(init_batch_inputs)

        data_folder = self.data_settings.simulation_folder
        experiment_outcomes_path = data_folder / self.initial_data_filename

        self.data_generator.run_experiment_from_csv_to_csv(batch_inputs_path, experiment_outcomes_path)
        self.data_settings.files[self.initial_data_key] = self.initial_data_filename

    def _generate_init_batch_inputs(
        self, num_init_points: int, design_type: DesignType = DesignType.RANDOM
    ) -> np.ndarray:

        bounds = self.data_settings.get_bounds_from_config_and_data()
        transformed_params: List[ContinuousParameter] = []
        for param_name in bounds.keys():
            parameter_config = self.data_settings.inputs[param_name]
            if parameter_config.log_transform:
                transformed_params.append(  # pragma: no cover
                    ContinuousParameter(
                        param_name,
                        np.log10(bounds[param_name][0] + parameter_config.offset),
                        np.log10(bounds[param_name][1] + parameter_config.offset),
                    )
                )
            else:
                transformed_params.append(ContinuousParameter(param_name, *bounds[param_name]))

        transformed_param_space = ParameterSpace(transformed_params)
        # Generate samples
        samples = suggest_samples(
            parameter_space=transformed_param_space, design_type=design_type, point_count=num_init_points
        )

        # Move back to original space
        for i, name in enumerate(transformed_param_space.parameter_names):
            parameter_config = self.data_settings.inputs[name]
            if parameter_config.log_transform:
                samples[:, i] = 10 ** samples[:, i] - parameter_config.offset  # pragma: no cover
        return samples

    def _save_init_batch_to_csv(self, batch_inputs: np.ndarray) -> Path:
        # Create the directory and infer the path
        results_dir = Path(self.results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)
        batch_path = results_dir / self.initial_batch_filename

        # Save the batch
        batch_df = pd.DataFrame(batch_inputs, columns=self.data_settings.input_names)
        batch_df.to_csv(batch_path, index=False)
        return batch_path

    def run_loop(
        self,
        num_iter: int,
        optimizer: OptimizerBase,
        plot_sim_function_slices: bool = False,
    ) -> None:
        """Runs the (Bayesian Optimization -> batch of inputs), (batch of inputs -> experiment) loop for a given number
        of iterations.

        Saves plots and results for each iteration along the way in a subdirectory of config.results_dir.

        Args:
            num_iter: Number of iterations to run the experiment/Bayes.Opt. loop
            plot_sim_function_slices: whether a plot with 1D and 2D slices of the objective should be created
            optimizer: optimizer to run
        """

        for i in range(1, num_iter + 1):
            logging.info(f"Starting Bayesian Optimization loop step {i}/{num_iter}")
            # Update num. batches left
            self.data_settings.num_batches_left = num_iter - i + 1

            # Run the optimization function. Check if it returned the path to the generated batch.
            iter_dir_name = self.iter_results_dir_base.format(i)
            experiment_batch_path, _ = optimizer.run_with_adjusted_config(iter_dir_name, self.data_settings.files)
            experiment_batch_path = experiment_batch_path or self.experiment_batch_path

            # Run the simulation on the batch from the Bayes-Opt iteration
            if num_iter > 1e6:
                logging.warning("Num. iter too large. Files won't be saved in lexicographic order.")  # pragma: no cover
            experiment_outcomes_filename = self.data_filename_base.format(i)
            experiment_outcomes_path = self.data_settings.simulation_folder / experiment_outcomes_filename
            self.data_generator.run_experiment_from_csv_to_csv(experiment_batch_path, experiment_outcomes_path)
            if experiment_outcomes_path.exists():
                logging.info(f"Written: {experiment_outcomes_path}")
            else:
                logging.error(f"NOT written: {experiment_outcomes_path}")  # pragma: no cover
            # Add the most recently simulated data file to config
            self.data_settings.files[self.data_key_base.format(i)] = experiment_outcomes_filename

        # Make final plots and evals
        batch_df_list = self._get_batch_df_list(num_iter)
        self._make_convergence_plot(batch_df_list)
        if plot_sim_function_slices:
            self._make_true_simulated_function_slices_plot(batch_df_list)  # pragma: no cover

    def _make_convergence_plot(self, batch_df_list: List[pd.DataFrame]) -> None:
        """Make a convergence plot for this loop run."""
        # Append a batch number indicator to each DataFrame
        batch_num_col = "Batch Number"
        run_col = "Run Name"
        for i, df in enumerate(batch_df_list):
            assert batch_num_col not in df.columns
            df[batch_num_col] = i
        combined_batches_df = pd.concat(batch_df_list)
        combined_batches_df[run_col] = self.results_dir.name
        fig, _ = plot_convergence(
            combined_batches_df,
            objective_col=self.data_settings.output_column,
            batch_num_col=batch_num_col,
            run_col=run_col,
        )
        assert fig is not None
        fig.savefig(self.results_dir / self.convergence_plot_filename, bbox_inches="tight")
        plt.close(fig)

    def _make_true_simulated_function_slices_plot(self, batch_df_list: List[pd.DataFrame]) -> None:  # pragma: no cover
        assert isinstance(
            self.data_generator, SimulatedDataGenerator
        ), "DataGenerator must be a simulator if slices from it are to be plotted"
        combined_df = pd.concat(batch_df_list)
        bounds_dict = self.data_settings.get_bounds_from_config_and_data()
        # Order the inputs as required by the simulator (ordering in config might be different)
        input_names = self.data_generator.parameter_space.parameter_names
        bounds_values = [bounds_dict[input_name] for input_name in input_names]
        # Get the best point observed in data
        max_idx = combined_df[self.data_settings.output_column].argmax()
        slice_point = combined_df[input_names].iloc[max_idx].values
        # Map the data in dataframes to a list of arrays (one per collected batch)
        obs_points = list(map(lambda df: df[input_names].values, batch_df_list))

        # Plot simulated objective function
        fig, _ = plot_multidimensional_function_slices(
            func=self.data_generator.simulate_experiment,
            slice_loc=slice_point,
            bounds=bounds_values,
            input_names=input_names,
            obs_points=obs_points,  # type: ignore # auto
            input_scales=[
                self.data_settings.input_plotting_scales[name] for name in input_names  # type: ignore # auto
            ],
            output_scale=self.data_settings.output_settings.plotting_scale,
        )
        fig.savefig(self.results_dir / self.sim_visualisation_filename, bbox_inches="tight")

    def _get_batch_df_list(self, num_iter: int) -> List[pd.DataFrame]:
        """Get a list of DataFrames with each corresponding to data from a single batch.
        The 0th element corresponds to initial_data batch, the 1st to the 1st Bayes-opt. batch collected, 2nd to 2nd
        etc.
        """
        # - Load batch DataFrames from Data folder
        # Get the files of the batches generated during the loop
        batch_loop_files = [self.data_filename_base.format(i + 1) for i in range(num_iter)]
        batch_loop_paths = [self.data_settings.simulation_folder / base for base in batch_loop_files]
        # Look for real data if any; if none, expect simulated initial batch
        init_batch_paths = sorted(self.data_settings.folder.glob("*.csv")) or [
            self.data_settings.simulation_folder / self.data_filename_base.format(0)
        ]
        init_batch_df = pd.concat(map(pd.read_csv, init_batch_paths))  # type: ignore # auto
        batch_loop_df_list = [pd.read_csv(batch_file) for batch_file in batch_loop_paths]
        df_list = [init_batch_df] + batch_loop_df_list
        return df_list
