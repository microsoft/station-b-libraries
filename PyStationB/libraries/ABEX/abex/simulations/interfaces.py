# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Interfaces for data generators and simulators.

Exports
    SimulatorBase, an interface for a function to be optimized
    DataGeneratorBase, an interface of any experiment, which provides new observations, experimental or simulated
    SimulatedDataGenerator, a wrapper around a simulator (new observations are predicted by the simulator)
"""
import abc
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from azureml.core import Run
from azureml.core.run import _OfflineRun
from emukit.core import ParameterSpace


class SimulatorBase(abc.ABC):
    """Abstract base class for any simulator we use to run artificial experiments.

    It defines a common API so that various synthethic data generation schemes can be considered (e.g. different types
    of noise).

    Methods:
        parameter_space (abstract property, must be implemented by all subclasses)
        _objective (abstract property, must be implemented by all subclasses)
        sample_objective: a wrapper around _objective, which checks the validity of inputs

    """

    @property
    @abc.abstractmethod
    def parameter_space(self) -> ParameterSpace:  # pragma: no cover
        """Returns the parameter space in which the objective can be optimized."""
        pass

    @abc.abstractmethod
    def _objective(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        """This method in facts implements the functionality described under `sample_objective`."""
        pass

    def sample_objective(self, X: np.ndarray) -> np.ndarray:
        """This function should be used to predict objective y1, ..., yn at points x1, ..., xn.

        Args:
            X (ndarray): array of shape (n, m), where m is the number of signalling inputs

        Returns
            ndarray: array of shape (n, 1).

        """
        points_not_in_domain = ~self.parameter_space.check_points_in_domain(X)
        if np.any(points_not_in_domain):  # pragma: no cover
            # Represent the offending points with a string table
            offending_points_table = "\n".join(["\t".join(point.astype(str)) for point in X[points_not_in_domain]])
            raise ValueError(
                f"Inputs out of bounds of the input space. Parameter space bounds for inputs "
                f"{self.parameter_space.parameter_names} are {self.parameter_space.get_bounds()}. Points given:\n"
                + "\t".join(self.parameter_space.parameter_names)
                + "\n"
                + offending_points_table
            )
        return self._objective(X)


class DataGeneratorBase(abc.ABC):
    """Abstract class for interfaces to generate experimental data from batch inputs. This interface can be used around
    a simulated data generator, but can also be used to wrap around an experimental lab protocol if it's desired.

    Abstract methods (must be implemented by all subclasses):
        run_experiment_from_csv_to_csv
        parameter_space (property)
    """

    @abc.abstractmethod
    def run_experiment_from_csv_to_csv(self, inputs_csv_path: Path, output_csv_path: Path) -> None:  # pragma: no cover
        """Simulates an experiment (or a batch) with the given inputs, and saves the results to a csv consumable by
        ABEX Bayes. opt. methods.

        Args:
            inputs_csv_path: CSV with inputs
            output_csv_path: where the output CSV shold be saved
        """
        pass

    @property
    @abc.abstractmethod
    def parameter_space(self) -> ParameterSpace:  # pragma: no cover
        pass


class SimulatedDataGenerator(DataGeneratorBase):
    """This is a data generator wrapping around a simulator.

    Methods:
        parameter_space
        run_experiment_from_csv_to_csv
        input_names
        simulate_experiment
    """

    def __init__(self, simulator: SimulatorBase, objective_col_name: str = "Crosstalk Ratio"):
        """Using a SimulatorBase to simulate the experiment, this class generates synthetic data and saves it to a
        CSV file consumable by the ABEX Bayesian Optimization functionality.

        Args:
            simulator: a simulator used to simulate the experiments
            objective_col_name: a new column with this name is added for storing the optimization objective
        """
        self.simulator = simulator
        self.objective_col_name: str = objective_col_name

    @property
    def parameter_space(self) -> ParameterSpace:
        return self.simulator.parameter_space

    @property
    def input_names(self) -> List[str]:
        """Returns the list of names of all inputs."""
        return self.parameter_space.parameter_names

    def simulate_experiment(self, inputs: np.ndarray) -> np.ndarray:
        """Take input (or a batch of inputs) for the next experiment, and return simulated experiment outputs.

        Args:
            inputs: An array of shape (batch_size, n_inputs) with the inputs to each experiment.

        Returns:
            np.ndarray: An array of shape (batch_size, 1) with the objective for each experiment.

        TODO: Allow for observing multiple outputs (not just signal cross-talk)
        """
        return self.simulator.sample_objective(inputs)

    def run_experiment_from_csv_to_csv(self, inputs_csv_path: Path, output_csv_path: Path) -> None:
        """Simulates an experiment (or a batch) with the given inputs, and saves the results to a csv consumable by
        ABEX Bayes. opt. methods.

        Args:
            inputs_csv_path: CSV with inputs
            output_csv_path: where the output CSV shold be saved
        """
        # Get the inputs as np.ndarray
        inputs: np.ndarray = self._get_inputs_from_csv(inputs_csv_path)
        # Simulate experiment
        objective: np.ndarray = self.simulate_experiment(inputs)
        # Save to CSV
        self._save_output_to_csv(inputs=inputs, output=objective, output_csv_path=output_csv_path)

    def _get_inputs_from_csv(self, inputs_csv_path: Path) -> np.ndarray:
        """Load inputs for the next experiments from a csv file.

        Args:
            inputs_csv_path: Path to the csv file with batch of inputs for the next experiment

        Returns:
            np.ndarray: [batch_size, 4] array with the rows corresponding to inputs for the experiments to perform
        """
        # Load the data from csv
        df = pd.read_csv(inputs_csv_path)
        # Select the right columns and make sure they're in the right order
        df = df[self.input_names]  # type: ignore # auto
        return df.to_numpy()  # type: ignore # auto

    def _save_output_to_csv(
        self, inputs: np.ndarray, output: np.ndarray, output_csv_path: Path
    ) -> None:  # pragma: no cover
        # Make a DataFrame
        df = pd.DataFrame(inputs, columns=self.input_names)
        # Add a column for GeneID which is necessary for parsing a csv into an abex Dataset
        # Append the objective as last column
        df[self.objective_col_name] = output
        # If running on AML, log outputs for comparison
        run = Run.get_context()
        if not isinstance(run, _OfflineRun):
            for col in df.columns:  # type: ignore
                run.log(name=col, value=df[col].values[0])
        # Save the DataFrame to csv.
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path_or_buf=output_csv_path, index=False)
