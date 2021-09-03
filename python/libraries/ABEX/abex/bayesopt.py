# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Module that allows to fit a GP model and generate a new batch.

Exports:
    BayesOptModel
    HMCBayesOptModel
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, OrderedDict, Sized, Tuple, Type

import GPy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from abex import plotting
from abex.dataset import Dataset
from abex.emukit.moment_matching_qei import SequentialMomentMatchingEICalculator
from abex.settings import AcquisitionOptimizer, BatchAcquisitionStrategy, ModelSettings, OptimizerConfig
from abex.transforms import InvertibleTransform
from emukit.bayesian_optimization.acquisitions import EntropySearch, ExpectedImprovement
from emukit.bayesian_optimization.acquisitions.expected_improvement import MeanPluginExpectedImprovement
from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition
from emukit.bayesian_optimization.local_penalization_calculator import LocalPenalizationPointCalculator
from emukit.core import CategoricalParameter, ContinuousParameter, Parameter, ParameterSpace
from emukit.core.acquisition import Acquisition, IntegratedHyperParameterAcquisition
from emukit.core.encodings import OneHotEncoding
from emukit.core.interfaces.models import IModel
from emukit.core.loop import CandidatePointCalculator, SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization import (
    AcquisitionOptimizerBase,
    GradientAcquisitionOptimizer,
    RandomSearchAcquisitionOptimizer,
)
from emukit.model_wrappers import GPyModelWrapper
from psbutils.arrayshapes import Shapes


def compute_emukit_parameter_space(
    continuous_parameters: OrderedDict[str, Tuple[float, float]],
    categorical_parameters: Optional[OrderedDict[str, List]] = None,
) -> ParameterSpace:
    """Compute an emukit parameter space from ordered dictionaries mapping parameter names to their domain.

    Args:
        continuous_parameters: An ordered dictionary mapping continuous parameter names to their domain.
            The order of the parameter names should correspond to the order in which they appear in the input array
            being passed into a model. For the continuous parameters, the domain is defined by a tuple of lower and
            upper bounds, e.g.:
                    {'cont_param1': (0, 10.), 'cont_param2': (0.1, 1.0)}
        categorical_parameters: Ordered dictionary mapping categorical parameter names to their domain. The order of
            parameter names should correspond to the order in which the onehot encoding will appear in the data
            being passed into a model. For categorical parameters, the domain is defined by the array of possible
            categories , e.g.:
                    {'categ_param1': ['red', 'green', 'blue']}

            Note: this array has to be ordered in the way that corresponds to that of onehot encodings
            being passed to the model! If onehot encoding [1, 0] encodes 'red' and [0, 1] encodes 'green', the array
            for that category _must_ be ordered as ['red', 'green'].

    Returns:
        ParameterSpace: An Emukit ParameterSpace instance
    """
    cont_variables: List[Parameter] = [
        ContinuousParameter(key, low, high) for key, (low, high) in continuous_parameters.items()
    ]
    categ_variables: List[Parameter] = [
        CategoricalParameter(key, OneHotEncoding(list(values)))
        for key, values in (categorical_parameters or {}).items()
    ]
    return ParameterSpace(cont_variables + categ_variables)


class BayesOptModel:
    """Class for fitting a surrogate model and applying Bayesian Optimization to maximize a given output.

    Note: The Bayesian Optimization procedure in this class _maximises_ a given output. Since EmuKit minimises by
        default, the internal surrogate model actually predicts the _negative objective_.

        The conversion to negative objective to interface with emukit happens internally in this class. The user
        should treat this as an objective maximising class, and can pass regular (non-negated) values of the objective
        when interfacing with this class.
    """

    def __init__(
        self,
        dataset: Dataset,
        model_settings: ModelSettings,
        acquisition_class: Type[Acquisition],
        results_dir: Path,
        training_iterations: int,
        bayesopt_context: Optional[Dict] = None,
        test_ids: Optional[Sized] = None,
        fold_id: int = 0,
    ):
        """
        Args:
            dataset (Dataset): Data to fit the model to (including train and test splits if doing cross-validation)
            model_settings (ModelSettings): model configuration options
            acquisition_class: acquisition function class.
            results_dir: directory to write results to
            training_iterations: number of training iterations to do
            bayesopt_context: optional context
            test_ids (Optional[Sized], optional): If doing cross-validation, contains the indices of the examples in
                dataset that are part of the test set for this model. Defaults to None (no train-test split).
            fold_id (int, optional): Which cross-validation fold does this model correspond to. Used for logging
                and saving metrics. Defaults to 0.
        """

        self.model_settings: ModelSettings = model_settings
        self.fold_id: int = fold_id

        # Train and test datasets. Partition the data as requested
        self.train: Dataset
        self.test: Optional[Dataset]
        # We can't shorten this to "if test_ids:" because it can be a numpy array, which cannot be treated
        # as a boolean.
        if test_ids is not None and len(test_ids) > 0:
            self.train, self.test = dataset.train_test_split(test_ids)  # pragma: no cover
        else:
            self.train = dataset
            self.test = None

        self.acquisition_class: Type[Acquisition] = acquisition_class
        self.acquisition: Optional[Acquisition] = None

        self.results_dir: Path = results_dir
        self.training_iterations = training_iterations
        # Params set when we define and train the GP
        self.emukit_model: Optional[GPyModelWrapper] = None

        # Params set when we prepare for Bayes-Opt by defining a parameter space
        self.continuous_parameters: OrderedDict[str, Tuple[float, float]]
        self.categorical_parameters: OrderedDict[str, List]
        self.continuous_parameters, self.categorical_parameters = self.train.parameter_space
        # Context for generating a batch with Bayesian optimization
        self.context: Dict[str, Any] = self._map_context(bayesopt_context or {}, dataset)

    @classmethod
    def from_config(cls, config: OptimizerConfig, dataset: Dataset, test_ids: Optional[Sized] = None, fold_id: int = 0):
        """
        Args:
            config: configuration options
            dataset: Data to fit the model to (including train and test splits if doing cross-validation)
            test_ids: If doing cross-validation, contains the indices of the examples in
                dataset that are part of the test set for this model. Defaults to None (no train-test split).
            fold_id: Which cross-validation fold does this model correspond to. Used for logging
                and saving metrics. Defaults to 0.
        """
        return cls(
            dataset=dataset,
            model_settings=config.model,
            acquisition_class=config.bayesopt.get_acquisition_class(),
            results_dir=config.results_dir,
            training_iterations=config.training.iterations,
            bayesopt_context=config.bayesopt.context,
            test_ids=test_ids,
            fold_id=fold_id,
        )

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def space(self) -> ParameterSpace:
        """Returns model's EmuKit parameter space. Caches the parameter space to avoid recomputing.
        Note: In Python >3.8 '@property@lru_cache' can be replaced with '@functools.cached_property'
        """
        return compute_emukit_parameter_space(self.continuous_parameters, self.categorical_parameters)

    @property
    def _x_train(self) -> np.ndarray:
        """Get the training input data"""
        return self.train.inputs_array

    @property
    def _y_train(self) -> np.ndarray:
        """Get the training output data"""
        return self.train.output_array

    @property
    def _x_test(self) -> Optional[np.ndarray]:  # pragma: no cover
        """Get the test input data"""
        return self.test.inputs_array if self.test else None

    @property
    def _y_test(self) -> Optional[np.ndarray]:  # pragma: no cover
        """Get the test output data"""
        return self.test.output_array if self.test else None

    def run(self) -> "BayesOptModel":
        """
        Define and train the Gaussian process, then quantify and log its performance.
        """
        self.run_training()
        self._quantify_performance()
        return self

    def _quantify_performance(self) -> Tuple[float, Optional[float]]:
        """
        Quantify and log performance; extend as required in subclasses.
        """
        return self.r_train, self.r_test  # type: ignore

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def r_train(self) -> float:
        """Get the Pearson correlation coefficient for the training data."""
        r_train = self._pearson(self._x_train, self._y_train)
        logging.debug(f"- r_train : {r_train:.3f}")
        return r_train

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def r_test(self) -> Optional[float]:
        """Get the Pearson correlation coefficient for the test data if there is any."""
        if self.test is not None:  # pragma: no cover
            assert self._x_test is not None and self._y_test is not None  # for mypy
            r_test = self._pearson(self._x_test, self._y_test)
            logging.debug(f"- r_test : {r_test:.3f}")
            return r_test
        else:
            return None

    def run_training(self) -> None:
        """
        Run Gaussian Process training and create the acquisition function.
        """
        gpr = self._construct_gpy_model()
        self.emukit_model = GPyModelWrapper(gpr)
        logging.info(f"- Running optimization with {self.training_iterations} iterations")
        self.emukit_model.optimize()
        self.acquisition = self._make_acquisition_from_model(self.emukit_model)

    def _make_acquisition_from_model(self, model: IModel) -> Acquisition:
        """
        Returns an instance of the acquisition class. The value of self.acquisition_class should be one of
        the members of the AcquisitionClass enum, all of which do take a "model" argument, even though this
        is not the case for Acquisition in general.
        """
        if self.acquisition_class is EntropySearch:  # pragma: no cover
            # Entropy Search requires one extra argument.
            # Here and in several other places, we ignore an apparent type error caused by the
            # use of @lru_cache, which causes variables that would otherwise just be float to have
            # the type lru_cache_wrapper(float). TODO figure out if lru_cache is really needed.
            return EntropySearch(model=model, space=self.space)  # type: ignore # lru_cache_wrapper
        else:
            return self.acquisition_class(model=model)  # type: ignore # may not have model argument

    def _construct_gpy_model(self) -> GPy.models.GPRegression:
        """Construct a GPy regression model."""
        kernel_func = self.model_settings.get_kernel()
        # This line used to be tried twice, with all exceptions being caught from the first try.
        # Removed, to test the hypothesis that it's no longer necessary.
        kern = kernel_func(
            self.train.ndims, ARD=self.model_settings.anisotropic_lengthscale
        )  # type: ignore # not callable

        if self.model_settings.add_bias:  # pragma: no cover
            bias_kern = GPy.kern.Bias(self.train.ndims)
            kern = kern + bias_kern

        # The model produces -1 * observations, so that Bayes-Opt can minimize it
        gpr = GPy.models.GPRegression(self._x_train, -self._y_train[:, np.newaxis], kern)

        # Set the priors for hyperparameters (if specified in the config):
        for param_name, prior_config in self.model_settings.priors.items():  # pragma: no cover
            if param_name not in gpr.parameter_names():
                raise ValueError(  # pragma: no cover
                    f"No such hyperparameter {param_name} to set a prior for. Hyperparameters present in the model "
                    f"are:\n{gpr.parameter_names()}"
                )
            prior = prior_config.get_prior()
            gpr[param_name].set_prior(prior)  # type: ignore # set_prior missing for ParamConcatenation
            logging.info(f"- Prior on {param_name}: {prior}")

        # Fix the hyperparameters as specified in config
        if self.model_settings.fixed_hyperparameters:
            for hyperparam_name, value in self.model_settings.fixed_hyperparameters.items():
                if hyperparam_name not in gpr.parameter_names():
                    raise ValueError(  # pragma: no cover
                        f"No such hyperparameter {hyperparam_name} to fix in the model. Hyperparameters present are:"
                        f"\n{gpr.parameter_names()}"
                    )
                logging.info(f"- Fixing {hyperparam_name} as {value}")
                gpr[hyperparam_name] = value
                gpr[hyperparam_name].fix()
        return gpr

    def minus_predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run prediction, return the negative of the result.
        """
        assert self.emukit_model is not None
        mean, var = self.emukit_model.predict(x)
        result = (-mean).ravel(), var.ravel()
        # Initial specification, used to get constraints via shape inference:
        # Shapes(x, "XX")(mean, "MU")(var, "VA")(result[0], "MR")(result[1], "VR")
        Shapes(x, "X,Z")(mean, "X,1")(var, "X,1")(result[0], "X")(result[1], "X")
        return result

    def _pearson(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Pearson correlation between observations y and the model evaluated at corresponding inputs x

        Args:
            x, input data, at which the model is to be evaluated
            y, output data
        """
        pred_mean, _ = self.minus_predict(x)
        Shapes(x, "X,Z")(y, "X")(pred_mean, "X")
        score = np.corrcoef(pred_mean, y)[1, 0]
        return float(score)

    # TODO: Currently quantifying model performance using the Pearson correlation coefficient.
    # TODO: Maybe useful to add marginal likelihood / cross-validation likelihood to plots/outputs?

    def predict_with_preprocessing(
        self, df: pd.DataFrame, confidence_interval: float = 0.6827
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
        """Given a DataFrame df in the pre-transformed (original) space, apply preprocessing transforms to transform
        data to model's input space, predict with the model, and transform the output back to original space.

        Output three arrays: The inverse-transformed mean of the predictions, and the lower and upper bounds of the
        confidence interval around the mean, such that the predicted probability of the output being within
        [lower_bound, upper_bound] is equal to the value of confidence_interval. Lower-bound and upper-bound
        are symmetrically placed around the mean in the model space.
        """
        pred_mean, pred_var = self._predict_transformed_output_with_preprocessing(df)
        s = Shapes(df, "DF")(pred_mean, "PM")(pred_var, "PV")
        pred_std = np.sqrt(pred_var)
        # Calculate the lower and upper bounds for the confidence interval given in model space
        std_multiplier = scipy.stats.norm.ppf(0.5 + confidence_interval / 2.0)  # type: ignore # norm does exist
        lower_bound = pred_mean - std_multiplier * pred_std
        upper_bound = pred_mean + std_multiplier * pred_std
        # Transform back to original space (os)
        pred_mean_os, pred_lower_bound_os, pred_upper_bound_os = map(
            self._output_array_to_orig_space, (pred_mean, lower_bound, upper_bound)
        )
        s(pred_mean_os, "PMO")(pred_lower_bound_os, "PLO")(pred_upper_bound_os, "PUO")
        return pred_mean_os, pred_lower_bound_os, pred_upper_bound_os

    def _predict_transformed_output_with_preprocessing(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        """Given a DataFrame df in the pretransformed (original) space, apply preprocessing transforms to transform
        data to model's input space, and predict with the model (returning the output in the transformed space).

        Returns:
            Outputs two arrays: mean and variance of prediction in the transformed space.
        """
        model_space_df = self.train.preprocessing_transform(df)
        model_space_df = self.train.onehot_transform(model_space_df) if self.train.onehot_transform else model_space_df
        # Order the columns in right order
        input_array = np.asarray(model_space_df[self.train.transformed_input_names].values)
        # Make the prediction
        pred_mean, pred_var = self.minus_predict(input_array)
        Shapes(df, "DF")(model_space_df, "MS")(input_array, "IA")(pred_mean, "PM")(pred_var, "PV")
        return pred_mean, pred_var

    def _output_array_to_orig_space(self, output_space_array: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Helper function to transform the arrays representing the predictions from the model to the original
        pretransformed output space. (e.g. if the model predicts log-outputs, this method will return the
        non-logged version of the output."""
        output_space_df = pd.DataFrame(
            {self.train.transformed_output_name: output_space_array}, index=np.arange(len(output_space_array))
        )
        assert isinstance(
            self.train.preprocessing_transform, InvertibleTransform
        ), "Pre-processing must be invertible to transform back to original space"

        original_space_df = self.train.preprocessing_transform.backward(output_space_df)
        result = original_space_df.values.ravel()
        Shapes(output_space_array, "OSA")(output_space_df, "OSD")(original_space_df, "ORD")(result, "RE")
        return result

    @staticmethod
    def _map_context(context_dict: Dict[str, Any], dataset: Dataset) -> Dict[str, Any]:
        """Map a given context through the preprocessing function of the dataset, where a context is a particular
        setting for a subset of the input variables to the model."""
        # Get the context in the transformed space (i.e. in model's input space)
        context_df = pd.DataFrame(context_dict, index=[0])
        transformed_context = dataset.preprocessing_transform(context_df)
        return transformed_context.iloc[0].to_dict()

    def generate_batch(
        self,
        batch_size: int,
        num_samples: int = 1000,
        num_anchors: int = 10,
        lipschitz_constant: Optional[float] = None,
        optimum: Optional[float] = None,
        batch_acquisition_strategy: BatchAcquisitionStrategy = BatchAcquisitionStrategy.LOCAL_PENALIZATION,
        acquisition_optimizer_type: AcquisitionOptimizer = AcquisitionOptimizer.GRADIENT,
        num_batches_left: int = 1,
        nonmyopic_batch: bool = False,
    ) -> np.ndarray:
        """Generate a new batch of experiments in the model's input space (input space after all transforms).

        Returns:
            array of shape (batch_size, n_input_dimensions)
        """
        assert self.acquisition is not None
        assert self.emukit_model is not None
        assert num_batches_left > 0

        effective_batch_size = num_batches_left * batch_size if nonmyopic_batch else batch_size
        acquisition_optimizer = self._make_acquisition_optimizer(acquisition_optimizer_type, num_anchors, num_samples)

        candidate_point_calculator = self._make_candidate_point_calculator(
            acquisition_optimizer, batch_acquisition_strategy, effective_batch_size, lipschitz_constant, optimum
        )

        loop_state = create_loop_state(self._x_train, -self._y_train[:, np.newaxis])
        new_x = candidate_point_calculator.compute_next_points(loop_state, context=self.context)

        if effective_batch_size > batch_size:  # pragma: no cover
            # Subsample the suggested points to fit the batch size:
            selected_point_idxs = np.random.choice(effective_batch_size, size=batch_size, replace=False)
            new_x = new_x[selected_point_idxs]

        assert (
            new_x.shape[0] == batch_size
        ), f"Generated batch size {new_x.shape[0]} doesn't match the requested batch size {batch_size}"
        return new_x

    def _make_acquisition_optimizer(
        self, acquisition_optimizer_type: AcquisitionOptimizer, num_anchors: int, num_samples: int
    ) -> AcquisitionOptimizerBase:
        """Make the acquisition optimizer"""
        space: ParameterSpace = self.space  # type: ignore # lru_cache_wrapper
        if acquisition_optimizer_type is AcquisitionOptimizer.GRADIENT:
            return GradientAcquisitionOptimizer(space, num_samples=num_samples, num_anchor=num_anchors)
        if acquisition_optimizer_type is AcquisitionOptimizer.RANDOM:  # pragma: no cover
            return RandomSearchAcquisitionOptimizer(space, num_eval_points=num_samples)
        raise ValueError(f"No such acquisition optimizer implemented: {acquisition_optimizer_type}")  # pragma: no cover

    def _make_candidate_point_calculator(
        self,
        acquisition_optimizer: AcquisitionOptimizerBase,
        batch_acquisition_strategy: BatchAcquisitionStrategy,
        effective_batch_size: int,
        lipschitz_constant: Optional[float],
        optimum: Optional[float],
    ) -> CandidatePointCalculator:
        """
        Make the (batch) point calculator
        """
        if effective_batch_size == 1:  # pragma: no cover
            assert self.acquisition is not None
            return SequentialPointCalculator(self.acquisition, acquisition_optimizer)
        if batch_acquisition_strategy is BatchAcquisitionStrategy.LOCAL_PENALIZATION:
            assert self.acquisition is not None
            log_acquisition = LogAcquisition(self.acquisition)
            # Note: If optimum given, -optimum is taken, as the internal emukit model minimizes -1*objective.
            fixed_minimum = -optimum if optimum is not None else None
            assert self.emukit_model is not None
            return LocalPenalizationPointCalculator(
                log_acquisition,
                acquisition_optimizer,
                self.emukit_model,
                self.space,  # type: ignore # lru_cache_wrapper
                batch_size=effective_batch_size,
                fixed_lipschitz_constant=lipschitz_constant,
                fixed_minimum=fixed_minimum,
            )
        if batch_acquisition_strategy is BatchAcquisitionStrategy.MMEI:  # pragma: no cover
            if self.acquisition_class not in [ExpectedImprovement, MeanPluginExpectedImprovement]:
                # self.acquisition_class is not used for MomentMatchingExpectedImprovement, but this check
                # ensures intention for using this batch acquisition strategy on the caller side
                raise ValueError(  # pragma: no cover
                    "Acquisition function must be either Expected Improvement or Mean plug-in Expected Improvement "
                    "for the Moment-Matched Expected Improvement strategy"
                )
            is_mean_plugin = self.acquisition_class == MeanPluginExpectedImprovement
            # Moment-matched multi-point Expected Improvement
            assert self.emukit_model is not None
            return SequentialMomentMatchingEICalculator(
                acquisition_optimizer=acquisition_optimizer,
                model=self.emukit_model,
                parameter_space=self.space,  # type: ignore # lru_cache_wrapper
                batch_size=effective_batch_size,
                mean_plugin=is_mean_plugin,
                # TODO: : If using HMC, self.hyperparam_samples will need to be passed in here
            )
        raise ValueError(
            f"No such batch acquisition optimizer implemented: {batch_acquisition_strategy}"
        )  # pragma: no cover

    def get_model_parameters(self, include_fixed: bool = True) -> pd.DataFrame:
        """Return a DataFrame with the values for each model parameter. The column names represent GPy parameter names
        and the single row has the values of those parameters

        Args:
            include_fixed (bool, optional): Whether to include the parameters which are constrained to be fixed
                in the model. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame with the values of model parameters.
        """
        assert self.emukit_model is not None
        assert self.acquisition is not None
        # Remove initial "GPRegression." string from parameter names
        parameter_names_list = [
            name.split(".", 1)[1] for name in self.emukit_model.model.parameter_names_flat(include_fixed=True)
        ]
        parameter_names = np.array(parameter_names_list)

        if include_fixed or not self.emukit_model.model._has_fixes():
            # If no fixed parameters, or parameter names to be included in DataFrame, just add all parameters
            param_df = pd.DataFrame(self.emukit_model.model.param_array[None, :], columns=parameter_names)
        else:
            # Only select parameters that aren't fixed
            is_fixed_param = ~self.emukit_model.model._fixes_  # type: ignore
            unfixed_param_names = parameter_names[~is_fixed_param]
            param_df = pd.DataFrame(self.emukit_model.model.unfixed_param_array[None, :], columns=unfixed_param_names)

        return param_df

    def get_model_parameters_and_log_likelihoods(self, include_fixed: bool = True) -> pd.DataFrame:
        """Return a DataFrame with the values for each parameter similarly to get_model_parameters(), but in
        addition include the marginal log-likelihood in the final column.

        Args:
            include_fixed (bool, optional): Whether to include the parameters which are constrained to be fixed
                in the model. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame with the values of model parameters and the log-likelihood.
        """
        assert self.emukit_model is not None
        assert self.acquisition is not None
        param_df = self.get_model_parameters(include_fixed=include_fixed)
        param_df["Marginal Log-Likelihood"] = self.emukit_model.model.log_likelihood()
        return param_df

    @property
    def has_priors(self) -> bool:  # pragma: no cover
        """Determine whether the model has any priors specified"""
        assert self.emukit_model is not None  # for mypy
        return len(list(self.emukit_model.model.priors.items())) > 0

    def make_priors_plot(self, save_path: Path) -> None:  # pragma: no cover
        """Make a plot of the distributions of the priors for the model.

        Args:
            save_path (Path): path to where to save the plot
        """
        assert self.emukit_model is not None
        assert self.has_priors
        parameter_names = [
            name.split(".", 1)[1] for name in self.emukit_model.model.parameter_names_flat(include_fixed=True)
        ]
        # Get tuples of priors and parameter indexes to which the prior applies
        prior_tuples = list(self.emukit_model.model.priors.items())
        priors = [prior for prior, param_idxs in prior_tuples for idx in param_idxs]
        prior_param_idxs = [idx for prior, param_idxs in prior_tuples for idx in param_idxs]
        prior_param_names = list(map(lambda idx: parameter_names[idx], prior_param_idxs))
        fig = plotting.plot_gpy_priors(priors, prior_param_names)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


class HMCBayesOptModel(BayesOptModel):
    """
    Class for fitting a surrogate model with multiple hyperparameter samples and applying Bayesian Optimization
    to maximize a given output.

    Note: The Bayesian Optimization procedure in this class _maximises_ a given output. Since EmuKit minimises by
        default, the internal surrogate model actually predicts the _negative objective_. For more info
        see BayesOptModel.
    """

    burnin_samples_multiplier = 5
    hmc_subsample_interval = 5

    def get_model_parameters(self, include_fixed: bool = True) -> pd.DataFrame:
        """Return a DataFrame with the HMC samples for each parameter. The column names represent GPy parameter names
        and each row is a single sample from the HMC algorithm.

        Args:
            include_fixed (bool, optional): Whether to include the parameters which are constrained to be fixed
                in the model. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame with the values of model parameters for each HMC sample.
        """
        assert self.emukit_model is not None
        assert self.acquisition is not None
        # It's safe to assume self.acquisition.samples exists, because for HMCBayesOpt we always use
        # IntegratedHyperParameterAcquisition.
        hmc_samples = self.acquisition.samples  # type: ignore

        # Remove initial "GPRegression." string from parameter names
        parameter_names_list = [
            name.split(".", 1)[1] for name in self.emukit_model.model.parameter_names_flat(include_fixed=True)
        ]
        parameter_names = np.array(parameter_names_list)

        if not self.emukit_model.model._has_fixes():
            # If no fixed parameters, just add all parameters
            samples_df = pd.DataFrame(hmc_samples, columns=parameter_names)
        else:
            is_fixed_param = ~self.emukit_model.model._fixes_  # type: ignore
            unfixed_param_names = parameter_names[~is_fixed_param]

            samples_df = pd.DataFrame(hmc_samples, columns=unfixed_param_names)
            if include_fixed:
                # If including fixed parameters and not all parameters are fixed
                fixed_param_names = parameter_names[is_fixed_param]
                fixed_param_array = self.emukit_model.model.param_array[is_fixed_param]
                samples_df[fixed_param_names] = fixed_param_array
        return samples_df

    def get_model_parameters_and_log_likelihoods(self, include_fixed: bool = True) -> pd.DataFrame:
        """Return a DataFrame with the HMC samples for each parameter similarly to get_model_parameters, but in
        addition, the returned DataFrame will have one extra column with the marginal log-likelihood values
        of each of the samples.

        Args:
            include_fixed (bool, optional): Whether to include the parameters which are constrained to be fixed
                in the model. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame with the values of model parameters and the log-likelihood for each HMC sample.
        """
        assert self.emukit_model is not None
        assert self.acquisition is not None
        samples_df = self.get_model_parameters(include_fixed=include_fixed)
        marginal_log_likelihoods = self.get_log_likelihoods()
        samples_df["Marginal Log-Likelihood"] = marginal_log_likelihoods
        return samples_df

    def get_log_likelihoods(self) -> np.ndarray:
        """Return an array of shape [num_hmc_samples] of marginal log-likelihoods corresponding to each of the
        HMC samples.
        """
        assert self.emukit_model is not None
        assert self.acquisition is not None

        marginal_log_likelihoods = np.zeros([self.acquisition.n_samples])  # type: ignore # assume has n_samples
        for i, sample in enumerate(
            self.acquisition.samples  # type: ignore # assume has samples
        ):  # TODO: e.g. replace with self.hyperparam_samples
            self.emukit_model.fix_model_hyperparameters(sample)
            marginal_log_likelihoods[i] = self.emukit_model.model.log_likelihood()
        return marginal_log_likelihoods

    def run_training(self) -> None:
        """Run the training sequence. HMC sampling, and then generate a plot of the samples."""
        gpr = self._construct_gpy_model()
        self.emukit_model = GPyModelWrapper(gpr)
        self._run_hmc()
        fname = self.results_dir / f"hmc_samples_fold{self.fold_id}.png"
        self._plot_hmc(fname=fname)

    def _run_hmc(self) -> None:
        """Perform HMC sampling"""
        assert self.emukit_model is not None
        # noinspection PyUnresolvedReferences
        logging.info(f"- Running HMC with {self.training_iterations} iterations")
        self.acquisition = IntegratedHyperParameterAcquisition(
            self.emukit_model,
            self._make_acquisition_from_model,
            n_burnin=self.burnin_samples_multiplier * self.training_iterations,
            n_samples=int(self.training_iterations / self.hmc_subsample_interval),
            subsample_interval=self.hmc_subsample_interval,
        )

    def _plot_hmc(self, fname: Optional[Path] = None) -> None:
        """
        Extract the HMC samples from the model and create plot(s).

        Args:
            - fname (str): Upon providing a filename, the plot will be saved to disk
        """
        assert self.emukit_model is not None
        assert self.acquisition is not None
        samples_df = self.get_model_parameters_and_log_likelihoods(include_fixed=False)
        plotting.hmc_samples(samples_df, fname=fname)

    def minus_predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run prediction, return the negative of the result.
        """
        assert self.emukit_model is not None
        mean, var = self._hmc_predict(x)
        return (-mean).ravel(), var.ravel()

    def _hmc_predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using the internal emukit_model using all the hmc samples. Note that this predicts minus the
        objective, as the internal emukit model is fed the negated objective as training data.

        Note: This isn't exact inference!! The actual predictive distribution will be a mixture of Gaussians.
            This function returns the first two moments of that mixture of Gaussians.

        Args:
            x (np.ndarray): Input data on which to predict corresponding objective

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and variance of the predictive distribution (which is not Gaussian)
                at points x.
        """
        assert isinstance(self.acquisition, IntegratedHyperParameterAcquisition)
        assert self.emukit_model is not None

        means, variances = np.zeros([x.shape[0], 1]), np.zeros([x.shape[0], 1])
        for sample in self.acquisition.samples:  # TODO: replace with self.hyperparam_samples
            self.emukit_model.fix_model_hyperparameters(sample)
            mean_sample, var_sample = self.emukit_model.predict(x)
            means += mean_sample
            variances += var_sample
        return means / self.acquisition.n_samples, variances / self.acquisition.n_samples
