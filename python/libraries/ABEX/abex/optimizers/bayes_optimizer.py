# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""A submodule implementing the Bayesian optimization strategy.
"""
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from abex import optim, plotting
from abex.bayesopt import BayesOptModel, HMCBayesOptModel
from abex.compute_optima import compute_optima
from abex.constants import FOLD
from abex.dataset import Dataset
from abex.optimizers.optimizer_base import OptimizerBase
from abex.settings import OptimizationStrategy
from abex.transforms import InvertibleTransform


class BayesOptimizer(OptimizerBase):

    # .csv filenames
    OPTIMA_CROSS_VALIDATION_BASE = "optima_cross-validation_{}"
    OPTIMA = "optima"
    BATCH_PREDICTED_OBJECTIVE_CSV_BASE = "batch_predicted_objective{}.csv"
    MODEL_PARAMETERS_CSV = "model_parameters.csv"
    TRAINING_CSV = "training.csv"

    # .png filenames
    ACQUISITION2D_ORIGINAL_SPACE_PNG_BASE = "acquisition2d_original_space{}.png"
    ACQUISITION_1D_PNG_BASE = "acquisition1d{}.png"
    BO_DISTANCE_PNG_BASE = "bo_distance{}.png"
    BO_EXPERIMENT_PNG_BASE = "bo_experiment{}.png"
    CALIBRATION_FOLD_PNG_BASE = "calibration{}_fold{}.png"
    CALIBRATION_TRAIN_ONLY_PNG_BASE = "calibration_train_only{}.png"
    MODEL_PRIORS_PNG = "model_priors.png"
    SLICE1D_PNG_BASE = "slice1d{}.png"
    SLICE2D_ORIGINAL_SPACE_PNG_BASE = "slice2d_original_space{}.png"
    SLICE2D_PNG_BASE = "slice2d{}.png"
    TRAIN_ONLY_PNG_BASE = "train_only{}.png"
    TRAIN_TEST_FOLD_PNG_BASE = "train_test{}_fold{}.png"
    XVAL_TEST_PNG_BASE = "xval_test{}.png"

    strategy_name = OptimizationStrategy.BAYESIAN.value

    def _run_logging(self) -> None:  # pragma: no cover
        # Log model parameters
        logging.info("-----------------")
        logging.info("Model")
        logging.info(f"- Kernel: {self.config.model.kernel}")
        lengthscale_type = "Anisotropic" if self.config.model.anisotropic_lengthscale else "Isotropic"
        logging.info(f"- {lengthscale_type} lengthscale")

    def run(self) -> Tuple[Optional[Path], Optional[pd.DataFrame]]:  # pragma: no cover
        """The main method, building the models via cross-validation and evaluating them, building the model using
        the whole data set and suggesting new samples.

        Returns:
            path to the file with suggested samples. If batch size is 0, None is returned instead
            data frame with suggested samples. If batch size is 0, None is returned instead
        """
        dataset: Dataset = self.construct_dataset()

        # - Log values
        self._run_logging()

        # - Cross-validation (train and evaluate auxiliary models)
        num_folds = self.config.training.num_folds
        if num_folds > 1:
            logging.info("-----------------")
            logging.info("Cross-validation")
            # Train all models
            models = self.run_multiple_folds(dataset, num_folds)
            # Evaluate all the models
            if compute_optima:
                self.evaluate_optima(dataset, num_folds, models)  # pragma: no cover

        # - Train the model using all data available
        logging.info("-----------------")
        logging.info("Training of the model using all data")
        penultimate_model = self.run_single_fold(dataset)

        # Find the optimum of the model
        optimum: Optional[float] = None
        if compute_optima:
            optima: pd.DataFrame = self.evaluate_optima(dataset, num_folds=0, models=[penultimate_model])
            optimum = optima[dataset.transformed_output_name].values[0]  # type: ignore # auto

        # - Generate an experiment using batch Bayesian optimization
        if self.config.bayesopt.batch > 0:
            if not isinstance(dataset.preprocessing_transform, InvertibleTransform):
                raise AttributeError("The preprocessing must be invertible to generate a batch")  # pragma: no cover

            # Use the trained penultimate model to generate the batch
            return self.suggest_batch(dataset=dataset, model=penultimate_model, optimum_value=optimum, fold=None)
        else:
            return None, None  # pragma: no cover

    def suggest_batch(
        self,
        dataset: Dataset,
        model: BayesOptModel,
        optimum_value: Optional[float] = None,
        fold: Optional[int] = None,
    ) -> Tuple[Path, pd.DataFrame]:  # pragma: no cover
        """Generates a new batch and saves plots illustrating it.

        Args:
            self.config: self.configuration, in particular contains parameters for Bayesian optimization and results
              location
            dataset: the data set
            model: a GP model
            optimum_value: value at the optimum, in the transformed space
            fold: if cross-validation is used, appropriate information will be added to the generated files

        Returns:
            path to the CSV with locations of new samples to be collected (if ``num_folds=0``, otherwise None)
            data frame with locations of new samples to be collected (if ``num_folds=0``, otherwise None)

        Raises:
            ValueError, if batch size is strictly less than 1, num_folds is strictly less than 0, or no model is
              provided
        """
        if self.config.bayesopt.batch <= 0:
            raise ValueError(
                f"It's impossible to generate a batch with {self.config.bayesopt.batch} samples."
            )  # pragma: no cover

        suffix: str = f"_fold{fold}" if fold is not None else ""

        # Generate the batch in the transformed space. It should have shape (batch_size, n_input_dimensions).
        batch_transformed_space = model.generate_batch(
            self.config.bayesopt.batch,
            num_samples=self.config.bayesopt.num_samples,
            num_anchors=self.config.bayesopt.num_anchors,
            lipschitz_constant=self.config.bayesopt.lipschitz_constant,
            optimum=optimum_value,
            batch_acquisition_strategy=self.config.bayesopt.batch_strategy,
            acquisition_optimizer_type=self.config.bayesopt.acquisition_optimizer,
            num_batches_left=self.config.data.num_batches_left or 1,
            nonmyopic_batch=self.config.bayesopt.nonmyopic_batch,
        )

        # Transform the batch back to original space
        batch_original_space: pd.DataFrame = self.suggestions_to_original_space(
            dataset=dataset, new_samples=batch_transformed_space
        )

        # Compute the search bounds and annotate plot of experiment against existing data
        filtered_dataset = dataset.filtered_on_context(self.config.bayesopt.context)

        fname = self.config.results_dir / self.BO_EXPERIMENT_PNG_BASE.format(suffix)
        columns_to_plot = [col for col in dataset.pretransform_input_names if col not in self.config.bayesopt.context]
        plotting.plot_pred_objective_for_batch(
            batch_original_space,
            model.predict_with_preprocessing,
            bounds=dataset.pretransform_cont_param_bounds,
            dataset=filtered_dataset,
            columns_to_plot=columns_to_plot,
            units=[self.config.data.inputs[col].unit for col in columns_to_plot],
            fname=fname,
            input_scales=[self.config.data.input_plotting_scales[name] for name in columns_to_plot],
            output_scale=self.config.data.output_settings.plotting_scale,
            output_label=dataset.pretransform_output_name,
        )
        # Save the model's prediction on the batch to a file
        self.save_predictions_on_batch(
            model,
            batch_original_space,
            filepath=self.config.results_dir / self.BATCH_PREDICTED_OBJECTIVE_CSV_BASE.format(suffix),
        )

        # Compute the distance between each experiment point (in the preprocessed space)
        plotting.experiment_distance(
            batch_transformed_space, fname=str(self.config.results_dir / self.BO_DISTANCE_PNG_BASE.format(suffix))
        )

        batch_path = (
            self.config.experiment_batch_path_for_fold(fold) if fold is not None else self.config.experiment_batch_path
        )
        batch_original_space.to_csv(batch_path, index=False)

        return batch_path, batch_original_space

    def save_predictions_on_batch(
        self, model: BayesOptModel, batch: pd.DataFrame, filepath: Path
    ) -> None:  # pragma: no cover
        """Save model's predictions (mean, lower and upper confidence bounds) on a batch of data to a file."""
        mean_pred, lb_pred, ub_pred = model.predict_with_preprocessing(batch)
        df = pd.DataFrame(
            {
                "Mean": mean_pred.ravel(),
                "Lower Confidence Bound": lb_pred.ravel(),
                "Upper Confidence Bound": ub_pred.ravel(),
            }
        )
        df.to_csv(filepath, index=False)

    def suggest_batches_crossvalidation(
        self, dataset: Dataset, models: List[BayesOptModel], optima: Optional[pd.DataFrame] = None
    ) -> List[Tuple[Path, pd.DataFrame]]:  # pragma: no cover
        """Generates a new batch and saves plots illustrating it.

        Args:
            dataset: the data set
            models: a list of models
            optima: data frame with locations (and values) of the optima found, in transformed space

        Returns:
            list of tuples:
              - path to the CSV with locations of new samples to be collected (if ``num_folds=0``, otherwise None)
              - data frame with locations of new samples to be collected (if ``num_folds=0``, otherwise None)

        Raises:
            ValueError, if batch size is strictly less than 1, num_folds is strictly less than 0, or no model is
              provided

        See Also:
            suggest_batch, a similar method used if cross-validation is not needed
        """

        logging.info("-----------------")
        logging.info("Generating experiment using Bayesian optimization")
        logging.info(f"- Batch size: {self.config.bayesopt.batch}")

        if len(models) == 0:
            raise ValueError("At least one model is needed.")

        results: List[Tuple[Path, pd.DataFrame]] = []

        # Iterate over models for each fold of cross-validation.
        for fold, model in enumerate(models, 1):
            logging.info(f"- Fold {fold} out of {len(models)}.")

            # Get the value at the optimum, if optima have previously been computed during model evaluation.
            # Note: As ``fold`` starts at 1, we use ``fold-1`` -- arrays in Python are indexed from 0.
            optimum: Optional[float] = (  # type: ignore # auto
                None if optima is None else optima[dataset.transformed_output_name].values[fold - 1]
            )

            # Generate the batch.
            single_output: Tuple[Path, pd.DataFrame] = self.suggest_batch(
                dataset=dataset, fold=fold, model=model, optimum_value=optimum
            )
            results.append(single_output)

        # Return a list of results
        return results

    def evaluate_optima(
        self, dataset: Dataset, num_folds: int, models: List[BayesOptModel]
    ) -> pd.DataFrame:  # pragma: no cover
        """Find and visualise the optima of each model's (mean) prediction. Save the optima in the original
        (pre-transform) space, and plot slices of model predictions at the optima locations. Also, make plots
        of slices through the acquisition function around the optimum found.

        Returns:
            a data frame storing the optimum found (in transformed space) for each model and each context
        """
        logging.info("-----------------")
        logging.info("Evaluating optima")
        # DataFrame storing the optima found for each model (fold)
        optima = pd.DataFrame(
            columns=(
                [FOLD]
                + dataset.transformed_cont_input_names
                + (dataset.transformed_categ_input_names if dataset.n_categorical_inputs else [])
                + [dataset.transformed_output_name]
            )
        )
        for fold, model in enumerate(models):
            if num_folds > 0:
                logging.info(f"- Fold {fold+1}")  # pragma: no cover
            optx_fold, opty_fold = optim.evaluate_model_optima(model, dataset, self.config)
            # Iterate over all contexts, or just iterate once with context=None if there are no categorical
            # variables in the dataset
            for i, context in enumerate(dataset.unique_context_generator_or_none()):
                # Make another row for the dataframe of optima
                optima_df_row = pd.DataFrame({FOLD: [fold]})
                optima_df_row[dataset.transformed_cont_input_names] = optx_fold[i]
                if context:
                    optima_df_row[dataset.transformed_categ_input_names] = context  # pragma: no cover
                optima_df_row[dataset.transformed_output_name] = opty_fold[i]
                optima = optima.append(optima_df_row)

                # Create slice plots
                proc_label: str = "_".join(map(str, context)).replace(" ", "") if context else ""
                suffix: str = "" if num_folds == 0 else f"_fold{fold + 1}"
                total_label: str = "" if proc_label == suffix == "" else f"_{proc_label}{suffix}"
                # Get indices of examples with this context, or all examples if context is None
                dlocs = (
                    np.where((dataset.categorical_inputs_df == context).all(axis=1).values)[0]
                    if context
                    else np.arange(len(dataset))
                )
                # Get the onehot encoding for the context (if context given)
                onehot: Iterable[float] = dataset.categ_context_to_onehot(context) if context else []  # type: ignore
                input_names = list(dataset.pretransform_cont_param_bounds.keys())
                # Make plots
                if self.config.training.plot_slice1d:  # pragma: no cover
                    logging.info(f"- Creating 1d slice plot for context: {proc_label}{suffix}")
                    fname_slice1d = self.config.results_dir / self.SLICE1D_PNG_BASE.format(total_label)
                    plotting.plot_prediction_slices1d(
                        predict_func=model.minus_predict,
                        parameter_space=dataset.continuous_param_bounds,
                        slice_loc=np.append(optx_fold[i, :], onehot),
                        slice_y=opty_fold[i],
                        scatter_x=dataset.continuous_inputs_array[dlocs, :],
                        scatter_y=dataset.output_array[dlocs],
                        output_label=dataset.transformed_output_name,
                        fname=fname_slice1d,
                        num_cols=self.config.training.slice_cols,
                    )
                    fname_acq1d = self.config.results_dir / self.ACQUISITION_1D_PNG_BASE.format(total_label)
                    plotting.acquisition1d(
                        model=model,
                        x0=np.append(optx_fold[i, :], onehot),
                        fname=fname_acq1d,
                    )
                if self.config.training.plot_slice2d:  # pragma: no cover
                    logging.info(f"- Creating 2d slice plot for context: {proc_label}{suffix}")
                    fname_slice2d = self.config.results_dir / self.SLICE2D_PNG_BASE.format(total_label)
                    plotting.plot_prediction_slices2d(
                        predict_func=model.minus_predict,
                        parameter_space=dataset.continuous_param_bounds,
                        slice_loc=np.append(optx_fold[i, :], onehot),
                        scatter_x=dataset.continuous_inputs_array[dlocs, :],
                        scatter_y=dataset.output_array[dlocs],
                        output_label=dataset.transformed_output_name,
                        fname=fname_slice2d,
                    )
                    # Also make a plot in the original space for reference
                    slice_loc_df = self.suggestions_to_original_space(
                        dataset=dataset, new_samples=np.append(optx_fold[i, :], onehot)
                    )[input_names]
                    slice_loc = slice_loc_df.to_numpy()[0, :]  # Convert to array and flatten
                    plotting.plot_acquisition_slices(
                        model=model,
                        dataset=dataset,
                        slice_loc=slice_loc,
                        input_names=input_names,
                        input_scales=[
                            self.config.data.input_plotting_scales[name] for name in input_names
                        ],  # type: ignore # auto
                        onehot_context=onehot if context else None,
                        fname=self.config.results_dir / self.ACQUISITION2D_ORIGINAL_SPACE_PNG_BASE.format(total_label),
                    )
                    # The 2D objective slice in original space currently only works if no categorical inputs
                    if dataset.n_categorical_inputs == 0:
                        fig, _ = plotting.plot_multidimensional_function_slices(
                            func=lambda x: model.predict_with_preprocessing(pd.DataFrame(x, columns=input_names)),
                            func_returns_confidence_intervals=True,
                            slice_loc=slice_loc,
                            bounds=list(dataset.pretransform_cont_param_bounds.values()),
                            input_names=input_names,
                            obs_points=dataset.pretransform_df[input_names].to_numpy(dtype=float),  # type: ignore
                            input_scales=[
                                self.config.data.input_plotting_scales[name] for name in input_names
                            ],  # type: ignore # auto
                            output_scale=self.config.data.output_settings.plotting_scale,
                        )
                        fig.savefig(
                            self.config.results_dir / self.SLICE2D_ORIGINAL_SPACE_PNG_BASE.format(total_label),
                            bbox_inches="tight",
                        )
                        plt.close(fig)

                # Make a plot with data binned in a 2D grid across two chosen dimensions. Plot the GP slice and
                # data for each bin.
                if self.config.training.plot_binned_slices and dataset.n_categorical_inputs == 0:  # pragma: no cover
                    # The binned slices currently work for continuous inputs only.
                    slice_loc_df = self.suggestions_to_original_space(
                        dataset=dataset, new_samples=np.append(optx_fold[i, :], onehot)
                    )[input_names]
                    slice_loc = slice_loc_df.to_numpy()[0, :]  # Convert to array and flatten
                    fig = plotting.plot_slices1d_with_binned_data(
                        input_data=dataset.pretransform_df[input_names].to_numpy(dtype=float),  # type: ignore # auto
                        outputs=dataset.pretransform_df[dataset.pretransform_output_name].to_numpy(),
                        dim_x=self.config.training.plot_binned_slices.dim_x,
                        slice_dim1=self.config.training.plot_binned_slices.slice_dim1,
                        slice_dim2=self.config.training.plot_binned_slices.slice_dim2,
                        num_slices=self.config.training.plot_binned_slices.num_bins,
                        slice_loc=slice_loc,
                        bounds=list(dataset.pretransform_cont_param_bounds.values()),
                        input_scales=[
                            self.config.data.input_plotting_scales[name] for name in input_names
                        ],  # type: ignore # auto
                        input_names=input_names,
                        output_scale=self.config.data.output_settings.plotting_scale,
                        predict_func=lambda x: model.predict_with_preprocessing(pd.DataFrame(x, columns=input_names)),
                    )
                    fig.savefig(self.config.results_dir / f"binned_slices{total_label}.png", bbox_inches="tight")
                    plt.close(fig)

        # Transform the optima back to original space (if possible)
        if isinstance(dataset.preprocessing_transform, InvertibleTransform):
            # We usually run this function for penultimate model (num_folds=0) and cross-validation (num_folds > 0)
            fname = self.OPTIMA if num_folds == 0 else self.OPTIMA_CROSS_VALIDATION_BASE.format(num_folds)

            optima_original_space = dataset.preprocessing_transform.backward(optima)
            optima_original_space.to_csv(self.config.results_dir / f"{fname}.csv", index=False)
            optim.plot_optima(
                optima_original_space,
                self.config.data.categorical_inputs,
                fname=self.config.results_dir / f"{fname}.png",
            )

        return optima

    def by_clause(self, category: Optional[str]) -> str:  # pragma: no cover
        return f"_by_{category.replace(' ', '_')}" if category else ""

    def run_single_fold(self, dataset: Dataset) -> BayesOptModel:  # pragma: no cover
        """
        Fit a single model to the data, and make plots visualising the fit of the model (actual vs. predicted plot)
        """
        # Initialize either an HMC or max-likelihood BayesOpt model
        model_class = HMCBayesOptModel if self.config.training.hmc else BayesOptModel
        model = model_class.from_config(config=self.config, dataset=dataset)
        # Train a single model on the whole data set
        model.run()

        logging.info("- Creating plots")
        # Make a plot of the prior
        if model.has_priors:
            model.make_priors_plot(self.config.results_dir / self.MODEL_PRIORS_PNG)

        # The following seems to be a MyPy False Negative
        categories: Sequence[Optional[str]] = self.config.data.categorical_inputs or [None]  # type: ignore

        for category in categories:
            # noinspection PyUnresolvedReferences
            f, ax = plt.subplots(figsize=(5, 5))
            plotting.plot_predictions_against_observed(
                ax=ax, models=[model], datasets=[model.train], category=category, title="Train only"
            )
            f.savefig(
                self.config.results_dir / self.TRAIN_ONLY_PNG_BASE.format(self.by_clause(category)), bbox_inches="tight"
            )
            fig, ax = plotting.plot_calibration_curve(model.minus_predict, datasets=[model.train], labels=["Train"])
            fig.savefig(
                self.config.results_dir / self.CALIBRATION_TRAIN_ONLY_PNG_BASE.format(self.by_clause(category)),
                bbox_inches="tight",
            )
            plt.close(fig)
            # noinspection PyArgumentList
            plt.close()
        # Save model parameters:
        param_df = model.get_model_parameters_and_log_likelihoods()
        param_df.to_csv(self.config.results_dir / self.MODEL_PARAMETERS_CSV, index=False)
        return model

    def run_multiple_folds(self, dataset: Dataset, num_folds: int) -> List[BayesOptModel]:  # pragma: no cover
        """Split dataset into cross-validation folds, fit model to each fold and make plots visualising the fit of
        the models on test and train data (actual vs. predicted plot)
        """
        # Get a list of arrays of indices for test examples for each cross-validation split
        test_chunks = dataset.split_folds(num_folds, seed=self.config.seed)

        # Create a model for each train/test split and train it
        model_class = HMCBayesOptModel if self.config.training.hmc else BayesOptModel
        models = [
            model_class.from_config(config=self.config, dataset=dataset, test_ids=test_chunks[fold], fold_id=fold + 1)
            for fold in range(num_folds)
        ]

        for model in models:
            model.run()

        # Create train/test figure for each model
        logging.info("- Creating plots for each fold")
        # It seems to be a MyPy False Negative
        categories: Sequence[Optional[str]] = self.config.data.categorical_inputs or [None]  # type: ignore
        for fold, model in enumerate(models, 1):
            for category in categories:
                fname = self.config.results_dir / self.TRAIN_TEST_FOLD_PNG_BASE.format(self.by_clause(category), fold)
                plotting.plot_train_test_predictions(model, category=category, output_path=fname)
                fig, ax = plotting.plot_calibration_curve(
                    model.minus_predict,
                    datasets=[model.train, model.test],  # type: ignore # auto
                    labels=["Train", "Test"],
                )
                fig.savefig(
                    self.config.results_dir / self.CALIBRATION_FOLD_PNG_BASE.format(self.by_clause(category), fold),
                    bbox_inches="tight",
                )
                plt.close(fig)

        # Create cross-validation figure
        logging.info("- Creating cross-validation plots")
        r: Optional[float] = None  # Pearson correlation coefficient
        # noinspection PyArgumentList
        for category in categories:
            fig = plt.figure(figsize=(5, 5))
            # noinspection PyUnresolvedReferences
            ax = plt.subplot()

            # Check if all models have their test data sets
            test_datasets: List[Dataset] = [model.test for model in models if model.test is not None]
            assert len(test_datasets) == len(models), "Some models don't have test data sets."

            # The category does not affect the returned value of r.
            r = plotting.plot_predictions_against_observed(
                ax,
                models=models,  # type: ignore # auto
                datasets=test_datasets,  # type: ignore # auto
                category=category,
                title="Cross-validation",
            )
            fig.savefig(
                self.config.results_dir / self.XVAL_TEST_PNG_BASE.format(self.by_clause(category)), bbox_inches="tight"
            )
            plt.close(fig)

        # Summarise
        assert r is not None
        logging.info(f"- Cross-validation r: {r:.3f}")
        results = pd.DataFrame(
            {
                "Index": [f"Fold {f + 1}" for f in range(num_folds)] + ["Xval"],
                "Train": [m.r_train for m in models] + [None],  # type: ignore # auto
                "Test": [m.r_test for m in models] + [r],  # type: ignore # auto
            }
        )
        results.to_csv(self.config.results_dir / self.TRAINING_CSV, index=False)

        return models  # type: ignore # auto
