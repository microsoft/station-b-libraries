# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Emukit's model wrapper using GPyTorch as backend."""
from typing import Dict, List, Optional, Tuple, OrderedDict, Iterator
from contextlib import nullcontext
from operator import itemgetter

import gpytorch
import gpytorch.settings as settings
import logging
import numpy as np
import torch
from emukit.core.interfaces import IDifferentiable, IJointlyDifferentiable, IModel
from gpytorch.lazy import delazify
from botorch.fit import fit_gpytorch_model


class GPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
    ) -> None:
        """
        Args:
            train_x: initial training inputs, shape (n_points, input_dim)
            train_y: initial training outputs, shape (n_points,)
            likelihood: likelihood to be used when training the model. Default: homoscedastic noise
            mean_module: mean to be used. Default: zero mean
            covar_module: kernel to be used. Default: RBF

        Note:
            train_y has shape (n,), not (n, 1)!
        """
        likelihood = likelihood or gpytorch.likelihoods.GaussianLikelihood()

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module or gpytorch.means.ZeroMean()
        self.covar_module = covar_module or gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def covariance_between_points(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Get the posterior cross-covariance matrix between outputs at x1 and x2

        Args:
            x1: tensor of shape [n1, d]
            x2: tensor of shape [n2, d]

        Returns:
            A covariance tensor of shape [n1, n2]
        """
        n1 = x1.shape[-2]
        out = self.__call__(torch.cat([x1, x2], dim=-2))
        cross_covar = out.lazy_covariance_matrix[..., :n1, n1:]
        # The covariance terms are lazily evaluated, so this is a fairly efficient operation
        return cross_covar.evaluate()


def _map_to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array)  # TODO: Consider using as_tensor


def _map_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().numpy()


def _outputs_add_dim(y: torch.Tensor) -> np.ndarray:
    """Adds another dimension, to pass from GPyTorch to GPy/Emukit convention.

    Args:
        y: tensor, shape (n,)

    Returns:
        numpy array, shape (n, 1)
    """
    return _map_to_numpy(y)[:, None]


def _outputs_remove_dim(y: np.ndarray) -> torch.Tensor:
    """Removes the dimension, to pass from GPy/Emukit to GPyTorch convention.

    Args:
        y: array, shape (n, 1)

    Returns:
        tensor, shape (n,)
    """
    return _map_to_tensor(y.ravel())


class AxisReflectiveSymmetricKernel(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel: gpytorch.kernels.Kernel) -> None:
        """Use if function has symmetry f(x) = f(-x). The kernel takes the form

        .. math::
          K(x, x') = k(x, x') + k(-x, x')

        Args:
            base_covar: base kernel to be used (see `k` above)
        """
        super().__init__()
        self.base_kernel = base_kernel

    # def forward(self, x1, x2, **params):
    #     return (
    #         self.base_covar.__call__(x1, x2)
    #         + self.base_covar.__call__(-x1, x2)
    #     )

    def forward(self, x1, x2, last_dim_is_batch: bool = False, diag: bool = False, **params):
        orig_output1 = self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        orig_output2 = self.base_kernel.forward(-x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        if diag:
            return delazify(orig_output1) + delazify(orig_output2)
        else:
            return orig_output1 + orig_output2


# class SymmetricKernel(gpytorch.kernels.Kernel):
#     def __init__(self, base_covar: gpytorch.kernels.Kernel, transform: torch.Tensor) -> None:
#         """If function has symmetry f(x) = f(Ax), where A is some transformation, the kernel of the form

#         .. math::
#           K(x, x') = k(x, x') + k(Ax, x') + k(x, Ax') + k(Ax, Ax')

#         can be used.

#         Args:
#             base_covar: base kernel to be used (see `k` above)
#             transform: transformation (see `A` above)

#         Todo:
#             Specify possible shapes for `transform`
#             Think whether this will work on batches as well.
#         """
#         self.base_covar = base_covar
#         self.transform = torch.clone(transform)
#         self.transform.requires_grad = False

#     def _apply_map(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.matmul(self.transform, x)

#     def forward(self, x1, x2, **params):
#         Ax1, Ax2 = self._apply_map(x1), self._apply_map(x2)
#         return (
#             self.base_covar.forward(x1, x2)
#             + self.base_covar.forward(x1, Ax2)
#             + self.base_covar.forward(Ax1, x2)
#             + self.base_covar.forward(Ax1, Ax2)
#         )


class GPyTorchModelWrapper(IModel, IDifferentiable, IJointlyDifferentiable):
    """This is a thin wrapper around a GPModel (with GPyTorch backend) to allow users to plug models into Emukit."""

    def __init__(
        self,
        model: GPModel,
        n_restarts: int = 1,
        fast_pred_var: bool = False,
    ) -> None:
        """
        Args:
            model: GPModel
            n_restarts: How many restarts to do when optimizing hyperparameters
            fast_pred_var: Whether to use the NICE method to speed up covariance calculations
                (see: https://arxiv.org/pdf/1803.06058.pdf)
        """
        self._n_restarts = n_restarts
        self._fast_pred_var = fast_pred_var
        self._pred_context_manager = settings.fast_pred_var() if fast_pred_var else nullcontext()
        self.model: GPModel = model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            X: (n_points, input_dim) array containing locations at which to get predictions

        Returns:
            means: array of size (n_points, 1)
            variance: array of size (n_points, 1)
        """
        self.model.eval()
        with self._pred_context_manager:
            with torch.no_grad():
                test_x = _map_to_tensor(X)
                preds_latent = self.model(test_x)
                preds = self.model.likelihood(preds_latent)  # Add the noise (likelihood)
                return _outputs_add_dim(preds.mean), _outputs_add_dim(preds.variance)

    def predict_with_full_covariance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            X: (n_points, input_dim) array containing locations at which to get predictions

        Returns:
            means: array of size (n_points, 1)
            covariance: array of size (n_points, n_points)
        """
        self.model.eval()
        with self._pred_context_manager:
            with torch.no_grad():
                test_x = _map_to_tensor(X)
                preds_latent = self.model(test_x)
                preds = self.model.likelihood(preds_latent)  # Add the noise (likelihood)
                return _outputs_add_dim(preds.mean), _map_to_numpy(preds.covariance_matrix)

    def predict_noiseless(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            X: (n_points x n_dimensions) array containing locations at which to get predictions

        Returns:
            mean: size (n_points, 1)
            variance: size (n_points, 1)
        """
        self.model.eval()
        with self._pred_context_manager:
            with torch.no_grad():
                test_x = _map_to_tensor(X)
                preds_latent = self.model(test_x)
                # Return predictions of the latent stochastic process f (rather than the noisy values y)
                return _outputs_add_dim(preds_latent.mean), _outputs_add_dim(preds_latent.variance)

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            X: (n_points, n_dimensions) array containing locations at which to get gradient of the predictions

        Returns:
            mean gradient, shape (n_points, input_dim)
            variance gradient, shape (n_points, input_dim)
        """
        self.model.eval()

        with self._pred_context_manager:

            def mean_and_var(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                preds_latent = self.model(x)
                preds = self.model.likelihood(preds_latent)  # Add the noise (likelihood)
                return torch.sum(preds.mean), torch.sum(preds.variance)

            test_x = _map_to_tensor(X)
            mean_grad, var_grad = torch.autograd.functional.jacobian(mean_and_var, test_x)
        return _map_to_numpy(mean_grad), _map_to_numpy(var_grad)

    def get_joint_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes and returns model gradients of mean and full covariance matrix at given points

        Args:
            X: points to compute gradients at, nd array of shape (q, d)

        Returns:
            gradient of the mean of shape (q) at X with respect to X (return shape is (q, q, d)).
            gradient of the full covariance matrix of shape (q, q) at X with respect to X (return shape is (q, q, q, d))
        """
        self.model.eval()

        with self._pred_context_manager:

            def mean_and_cov(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                preds_latent = self.model(x)
                preds = self.model.likelihood(preds_latent)  # Add the noise (likelihood)
                return preds.mean, preds.covariance_matrix

            test_x = _map_to_tensor(X)
            mean_grad, cov_grad = torch.autograd.functional.jacobian(mean_and_cov, test_x)
        return _map_to_numpy(mean_grad), _map_to_numpy(cov_grad)

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Sets training data in model.

        Args:
            X: New training features
            Y: New training outputs
        """
        self.model.set_train_data(_map_to_tensor(X), _outputs_remove_dim(Y), strict=False)

    def optimize(self):
        """Optimizes model hyperparameters."""
        # Find optimal model hyperparameters
        self.model.train()
        self.model.likelihood.train()

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        train_x = _map_to_tensor(self.X)
        train_y = _outputs_remove_dim(self.Y)

        # Store parameters resulting from optimization and their corresponding marginal likelihoods
        ParameterDict = Dict[str, torch.Tensor]
        state_dicts_and_marginal_lik: List[Tuple[ParameterDict, float]] = []

        # Keep track of last exception raised during optimization
        last_error: Optional[gpytorch.utils.errors.NanError] = None

        for i in range(self._n_restarts):
            if i > 0:
                # On additional restarts (after 1st one), sample init. point for parameters heuristically
                self.model.initialize(
                    **{param_name: torch.randn_like(param) for param_name, param in self.model.named_parameters()}
                )

            try:
                fit_gpytorch_model(mll)
            except gpytorch.utils.errors.NanError as e:
                logging.warning("Hyperparameter optimization failed.")
                last_error = e
                continue
            # fit_gpytorch_model resets model back to eval mode. Put back in train mode
            self.model.train()
            self.model.likelihood.train()

            # Save the optimized params and compute the corresponding marginal likelihood
            with torch.no_grad():
                final_mll = mll(self.model(train_x), train_y).item()
            state_dicts_and_marginal_lik.append((_copy_state_dict(self.model.state_dict()), final_mll))

        if len(state_dicts_and_marginal_lik) == 0:
            raise ValueError("All optimization steps failed.") from last_error

        # Get the best model params. based on final marginal likelihood
        best_state_dict, best_mll = max(state_dicts_and_marginal_lik, key=itemgetter(1))

        # Load the best parameters
        self.model.load_state_dict(best_state_dict)
        self.model.initialize()

        # Set back to eval mode
        self.model.eval()
        self.model.likelihood.eval()
        return state_dicts_and_marginal_lik

    def predict_covariance(self, X: np.ndarray, with_noise: bool = True) -> np.ndarray:
        """Calculates posterior covariance between points in X.

        Args:
            X: Array of size n_points x n_dimensions containing input locations to compute posterior covariance at
            with_noise: not used at all, just to conform to the original API

        Returns:
            posterior covariance matrix of size n_points x n_points
        """
        _, cov = self.predict_with_full_covariance(X)
        return cov

    def get_covariance_between_points(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Calculate posterior covariance between two points

        Args:
            X1: An array of shape (n_points1, input_dim). It is the first argument of the
                posterior covariance function
            X2: An array of shape (n_points2, input_dim). This is the second argument of the
                posterior covariance function.

        Returns:
            array of shape (n_points1, n_points2) of posterior covariances between X1 and X2.
                Namely, [i, j]-th entry of the returned array will represent the posterior covariance
                between i-th point in X1 and j-th point in X2.

        Note:
            This function can operate in batch mode. In this case, X1 has shape (batch, n_points1, input_dim),
            X2 has shape (batch, n_points2, input_dim) and the returned matrix has shape (batch, n_points1, n_points2).
        """
        self.model.eval()
        self.model.covar_module.eval()

        with self._pred_context_manager:
            with torch.no_grad():
                x1, x2 = _map_to_tensor(X1), _map_to_tensor(X2)
                cross_cov = self.model.covariance_between_points(x1, x2)
                return _map_to_numpy(cross_cov)

    def get_covariance_between_points_gradients(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the derivative of the posterior covariance matrix between prediction at inputs x1 and x2
            with respect to X1.

        Args:
            X1: Prediction inputs of shape (n1, input_dim), where n1 is the number of samples
            X2: Prediction inputs of shape (n2, input_dim), where n2 is the number of samples

        Returns:
            array of shape (n1, n2, input_dim) representing the gradient of the posterior covariance between x1 and x2
                with respect to x1. res[i, j, k] is the gradient of Cov(y1[i], y2[j]) with respect to x1[i, k]

        Note:
            This function can operate in batch mode. In that case, X1 has shape (batch, n1, input_dim),
            X2 has shape (batch, n2, input_dim), and the returned array has shape (batch, n1, n2, input_dim).
        """
        self.model.eval()
        self.model.covar_module.eval()

        x2 = _map_to_tensor(X2)

        def cross_covar(x: torch.Tensor) -> torch.Tensor:
            with self._pred_context_manager:
                cross_cov = self.model.covariance_between_points(x, x2)
            # C'[j] = sum_a C[a, j], what doesn't affect the gradient with respect to the X[i, k],
            # but reduces the number of dimensions for computational efficiency.
            # The output shape is (n2,)
            reduce_dim = tuple(range(cross_cov.dim() - 1))
            return torch.sum(cross_cov, dim=reduce_dim)

        x1 = _map_to_tensor(X1)
        # This tensor has shape (n2, n1, d), need to transpose it
        cross_cov_grad = torch.autograd.functional.jacobian(cross_covar, x1)
        return self._transpose_cross_cov_grad(_map_to_numpy(cross_cov_grad))

    @staticmethod
    def _transpose_cross_cov_grad(a: np.ndarray) -> np.ndarray:
        dims = len(a.shape)
        permutation = list(range(1, dims + 1))
        permutation[dims - 1] = dims - 1
        permutation[dims - 2] = 0
        return np.transpose(a, permutation)

    @property
    def X(self) -> np.ndarray:
        """An array of shape (n_points, input_dim) containing training inputs."""
        train_x = self.model.train_inputs
        if isinstance(train_x, tuple):
            train_x = train_x[0]

        return _map_to_numpy(train_x)

    @property
    def Y(self) -> np.ndarray:
        """An array of shape (n_points, 1) containing training outputs."""
        return _outputs_add_dim(self.model.train_targets)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model}, n_restarts={self._n_restarts})"

    def __str__(self) -> str:
        return repr(self)


def _copy_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((param_name, param.detach().clone()) for param_name, param in state_dict.items())


def get_constrained_named_parameters(gpytorch_model: gpytorch.models.GP) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Return a list of parameter names and parameter values after passing through a constraint
    function (rather than raw values) for a given GPyTorch model.
    """

    def get_constrained_named_parameter(
        param_name: str, raw_param: torch.nn.Parameter, param_constraint: gpytorch
    ) -> Tuple[str, torch.Tensor]:
        return (
            param_name + "_constrained",
            param_constraint.transform(raw_param) if param_constraint is not None else raw_param,
        )

    yield from (
        get_constrained_named_parameter(param_name, raw_param, param_constraint)
        for param_name, raw_param, param_constraint in gpytorch_model.named_parameters_and_constraints()
    )
