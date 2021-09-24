# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
A collection of various functions pulled from jupyter notebooks with inital experiments.
TODO: these should all either be put into correct files within a proper directory structure, or deleted.
"""
import numpy as np
import GPy
from GPy.models import GPRegression

from emukit.bayesian_optimization.acquisitions.expected_improvement import MultipointExpectedImprovement
from emukit.model_wrappers import GPyModelWrapper


# Modelling
def make_additive_kernel(lengthscales=(1.0, 1.0), variances=(1.0, 1.0), fix_lengthscales=False, fix_variances=False):
    kern0 = GPy.kern.RBF(input_dim=1, active_dims=[0], lengthscale=lengthscales[0], variance=variances[0])
    kern1 = GPy.kern.RBF(input_dim=1, active_dims=[1], lengthscale=lengthscales[1], variance=variances[1])
    if fix_lengthscales:
        kern0.lengthscale.fix()
        kern1.lengthscale.fix()
    if fix_variances:
        kern0.variance.fix()
        kern1.variance.fix()
    additive_kernel = kern0 + kern1
    return additive_kernel


# Model:
def make_model(
    X,
    Y,
    lengthscales=(1.0, 1.0),
    variances=(1.0, 1.0),
    gaussian_noise=0.0,
    fix_lengthscales=False,
    fix_variances=False,
    fix_noise=True,
):
    additive_kernel = make_additive_kernel(
        lengthscales=lengthscales, variances=variances, fix_lengthscales=fix_lengthscales, fix_variances=fix_variances
    )
    gpy_model = GPRegression(X, Y, additive_kernel)
    if fix_noise:
        gpy_model.Gaussian_noise.fix(gaussian_noise)
    model = GPyModelWrapper(gpy_model)
    return model


def sample_from_prior(x, kernel):
    mu = np.zeros([x.shape[0]])  # vector of the means
    C = kernel.K(x, x)  # covariance matrix
    # Generate a sample path with mean mu and covariance C
    Z = np.random.multivariate_normal(mu, C)
    return Z


# Sequential exact multipoint implementation very slow
class SequentialMultipointExpectedImprovement(MultipointExpectedImprovement):
    def __init__(
        self, prev_x: np.ndarray, model, jitter: float = 0.0, fast_compute: bool = False, eps: float = 1e-3
    ) -> None:
        super().__init__(model, jitter, fast_compute, eps)
        self.prev_x = prev_x

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the multipoint Expected Improvement.
        :param x: points where the acquisition is evaluated.
        :return: multipoint Expected Improvement at the input.
        """
        acq = np.zeros([x.shape[0], 1])
        for i in range(x.shape[0]):
            x_batch = np.concatenate((self.prev_x, x[None, i]), axis=0)
            acq[i] = -super().evaluate(x_batch)
        if acq.shape[0] == 1:
            return acq[0, 0]
        else:
            return acq

    def evaluate_with_gradients(self, x: np.ndarray):
        """
        Computes the multipoint Expected Improvement and its derivative.
        :param x: locations where the evaluation with gradients is done.
        :return: multipoint Expected Improvement and its gradient at the input.
        """
        NotImplementedError()
