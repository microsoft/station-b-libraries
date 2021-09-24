# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import numpy as np
import pytest

import GPy
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

import gp.model_wrapper as mw  # GPyTorch model wrapper


@pytest.fixture
def train_x() -> np.ndarray:
    np.random.seed(31415926)
    return np.random.rand(10, 2)


@pytest.fixture
def train_y(train_x: np.ndarray) -> np.ndarray:
    return np.sum(np.cos(train_x), axis=1).reshape((-1, 1))


@pytest.fixture
def test_x() -> np.ndarray:
    np.random.seed(10)
    return np.random.rand(7, 2)


@pytest.fixture
def test_x2() -> np.ndarray:
    np.random.seed(100)
    return np.random.rand(15, 2)


@pytest.fixture
def wrapper(train_x: np.ndarray, train_y: np.ndarray) -> mw.GPyTorchModelWrapper:
    train_x = mw._map_to_tensor(train_x)
    train_y = mw._outputs_remove_dim(train_y)

    model = mw.GPModel(train_x, train_y)
    model.covar_module.base_kernel.lengthscale = 1.0
    model.covar_module.outputscale = 1.0
    model.likelihood.noise = 1e-3
    return mw.GPyTorchModelWrapper(model)


@pytest.fixture
def gpy_wrapper(train_x: np.ndarray, train_y: np.ndarray) -> GPyModelWrapper:
    model = GPy.models.GPRegression(train_x, train_y)
    model.kern.variance = 1.0
    model.kern.lengthscale = 1.0
    model.Gaussian_noise.variance.constrain_fixed(1e-3)

    wrapper = GPyModelWrapper(model)
    return wrapper


def test_init(wrapper: mw.GPyTorchModelWrapper, train_x: np.ndarray, train_y: np.ndarray) -> None:
    assert isinstance(wrapper, mw.GPyTorchModelWrapper)
    np.testing.assert_allclose(wrapper.X, train_x)
    np.testing.assert_allclose(wrapper.Y, train_y)


def test_set_data(wrapper: mw.GPyTorchModelWrapper) -> None:
    np.random.seed(10)
    x = np.random.rand(20, 2)
    y = np.sin(x)[:, 0].reshape((-1, 1))
    wrapper.set_data(x, y)
    np.testing.assert_allclose(wrapper.X, x)
    np.testing.assert_allclose(wrapper.Y, y)


def test_optimize_smoke(wrapper: mw.GPyTorchModelWrapper) -> None:
    wrapper.optimize()


class TestShapes:
    """Simple tests"""

    def test_predict(self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray) -> None:
        n = len(test_x)
        means, var = wrapper.predict(test_x)
        assert isinstance(means, np.ndarray)
        assert isinstance(var, np.ndarray)
        assert means.shape == (n, 1) == var.shape

    def test_predict_with_full_covariance(self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray) -> None:
        n = len(test_x)
        means, cov = wrapper.predict_with_full_covariance(test_x)
        assert isinstance(means, np.ndarray)
        assert isinstance(cov, np.ndarray)
        assert cov.shape == (n, n)

    def test_get_prediction_gradients(self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray) -> None:
        n, input_dim = test_x.shape
        mean_grad, var_grad = wrapper.get_prediction_gradients(test_x)
        assert mean_grad.shape == var_grad.shape == (n, input_dim)

    def test_get_joint_prediction_gradients(self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray) -> None:
        n, input_dim = test_x.shape
        mean_joint_grad, cov_joint_grad = wrapper.get_joint_prediction_gradients(test_x)
        assert mean_joint_grad.shape == (n, n, input_dim)
        assert cov_joint_grad.shape == (n, n, n, input_dim)

    def test_get_covariance_between_points(
        self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray, test_x2: np.ndarray
    ) -> None:
        cross_cov = wrapper.get_covariance_between_points(test_x, test_x2)

        assert isinstance(cross_cov, np.ndarray)
        assert cross_cov.shape == (len(test_x), len(test_x2))

    def test_get_covariance_between_points_gradients(
        self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray, test_x2: np.ndarray
    ) -> None:
        n1, d = test_x.shape
        n2 = test_x2.shape[0]

        cross_cov_grad = wrapper.get_covariance_between_points_gradients(test_x, test_x2)
        assert isinstance(cross_cov_grad, np.ndarray)
        assert cross_cov_grad.shape == (n1, n2, d)


class TestInternalConsistency:
    """Many functions in the API calculate the same thing. This test suite checks if they are consistent."""

    def test_predict_with_full_covariance(self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray) -> None:
        means, var = wrapper.predict(test_x)
        means_fc, covar = wrapper.predict_with_full_covariance(test_x)

        np.testing.assert_allclose(means, means_fc)
        np.testing.assert_allclose(np.diag(covar), var.ravel())

        cov = wrapper.predict_covariance(test_x)
        np.testing.assert_allclose(cov, covar)

    def test_get_covariance_between_points__same_as_full_covariance(
        self,
        wrapper: mw.GPyTorchModelWrapper,
        test_x: np.ndarray,
        test_x2: np.ndarray,
    ):
        cross_covar = wrapper.get_covariance_between_points(test_x, test_x2)
        _, full_covar = wrapper.predict_with_full_covariance(np.concatenate((test_x, test_x2), axis=-2))

        n1 = test_x.shape[-2]  # num. points in test_x
        # The [:n1, n1:] sub-matrix of full-covariance should be the same as cross-covariance
        assert pytest.approx(cross_covar, abs=1e-12, rel=1e-10) == full_covar[:n1, n1:]

    def test_get_covariance_between_points_gradients__same_as_full_covariance(
        self,
        wrapper: mw.GPyTorchModelWrapper,
        test_x: np.ndarray,
        test_x2: np.ndarray,
    ):
        cross_covar_grad = wrapper.get_covariance_between_points_gradients(test_x, test_x2)
        _, full_cov_grad = wrapper.get_joint_prediction_gradients(np.concatenate((test_x, test_x2), axis=-2))

        n1 = test_x.shape[-2]  # num. points in test_x
        # The gradient of the right submatrix of full covariance wrt to first n1 points of the passed
        # set of points should be the same as gradient of cross-covariance
        cross_cov_grad_from_full_cov_grad = full_cov_grad[:n1, n1:, :n1, :]
        cross_cov_grad_from_full_cov_grad = np.diagonal(cross_cov_grad_from_full_cov_grad, axis1=0, axis2=2)
        cross_cov_grad_from_full_cov_grad = cross_cov_grad_from_full_cov_grad.transpose((2, 0, 1))
        assert pytest.approx(cross_covar_grad, abs=1e-12, rel=1e-10) == cross_cov_grad_from_full_cov_grad

    def test_get_covariance_between_points__invariant_to_set_swap(
        self,
        wrapper: mw.GPyTorchModelWrapper,
        test_x: np.ndarray,
        test_x2: np.ndarray,
    ):
        # cross-covariance between X1 and X2 should be the transpose of cross-covariance between X2 and X1
        cross_covar1 = wrapper.get_covariance_between_points(test_x, test_x2)
        cross_covar2 = wrapper.get_covariance_between_points(test_x2, test_x)
        assert pytest.approx(cross_covar1, rel=1e-12, abs=1e-12) == cross_covar2.T


class TestBatching:
    """Tests if adding a batch dimension works as expected."""

    @pytest.mark.parametrize("batch_size", (2, 5, 10))
    def test_predict_covariance(self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray, batch_size: int) -> None:
        np.random.seed(20)

        n_points, input_dim = test_x.shape
        test_x_batched = np.random.rand(batch_size, n_points, input_dim)

        cov_batched = wrapper.predict_covariance(test_x_batched)

        assert isinstance(cov_batched, np.ndarray)
        assert cov_batched.shape == (batch_size, n_points, n_points)

        cov_naive = np.asarray([wrapper.predict_covariance(arr) for arr in test_x_batched])
        np.testing.assert_allclose(cov_batched, cov_naive)

    @pytest.mark.parametrize("batch_size", (5, 10))
    @pytest.mark.parametrize("n_points2", (3, 20))
    def test_get_covariance_between_points(
        self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray, batch_size: int, n_points2: int
    ) -> None:
        np.random.seed(20)

        n_points1, input_dim = test_x.shape
        test_x1_batched = np.random.rand(batch_size, n_points1, input_dim)
        test_x2_batched = np.random.rand(batch_size, n_points2, input_dim)

        cross_cov = wrapper.get_covariance_between_points(test_x1_batched, test_x2_batched)
        assert isinstance(cross_cov, np.ndarray)
        assert cross_cov.shape == (batch_size, n_points1, n_points2)

        cross_cov_naive = np.asarray(
            [wrapper.get_covariance_between_points(x1, x2) for x1, x2 in zip(test_x1_batched, test_x2_batched)]
        )
        np.testing.assert_allclose(cross_cov, cross_cov_naive)

    @pytest.mark.parametrize("batch_size", (5, 10))
    @pytest.mark.parametrize("n_points2", (3, 20))
    def test_get_covariance_between_points_gradients(
        self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray, batch_size: int, n_points2: int
    ) -> None:
        np.random.seed(20)

        n_points1, input_dim = test_x.shape
        test_x1_batched = np.random.rand(batch_size, n_points1, input_dim)
        test_x2_batched = np.random.rand(batch_size, n_points2, input_dim)

        print(f"batch: {batch_size}, n1: {n_points1}, n2: {n_points2}, d: {input_dim}")

        cross_cov_grad = wrapper.get_covariance_between_points_gradients(test_x1_batched, test_x2_batched)
        assert isinstance(cross_cov_grad, np.ndarray)
        assert cross_cov_grad.shape == (batch_size, n_points1, n_points2, input_dim)

        cross_cov_grad_naive = np.asarray(
            [
                wrapper.get_covariance_between_points_gradients(x1, x2)
                for x1, x2 in zip(test_x1_batched, test_x2_batched)
            ]
        )
        np.testing.assert_allclose(cross_cov_grad, cross_cov_grad_naive)


class TestEmukitPorted:
    """Tests ported from Emukit's GPyModelWrapper."""

    epsilon = 1e-5

    def test_joint_prediction_gradients(self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray):
        wrapper.optimize()

        mean, cov = wrapper.predict_with_full_covariance(test_x)
        # Get the gradients
        mean_dx, cov_dx = wrapper.get_joint_prediction_gradients(test_x)

        for i in range(test_x.shape[0]):  # Iterate over each test point
            for j in range(test_x.shape[1]):  # Iterate over each dimension
                # Approximate the gradient numerically
                perturbed_input = test_x.copy()
                perturbed_input[i, j] += self.epsilon
                mean_perturbed, cov_perturbed = wrapper.predict_with_full_covariance(perturbed_input)
                mean_dx_numerical = (mean_perturbed - mean) / self.epsilon
                cov_dx_numerical = (cov_perturbed - cov) / self.epsilon
                # Check that numerical approx. similar to true gradient
                assert pytest.approx(mean_dx_numerical.ravel(), abs=1e-8, rel=1e-2) == mean_dx[:, i, j]
                assert pytest.approx(cov_dx_numerical, abs=1e-8, rel=1e-2) == cov_dx[:, :, i, j]

    def test_get_covariance_between_points_gradients(
        self, wrapper: mw.GPyTorchModelWrapper, test_x: np.ndarray, test_x2: np.ndarray
    ):
        wrapper.optimize()

        cov = wrapper.get_covariance_between_points(test_x, test_x2)
        # Get the gradients
        cov_dx = wrapper.get_covariance_between_points_gradients(test_x, test_x2)

        for i in range(test_x.shape[0]):  # Iterate over each test point
            for j in range(test_x.shape[1]):  # Iterate over each dimension
                # Approximate the gradient numerically
                perturbed_input = test_x.copy()
                perturbed_input[i, j] += self.epsilon
                cov_perturbed = wrapper.get_covariance_between_points(perturbed_input, test_x2)
                cov_dx_numerical = (cov_perturbed[i] - cov[i]) / self.epsilon
                # Check that numerical approx. similar to true gradient
                assert pytest.approx(cov_dx_numerical, abs=1e-8, rel=1e-2) == cov_dx[i, :, j]


class TestAgainstGPy:
    """As we trust the implementation with GPy backend more, we check whether this new implementation provides
    consistent answers."""

    def test_predict(self, wrapper: mw.GPyTorchModelWrapper, gpy_wrapper: GPyModelWrapper, test_x: np.ndarray) -> None:
        mean1, var1 = wrapper.predict(test_x)
        mean2, var2 = gpy_wrapper.predict(test_x)

        np.testing.assert_allclose(mean1, mean2)
        np.testing.assert_allclose(var1, var2, rtol=1e-3)

    def test_predict_with_full_covariance(
        self, wrapper: mw.GPyTorchModelWrapper, gpy_wrapper: GPyModelWrapper, test_x: np.ndarray
    ) -> None:
        mean1, cov1 = wrapper.predict_with_full_covariance(test_x)
        mean2, cov2 = gpy_wrapper.predict_with_full_covariance(test_x)

        np.testing.assert_allclose(mean1, mean2, rtol=1e-3)
        np.testing.assert_allclose(cov1, cov2, rtol=1e-3)

    def test_get_prediction_gradients(
        self, wrapper: mw.GPyTorchModelWrapper, gpy_wrapper: GPyModelWrapper, test_x: np.ndarray
    ) -> None:
        mean_grad1, var_grad1 = wrapper.get_prediction_gradients(test_x)
        mean_grad2, var_grad2 = gpy_wrapper.get_prediction_gradients(test_x)

        np.testing.assert_allclose(mean_grad1, mean_grad2, rtol=1e-3)
        np.testing.assert_allclose(var_grad1, var_grad2, rtol=1e-3)

    def test_get_covariance_between_points_gradients(
        self,
        wrapper: mw.GPyTorchModelWrapper,
        gpy_wrapper: GPyModelWrapper,
        test_x: np.ndarray,
        test_x2: np.ndarray,
    ) -> None:
        xcov_grad1 = wrapper.get_covariance_between_points_gradients(test_x, test_x2)
        xcov_grad2 = gpy_wrapper.get_covariance_between_points_gradients(test_x, test_x2)

        np.testing.assert_allclose(xcov_grad1, xcov_grad2, rtol=1e-3)
