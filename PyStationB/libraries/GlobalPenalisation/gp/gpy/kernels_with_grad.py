# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import GPy
import numpy as np
from functools import reduce


class Add(GPy.kern.Add):
    def dK_dX(self, X: np.ndarray, X2: np.ndarray, dim: int) -> np.ndarray:
        return reduce(lambda d1, d2: d1 + d2, (p.dK_dX(X, X2, dim) for p in self.parts))


class Symmetric(GPy.kern.Symmetric):
    def dK_dX(self, X: np.ndarray, X2: np.ndarray, dim: int) -> np.ndarray:
        X_sym = X.dot(self.transform)
        X2_sym = X2.dot(self.transform)
        return (
            self.base_kernel.dK_dX(X, X2, dim)
            + self.symmetry_sign * self.base_kernel.dK_dX(X_sym, X2, dim)
            + self.symmetry_sign * self.base_kernel.dK_dX(X, X2_sym, dim)
            + self.base_kernel.dK_dX(X_sym, X2_sym, dim)
        )

    def gradients_X_diag(self, dL_dK_diag: np.ndarray, X: np.ndarray) -> np.ndarray:
        dL_dK_full = np.diag(dL_dK_diag)
        X_sym = X.dot(self.transform)

        # Calculate gradient for k(Ax, Ax')
        gradient = self.base_kernel.gradients_X_diag(dL_dK_diag, X_sym)

        # Calculate gradient for k(x, x')
        gradient += self.base_kernel.gradients_X_diag(dL_dK_diag, X_sym)

        gradient += self.symmetry_sign * self.base_kernel.gradients_X(dL_dK_full, X, X_sym)
        gradient += self.symmetry_sign * self.base_kernel.gradients_X(dL_dK_full, X_sym, X)

        return gradient

    def gradients_X(self, dL_dK, X, X2):
        X_sym = X.dot(self.transform)
        if X2 is None:
            X2 = X
            X2_sym = X.dot(self.transform)
            dL_dK = dL_dK + dL_dK.T
        else:
            X2_sym = X2.dot(self.transform)

        return (
            self.base_kernel.gradients_X(dL_dK, X, X2)
            + self.base_kernel.gradients_X(dL_dK, X_sym, X2_sym).dot(self.transform.T)
            + self.symmetry_sign * self.base_kernel.gradients_X(dL_dK, X, X2_sym)
            + self.symmetry_sign * self.base_kernel.gradients_X(dL_dK, X_sym, X2).dot(self.transform.T)
        )
