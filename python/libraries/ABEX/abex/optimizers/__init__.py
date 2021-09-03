# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""A submodule with different optimization strategies.

This submodule provides the access to (sub)submodules containing the `run` method, which takes the config and returns
a batch of new samples. (See `run` methods of any of these submodules for more details).

Currently the following classes are available:
    BayesOptimizer, a high-level wrapper over abex.bayesopt, which builds GP model and does batch Bayesian optimization
    ZoomOptimizer, a method of "zooming in" and sampling such subspace in quite uniform way
"""
import abex.optimizers.bayes_optimizer as bayes  # noqa: F401
import abex.optimizers.zoom_optimizer as zoom  # noqa: F401
