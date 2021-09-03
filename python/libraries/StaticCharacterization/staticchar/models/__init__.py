# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Growth models.

Todo:
    Consider adding MCMC inference of parameters, e.g. by emcee or pymc3 package.
"""
from .base import BaseModel, CurveParameters  # noqa
from .gompertz import GompertzModel  # noqa
from .logistic import LogisticModel  # noqa
