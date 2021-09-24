# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Helper functions for extracting and storing useful information for analysis
"""
import itertools
import numpy as np
import pandas as pd
from emukit.core.interfaces import IModel
from emukit.model_wrappers import GPyModelWrapper
from functools import singledispatch

from gp.model_wrapper import GPyTorchModelWrapper, get_constrained_named_parameters


@singledispatch
def get_model_hyperparam(model: IModel) -> pd.DataFrame:
    raise NotImplementedError(f"Can't get parameters for type {model}")


@get_model_hyperparam.register
def _(model: GPyModelWrapper) -> np.ndarray:
    parameter_names_list = [name.split(".", 1)[1] for name in model.model.parameter_names_flat(include_fixed=True)]
    return pd.DataFrame(model.model.param_array[None, :], columns=parameter_names_list)


@get_model_hyperparam.register
def _(model: GPyTorchModelWrapper) -> np.ndarray:
    # Save both raw (unconstrained) parameters, and the constrained counter-equivalents
    params_dict = {
        param_name: [param.item()]
        for param_name, param in itertools.chain(
            model.model.named_parameters(), get_constrained_named_parameters(model.model)
        )
    }
    return pd.DataFrame(params_dict)
