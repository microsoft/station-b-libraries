# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import TypeVar
import torch
import numpy as np


ArrayLike = TypeVar("ArrayLike", torch.Tensor, np.ndarray)


def calc_cum_min_mean(
    mean: ArrayLike, cum_min_prev_mean: ArrayLike, alpha_pdf: ArrayLike, alpha_cdf: ArrayLike, beta_sqrt: ArrayLike
) -> ArrayLike:
    return cum_min_prev_mean * (1 - alpha_cdf) + mean * alpha_cdf - beta_sqrt * alpha_pdf


def calc_cum_min_uncentered_2nd_moment(
    mean: ArrayLike,
    variance: ArrayLike,
    cum_min_prev_mean: ArrayLike,
    cum_min_prev_var: ArrayLike,
    alpha_pdf: ArrayLike,
    alpha_cdf: ArrayLike,
    beta_sqrt: ArrayLike,
) -> ArrayLike:
    return (
        (cum_min_prev_mean ** 2 + cum_min_prev_var) * (1 - alpha_cdf)
        + (mean ** 2 + variance) * alpha_cdf
        - (cum_min_prev_mean + mean) * beta_sqrt * alpha_pdf
    )


def calc_cum_min_var(
    mean: ArrayLike,
    variance: ArrayLike,
    cum_min_mean: ArrayLike,
    cum_min_prev_mean: ArrayLike,
    cum_min_prev_var: ArrayLike,
    alpha_pdf: ArrayLike,
    alpha_cdf: ArrayLike,
    beta_sqrt: ArrayLike,
) -> ArrayLike:
    cum_min_uncentered_2nd_moment = calc_cum_min_uncentered_2nd_moment(
        mean=mean,
        variance=variance,
        cum_min_prev_mean=cum_min_prev_mean,
        cum_min_prev_var=cum_min_prev_var,
        alpha_pdf=alpha_pdf,
        alpha_cdf=alpha_cdf,
        beta_sqrt=beta_sqrt,
    )
    cum_min_var = cum_min_uncentered_2nd_moment - cum_min_mean ** 2
    return cum_min_var
