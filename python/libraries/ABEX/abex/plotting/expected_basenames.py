# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from psbutils.misc import flatten_list  # pragma: no cover


def expected_basenames_1d(n_folds, variant):  # pragma: no cover
    lim = n_folds + 1
    if variant == 2:
        sub = ["binned_slices.png", "model_priors.png"] + [f"binned_slices__fold{n}.png" for n in range(1, lim)]
    elif variant == 1:
        sub = ["config.yml"] + [f"hmc_samples_fold{n}.png" for n in range(0, lim)]
    else:
        raise ValueError(f"variant must be 1 or 2, not {variant}")
    lst = [
        ["acquisition1d.png"],
        [f"acquisition1d__fold{n}.png" for n in range(1, lim)],
        ["batch.csv", "batch_predicted_objective.csv", "bo_distance.png", "bo_experiment.png"],
        [f"calibration_fold{n}.png" for n in range(1, lim)],
        ["calibration_train_only.png"],
        sub,
        ["model_parameters.csv", "optima.csv", "optima.png", "slice1d.png"],
        [f"optima_cross-validation_{n_folds}.{suf}" for suf in ["csv", "png"]],
        [f"slice1d__fold{n}.png" for n in range(1, lim)],
        [f"train_test_fold{n}.png" for n in range(1, lim)],
        ["train_only.png", "training.csv", "xval_test.png"],
    ]
    return sorted(flatten_list(lst))


def expected_basenames_2d(n_folds, variant):  # pragma: no cover
    lim = n_folds + 1
    lst = [
        ["acquisition2d_original_space.png"],
        [f"acquisition2d_original_space__fold{n}.png" for n in range(1, lim)],
        ["slice2d.png"],
        [f"slice2d__fold{n}.png" for n in range(1, lim)],
        ["slice2d_original_space.png"],
        [f"slice2d_original_space__fold{n}.png" for n in range(1, lim)],
    ]
    return sorted(expected_basenames_1d(n_folds, variant) + flatten_list(lst))
