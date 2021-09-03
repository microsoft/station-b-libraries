# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import random

import pytest
from abex import settings
from abex.expand import FixedSeed, expand_structure, generate_resolutions, get_dimensions, increment_resolution
from psbutils.misc import find_subrepo_directory
from pytest import approx


def test_simple():
    assert list(expand_structure("foo")) == ["foo"]


def test_choice_list():
    data = ["@x", 2, 4, 6]
    assert sorted(expand_structure(data)) == [2, 4, 6]


def test_short_lists():
    # A choice list must have at least two elements
    data0 = ["@"]
    assert sorted(expand_structure(data0)) == [data0]
    data1 = ["@", 1]
    assert sorted(expand_structure(data1)) == [data1]


def test_repeated_dimension():
    # Same dimension name, so 1 is selected with 4, 2 with 5, and 3 with 6.
    data = {"foo": ["@a", 1, 2, 3], "bar": ["@a", 4, 5, 6]}
    results = list(expand_structure(data))
    assert len(results) == 3
    for res in results:
        assert res["bar"] - res["foo"] == 3


def test_named_independent_dimensions():
    # Different variable names, so 3x3=9 results
    data = {"foo": ["@a", 1, 2, 3], "bar": ["@b", 4, 5, 6]}
    results = list(expand_structure(data))
    assert len(results) == 9


def test_length_mismatch():
    # Dimension "a" cannot be both size 3 and size 2
    with pytest.raises(ValueError):
        data = {"foo": ["@a", 1, 2, 3], "bar": ["@a", 4, 5]}
        list(expand_structure(data))


def test_uniform():
    data = ["@x", "#uniform#", 5, 1, 3]
    results = sorted(expand_structure(data))
    # 5 points from the uniform distribution on [1,3], equally spaced, with half
    # a space at each end.
    assert results == approx([1.2, 1.6, 2.0, 2.4, 2.8])


def test_normal():
    data = ["@x", "#normal#", 5, 0, 1]
    results = sorted(expand_structure(data))
    # 5 points from the standard normal distribution.
    assert len(results) == 5
    assert results[0] == approx(-results[4])
    assert results[1] == approx(-results[3])
    assert results[2] == approx(0.0)


def test_unknown_distribution():
    data = ["@x", "#foobar#", 5, 0, 1]
    with pytest.raises(ValueError):
        list(expand_structure(data))


def test_bad_param_count():
    data = ["@x", "#uniform#", 5, 1, 3, 3.14]
    with pytest.raises(ValueError):
        list(expand_structure(data))


def test_load_resolutions_for_expansion():
    yml_path = find_subrepo_directory() / "tests/data/Specs/two_kernels.yml"
    indexed_configs = list(settings.load_resolutions(str(yml_path)))
    # Check we get two configs, differing only in their values of model.kernel and results_dir
    assert len(indexed_configs) == 2
    assert len(indexed_configs[0]) == 1  # single seed value
    assert len(indexed_configs[1]) == 1  # single seed value
    config0 = indexed_configs[0][0][1]  # first resolution, single seed, config
    config1 = indexed_configs[1][0][1]  # second resolution, single seed, config
    assert sorted([config0.model.kernel, config1.model.kernel]) == ["Matern", "RBF"]
    assert sorted([config0.resolution_spec, config1.resolution_spec]) == ["k1", "k2"]
    assert config0.results_dir != config1.results_dir
    assert config0.data.folder == config1.data.folder
    assert config0.data.simulation_folder != config1.data.simulation_folder
    config0.model.kernel = config1.model.kernel
    config0.data.simulation_folder = config1.data.simulation_folder
    config0.results_dir = config1.results_dir
    config0.resolution_spec = config1.resolution_spec
    assert config0 == config1


def test_fixed_seed():
    # With the same initial seed, calling randint inside a FixedSeed does not affect what is generated afterwards.
    random.seed(123)
    before1 = random.randint(0, 1000)
    after1 = random.randint(0, 1000)
    random.seed(123)
    before2 = random.randint(0, 1000)
    with FixedSeed(3):
        inside2 = random.randint(0, 1000)
    after2 = random.randint(0, 1000)
    assert (before1, after1) == (before2, after2)
    # With different initial seeds, we get the same value inside FixedSeed with the same argument.
    random.seed(124)
    with FixedSeed(3):
        inside3 = random.randint(0, 1000)
    assert inside2 == inside3
    random.seed(123)
    # With the same initial seeds but different FixedSeed, we get different values.
    with FixedSeed(4):
        inside4 = random.randint(0, 1000)
    assert inside4 != inside2


def test_generate_selections():
    # Check exhaustive generation produces results in the expected order
    assert list(generate_resolutions((2, 3))) == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    # Check that when we have max_selections, we get the same (random) choice on two occasions
    sel1 = list(generate_resolutions((100, 100, 100), max_resolutions=10))
    sel2 = list(generate_resolutions((100, 100, 100), max_resolutions=10))
    assert len(sel1) == 10
    assert sel1 == sel2


def test_increment_selection():
    assert increment_resolution([1], (2,)) == [2]
    assert increment_resolution([2], (2,)) is None
    assert increment_resolution([1, 2], (2, 3)) == [1, 3]
    assert increment_resolution([1, 3], (2, 3)) == [2, 1]


def test_get_dimensions():
    struc = {"foo": ["@a", 11, 22], "bar": ["@b", 3, 4, 5]}
    dim_dict = {}
    get_dimensions(struc, dim_dict)
    assert dim_dict == {"a": 2, "b": 3}
