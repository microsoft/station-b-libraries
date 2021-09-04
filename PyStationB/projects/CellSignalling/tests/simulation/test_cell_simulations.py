# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from unittest import mock

import numpy as np
import pytest
from cellsig_sim.simulations import FourInputCell, LatentCell, ThreeInputCell

np.random.seed(42)


class TestLatentCell:
    @pytest.fixture(autouse=True)
    def _valid_inputs(self):
        bounds = LatentCell().parameter_space.get_bounds()
        self.valid_batch_of_inputs = np.stack([np.random.uniform(low, upp, 10) for low, upp in bounds], axis=1)

    @pytest.fixture(autouse=True)
    def _invalid_inputs(self):
        self.invalid_batch_of_inputs = np.array([[0, 3, 100, 140], [9, 1e-7, 500, 4000], [7, 1, 3e18, 100]])

    def test_signal_to_crosstalk_ratio_positive(self):
        obj = LatentCell().sample_objective(self.valid_batch_of_inputs)
        assert np.all(obj > 0)

    @mock.patch.object(LatentCell, "_production_rate_cfp", return_value=1 * np.ones([3]))
    @mock.patch.object(LatentCell, "_production_rate_yfp", return_value=2 * np.ones([3]))
    def test_sample_objective_returns_expected(self, mock_prod_rate_yfp, mock_prod_rate_cfp):
        expected_crosstalk_ratio = np.ones([3])  # Because production_rate_xx methods are mocked
        crosstalk_ratio = LatentCell().sample_objective(self.valid_batch_of_inputs)
        assert np.all(crosstalk_ratio == expected_crosstalk_ratio)

    @mock.patch.object(LatentCell, "_production_rate_cfp", return_value=1.0)
    @mock.patch.object(LatentCell, "_production_rate_yfp", return_value=2.0)
    def test_signal_to_crosstalk_ratio_returns_expected(self, mock_prod_rate_yfp, mock_prod_rate_cfp):
        expected_crosstalk_ratio = 1.0  # Because production_rate_xx methods are mocked
        crosstalk_ratio = LatentCell()._signal_to_crosstalk_ratio(
            luxr=3, lasr=2, c12_on=0.3, c6_on=0.5  # type: ignore # auto
        )  # type: ignore # auto
        assert crosstalk_ratio == expected_crosstalk_ratio

    def test_inputs_outside_parameter_space_raise_error(self):
        with pytest.raises(ValueError):
            LatentCell().sample_objective(self.invalid_batch_of_inputs)


class TestFourInputCell:
    @pytest.fixture(autouse=True)
    def _valid_inputs(self):
        bounds = FourInputCell().parameter_space.get_bounds()
        self.valid_batch_of_inputs = np.stack([np.random.uniform(low, upp, 10) for low, upp in bounds], axis=1)

    @pytest.fixture(autouse=True)
    def _invalid_inputs(self):
        self.invalid_batch_of_inputs = np.array(
            [
                [1e-6, 1e-6, 1e-6, 1e-6],
                [140, 5, 1e3, 1e3],
                [10, 20, 1e3, 1e3],
                [10, 5, 3e5, 1e3],
                [10, 5, 1e3, 3e5],
            ]
        )

    def test_signal_to_crosstalk_ratio_positive(self):
        inputs = self.valid_batch_of_inputs
        obj1 = FourInputCell(use_growth_in_objective=True).sample_objective(inputs)
        assert np.all(obj1 > 0)

        obj2 = FourInputCell(use_growth_in_objective=False).sample_objective(inputs)
        assert np.all(obj2 > 0)

    def test_returns_expected(self):
        direct_cell = FourInputCell(use_growth_in_objective=False)
        latent_cell = LatentCell()

        # Inputs in the "latent" LuxR/LasR space.
        inputs = np.array(
            [
                [1, 1, 200, 120],
                [0.3, 2, 200, 900],
                [0.1, 3, 700, 700],
            ]
        )

        assert np.all(direct_cell.latent_cell.sample_objective(inputs) == latent_cell.sample_objective(inputs))

    def test_inputs_outside_parameter_space_raise_error(self):
        with pytest.raises(ValueError):
            FourInputCell().sample_objective(self.invalid_batch_of_inputs)

    def test_growth_factor(self):
        inputs = self.valid_batch_of_inputs
        obj1 = FourInputCell(use_growth_in_objective=True).sample_objective(inputs)
        obj2 = FourInputCell(use_growth_in_objective=False).sample_objective(inputs)
        assert np.all(obj2 > obj1)

    def test_growth_factor_2(self):
        inputs_small = np.random.uniform(10, 100, (30, 4))
        inputs_big = np.random.uniform(120, 500, (30, 4))

        cell = FourInputCell(use_growth_in_objective=True)
        assert np.all(cell._growth_factor(inputs_small) > cell._growth_factor(inputs_big))

    def test_growth_penalty(self):
        x = np.array([0.0, 0.4, 0.9, 1.0])  # Points at which we check the values
        y = FourInputCell()._growth_penalty(x)
        assert (y[0] >= y[1] > 0.9) and (y[3] <= y[2] < 0.1)


class TestThreeInputCell:
    @pytest.fixture(autouse=True)
    def _valid_inputs(self):
        bounds = ThreeInputCell().parameter_space.get_bounds()
        self.valid_batch_of_inputs = np.stack([np.random.uniform(low, upp, 10) for low, upp in bounds], axis=1)

    @pytest.fixture(autouse=True)
    def _invalid_inputs(self):
        self.invalid_batch_of_inputs = np.array(
            [
                [1e-6, 1e-6, 1e-6],
                [140, 5, 1e7],
                [10, 20, 1e7],
                [10, 5, 3e6],
            ]
        )

    def test_signal_to_crosstalk_ratio_positive(self):
        inputs = self.valid_batch_of_inputs
        obj1 = ThreeInputCell(use_growth_in_objective=True).sample_objective(inputs)
        assert np.all(obj1 > 0)

        obj2 = ThreeInputCell(use_growth_in_objective=False).sample_objective(inputs)
        assert np.all(obj2 > 0)

    def test_returns_same_as_four_input_cell(self):
        for flag in [True, False]:
            three_input_cell = ThreeInputCell(use_growth_in_objective=flag)
            four_input_cell = FourInputCell(use_growth_in_objective=flag)

            three_inputs = self.valid_batch_of_inputs
            four_inputs = np.zeros((three_inputs.shape[0], 4))
            four_inputs[:, :3] = three_inputs
            four_inputs[:, 3] = three_inputs[:, -1]

            assert np.all(
                three_input_cell.sample_objective(three_inputs) == four_input_cell.sample_objective(four_inputs)
            )

    def test_inputs_outside_parameter_space_raise_error(self):
        with pytest.raises(ValueError):
            FourInputCell().sample_objective(self.invalid_batch_of_inputs)

    def test_growth_factor(self):
        inputs = self.valid_batch_of_inputs
        obj1 = ThreeInputCell(use_growth_in_objective=True).sample_objective(inputs)
        obj2 = ThreeInputCell(use_growth_in_objective=False).sample_objective(inputs)
        assert np.all(obj2 > obj1)
