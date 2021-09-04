# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import logging
from typing import Optional, Callable, Any

from abex.settings import OptimizerConfig
from abex.simulations import SimulatorBase
from cellsig_sim.simulations import HillFunction, FourInputCell, ThreeInputCell


class CellSignallingOptimizerConfig(OptimizerConfig):
    """Expands upon the OptimizerConfig for the core ABEX run function to allow for specifying the
    parameters for the cell-signalling simulator.
    Parameters:
        incorporate_growth: Should cell growth be incorporated into the objective? Defaults to False.
        multimodal: Whether to adjust the transfer functions to give multimodal behaviour
        simulator_noise: Standard deviation of the optional multiplicative noise on the output
            (noise is log-normal distributed).
    """

    incorporate_growth: bool = False
    multimodal: bool = False
    simulator_noise: Optional[float] = None
    heteroscedastic_noise: bool = False

    def get_simulator(self) -> SimulatorBase:  # pragma: no cover
        """Make a cell simulator class according to settings in self."""
        # Infer the type of the cell (three or four) and create an Arabinose/ATC cell
        n_inputs = len(self.data.input_names)
        # Select the transfer functions for the simulator
        luxr_transfer: Optional[Callable[[Any], Any]] = None  # type: ignore # auto
        lasr_transfer: Optional[Callable[[Any], Any]] = None  # type: ignore # auto
        if self.multimodal:
            # These transfer functions have been chosen mostly ad-hoc to make for what
            # visually appeared like a somewhat realistic, but non-trivial problem.

            def luxr_transfer(ara):
                return HillFunction(n=0.8, K=0.1, scale=100)(ara) - HillFunction(n=0.8, K=4, scale=75)(ara)

            def lasr_transfer(atc):
                return HillFunction(n=0.9, K=0.5, scale=20)(atc) - HillFunction(n=0.9, K=1.5, scale=20)(atc)

        if n_inputs == 4:
            simulator = FourInputCell(  # pragma: no cover
                self.incorporate_growth,
                noise_std=self.simulator_noise,
                luxr_transfer_func=luxr_transfer,
                lasr_transfer_func=lasr_transfer,
                heteroscedastic=self.heteroscedastic_noise,
            )
        elif n_inputs == 3:
            simulator = ThreeInputCell(
                self.incorporate_growth,
                noise_std=self.simulator_noise,
                luxr_transfer_func=luxr_transfer,
                lasr_transfer_func=lasr_transfer,
                heteroscedastic=self.heteroscedastic_noise,
            )
        else:
            raise ValueError("Only three- and four-input cells are supported.")  # pragma: no cover

        logging.info(f"Running the simulated loop with the {n_inputs}-input simulator.")
        logging.info(f"Including growth in objective: {self.incorporate_growth}")
        return simulator
