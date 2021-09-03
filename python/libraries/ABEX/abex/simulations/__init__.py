# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This submodule allows running many steps of Bayesian Optimization of an artificial function.

A DataLoop allows multiple Bay. Opt. iterations applied to any DataGeneratorBase, which is a class for 'running
experiments'.

Any artificial function should be implemented as a SimulatorBase. For convenience, we implemented
a SimulatedDataGenerator, which is a data generator wrapping around simulators, so that they can be optimized using
the data loop.

We provide examples of such artificial functions in `toy_models` submodule.

Exports
    SimulatorBase, an interface for a function to be optimized
    DataGeneratorBase, an interface of any experiment, which provides new observations, experimental or simulated
    SimulatedDataGenerator, a wrapper around a simulator (new observations are predicted by the simulator)
    DataLoop, an utility for running many steps of a data generator
    toy_models, a submodule implementing example simulators
"""
from abex.simulations.data_loop import DataLoop  # noqa: F401
from abex.simulations.interfaces import SimulatedDataGenerator, DataGeneratorBase, SimulatorBase  # noqa: F401
from abex.simulations import toy_models  # noqa: F401
