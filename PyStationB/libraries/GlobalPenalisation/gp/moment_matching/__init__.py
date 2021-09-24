# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
A submodule implementing Clark's moment-matching algorithm implementation.

Here, this algorithm is implemented in multiple different back-ends, including pytorch, jax and raw
numpy.

PyTorch is currently likely most throughly tested. The JAX backend hasn't been fully
optimised for performance, and hence is a bit slower. The pure numpy backend is disirable, but currently
obtaining gradients of the outputs wrt. to the input to the algorithm is not possible (as the backward
pass would have to be implemented manually).
"""
