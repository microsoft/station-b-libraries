### Implementations of Clark's Moment-matching algorithm
A submodule implementing Clark's moment-matching algorithm implementation. For reference, see
the "Moment-matched Batch Acquisition Functions" paper.

Here, this algorithm is implemented in multiple different back-ends, including pytorch, jax and raw numpy.

PyTorch is currently likely most throughly tested. The JAX backend hasn't been fully optimised for performance, and hence is a bit slower. The pure numpy backend is disirable, but currently obtaining gradients of the outputs wrt. to the input to the algorithm is not possible (as the backward pass would have to be implemented manually).
