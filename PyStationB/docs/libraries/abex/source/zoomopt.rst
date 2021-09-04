.. _zoom_optimization:

"Zooming in" optimization
=========================

ABEX focuses on batch Bayesian optimization -- this approach quantifies uncertainty and is perfect for
optimization of noisy functions which may have several maxima.

However, in biology another approach is quite common. One starts with building a grid consisting of uniformly
spaced points from the hypercube:

.. math::

    S_0 = [a_1, b_1] \times [a_2, b_2] \times \cdots \times [a_n, b_n]

When the results of experiments are collected, the most promising point is chosen. This is usually the maximum
over collected samples, though it may be shifted according to expert knowledge and intuition.

When an initial estimate of the maximum :math:`x=(x_1, x_2, \dots, x_n)` is chosen, a new "shrunk" space is constructed:

.. math::

    S_1 = [x_1-\delta_1, x_1+\delta_1] \times [x_2-\delta_2, x_2+\delta_2] \times \cdots \times [x_n-\delta_n, x_n+\delta_n],

such that the volume of :math:`S_1` is a prescribed fraction of :math:`S_0`, for example :math:`f=1/2`. This is called
the *shrinking factor*.

.. note:: The length of each bounding interval shrinks by a factor of :math:`\sqrt[n]{f}`.

Then the experiment may be repeated. Now a new estimate is chosen based on both batches, and space :math:`S_2`
is constructed in a similar way. Note that

.. math::

    \mathrm{vol}\, S_k =  f\cdot \mathrm{vol}\, S_{k-1} =  \cdots = f^k \cdot \mathrm{vol}\, S_0,

meaning that the space we sample becomes smaller and smaller and the shrinking factor controls how quickly the space
converges.

It is important to note that this algorithm is a simple heuristic -- it is not expected to work well with noisy functions
or functions with several maxima. On the other hand, each experimental batch is easy to collect in the laboratory,
the algorithm does not require as much computational power as building a surrogate model, and it is commonly used in practice.

Our flavour of the algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Our implementation of the algorithm (see :py:mod:`abex.optimizers.zoom`) differs in two ways from the "vanilla" algorithm
described above.

First, when the space :math:`S_n` is selected, we do not have to build a uniform grid over it. Our variant of the
algorithm allows random, Latin and Sobol sampling as well.

Second, it may happen that the maximum :math:`x` over first :math:`k` batches is near the boundary of :math:`S_0`, so
that the new space :math:`S_k` is not a subset of :math:`S_0`.
In this case, we translate :math:`S_k` by a vector (so the volume does not change) such that it is a subset of the original
space.

.. note:: This assures that :math:`S_k\subset S_0`, but it does *not* imply that :math:`S_k\subset S_{k-1}` in general.

There is also a small remark regarding running the algorithm. This approach needs information about the optimization step, so that
the space :math:`S_k` with appropriate volume is constructed. If this is not specified in the configuration file, we estimate the step
to be (the floor of) the number of samples collected divided by the batch size.

.. todo:: What is meant by "optimization step" here? Should it be "shrinking factor"? And how is "number of samples" different from "batch size"?

Moreover, sometimes a grid with given batch size does not exist (for example in two-dimensional space, one can build a grid
for batch size :math:`b` if and only if :math:`\sqrt{b}` is integral). In that case, some of the points in the grid may be missing.
(Of course this comment does not apply to random or almost random sampling methods, such as Sobol or Latin).

Running experiments
-------------------

We will use this strategy to optimize a function

.. math::

    f(x) = 2 - (x-0.42)^2 + \varepsilon, \quad x \in [0, 10],

where :math:`\varepsilon` is small (but not necessarily Gaussian) noise. We will use batches of size 5.

.. code-block:: python

    import numpy as np
    import pandas as pd

    np.random.seed(24601)

    batch_size = 5

    x = np.linspace(0, 10, batch_size)
    eps = np.random.rand(batch_size)

    objective = 2 - (x-0.42)**2 + eps

    collected_data = pd.DataFrame({"x": x})
    collected_data["objective"] = objective

    collected_data.to_csv("CellSignalling/Data/tutorial-zoomopt.csv", index=False)

Now we need to create a configuration file:

.. code-block:: YAML
    :caption: CellSignalling/Specs/tutorial-zoomopt.yml

    data:
      folder: CellSignalling/Data
      files:
        init: tutorial-zoomopt.csv
      inputs:
        x:
          lower_bound: 0.0
          upper_bound: 10.0
          normalise: None
          log_transform: False
      output_column: objective
      output_settings:
        normalise: None
        log_transform: False

    zoomopt:
      batch: 5
      design: Grid
      shrinking_factor: 0.5

    optimization_strategy: Zoom

    results_dir: Results/tutorial-zoomopt

The ``data`` and ``results_dir`` sections do not differ from the usual Bayesian optimization. However, we do have
two new sections: a self-explanatory ``optimization_strategy`` (which, if not specified, defaults to ``Bayesian``,
so is not visible in Bayesian optimization) and ``zoomopt`` providing the settings of the optimization.

Apart from shrinking factor and batch size, we can set the sampling method (see :py:mod:`abex.space_designs` for an
overview of available methods) and the optimization step ``n_step: 1`` (as this is the first time that we run the
optimization). As we didn't set it directly, the step number will be automatically inferred from the number of observations
and the batch size.

.. todo::

    Clarify the above. Do "optimization step" and "step number" both mean "number of steps"?
    Where is the number of observations set?

Now we can run the zoom optimization:

.. code-block:: bash

    python scripts/run.py --spec_file tutorial-zoomopt.yml

Our data set has only 5 points, so it is instantly created. Zoom optimization does not require any model, so the suggestions
are given immediately as well. In ``Results/tutorial-zoomopt``, there should be a file with a batch of size 5, evenly spanning
the range :math:`S_1=[0, 5]` -- the observed maximum so far was at 0. As the volume (which is in this case ordinary length)
of the initial interval :math:`S_0=[0, 10]` is 10 and the shrinking factor is :math:`f=0.5`, interval :math:`S_1` should have
length 5 and be centered at 0. However, as :math:`[-2.5, 2.5]` is not inside :math:`S_0`, we translate it to obtain the interval
:math:`S_1=[0, 5]`.

Let's collect a new batch of data:

.. code-block:: python

    import numpy as np
    import pandas as pd

    np.random.seed(42)

    batch_size = 5

    x = np.linspace(0, 5, batch_size)  # Check that these are the suggestions given by the algorithm!
    eps = np.random.rand(batch_size)

    objective = 2 - (x-0.42)**2 + eps

    collected_data = pd.DataFrame({"x": x})
    collected_data["objective"] = objective

    collected_data.to_csv("CellSignalling/Data/tutorial-zoomopt-1.csv", index=False)

Now we need to append the new batch to the YAML specification file:

.. code-block:: YAML
    :caption: CellSignalling/Specs/tutorial-zoomopt.yml

    data:
      folder: CellSignalling/Data
      files:
        init: tutorial-zoomopt.csv
        1: tutorial-zoomopt-1.csv
      inputs:  # The rest of the file stays the same
      ...

.. note:: If you previously set ``n_step: 1``, modify it to ``n_step: 2``.

Now run the pipeline. Again, in ``Results/tutorial-zoomopt`` we will see 5 points. This time they will span an interval
of length :math:`2.5=0.5^2 \cdot 10`.

By repeating this procedure, you may get closer and closer to the optimum. However, if we have a simulator which can be used (as in
this case), it's easier to use :ref:`simulators`, rather than repeating this process manually.
