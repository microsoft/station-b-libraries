.. _abex-introduction:

Introduction
============

This tutorial introduces the main functionality of ABEX, which is to suggest new experiments to be carried out.
It assumes that the installation was successful.

Vanilla optimization
--------------------

The problem
^^^^^^^^^^^

Assume we have ann experimental device which we used to collect some observations. For the purposes of this tutorial,
we will mock the data using a (noisy) function

.. math::

    f(x, y) = 4 - (x-0.2)^2 - (y-0.6)^2 + \varepsilon, \quad x, y \in [0, 1],

where :math:`\varepsilon` is Gaussian noise. In reality we don't have the access to the functional
form of the objective, but for demonstration purposes it is very convenient -- observe that, up to the noise,
the maximum is at :math:`(0.2, 0.6)`.

Generate the data using the following Python script:

.. code-block:: python

    import numpy as np
    import pandas as pd

    np.random.seed(42)

    n_observations = 50

    x = np.random.rand(n_observations)
    y = np.random.rand(n_observations)
    eps = np.random.normal(scale=0.05, size=n_observations)

    objective = 4 - (x-0.2)**2 - (y-0.6)**2 + eps

    collected_data = pd.DataFrame({"x": x, "y": y})
    collected_data["objective"] = objective

    collected_data.to_csv("Tutorial/Data/tutorial-intro-data.csv", index=False)

.. note:: The values of the objective function need to be stored in the *last* column of the data frame.

.. todo:: Check whether the above is still true.

Running ABEX
^^^^^^^^^^^^

Now we need to create a YAML configuration file that tells ABEX what data is available and what we want to optimize.
The file ``Tutorial/Specs/tutorial-intro.yml`` contains the following specification:

.. warning:: Just as in Python, indentation in YAML matters. Make sure that your editor doesn't break it if you make changes.

.. code-block:: YAML
    :caption: Tutorial/Specs/tutorial-intro.yml

    data:
      folder: Tutorial/Data
      files:
        init: tutorial-intro-data.csv
      inputs:
        x:
          lower_bound: 0.0
          upper_bound: 1.0
          normalise: None
        y:
          lower_bound: 0.0
          upper_bound: 1.0
          normalise: None
      output_column: objective
      output_settings:
        normalise: None

    model:
      kernel: RBF
      add_bias: True

    training:
      num_folds: 3
      iterations: 100
      upper_bound: 1
      plot_slice1d: True
      plot_slice2d: True

    bayesopt:
      batch: 5

    results_dir: Results/tutorial-intro

.. note::

    Configuration files can only be stored in ``Tutorial/Specs``.
    There is a utility finding matching files in these directories, so you *shouldn't* specify the whole path
    when you run ABEX.

The ``data`` section of the config specifies the directory (``folder``) where the files with data can be found.
Then we specify individual files which will be used to contruct a data set.
We have only one file ``tutorial-intro-data.csv``, which we labelled as ``init`` -- the label can be arbitrary and
can correspond, for example, to the date at which experiments were collected or their order.

Then we specify the inputs. There is an important property: ``normalise``.
For example, if :math:`x` and :math:`y` were to range throughout the interval :math:`[0, 10^9]`,
numerical issues might arise when constructing the model.
For this reason, the inputs are usually normalized. In our case this is not needed.

Then we specify the objective column (``output_column``, ``output_settings``) and ``device``,
which corresponds to the ``Gene ID`` set above.

The next section (``model``) specifies the Gaussian Process model which will be used.

Then we come to ``training``. ABEX will use cross-validation with ``num_folds: 3``: it will split the data in 3
different ways into training and validation sets to evaluate the accuracy of constructed model.

The section ``bayesopt`` specifies the settings of Bayesian Optimization -- we simply specified that in our next
experiment we can investigate 5 points. Usually large batches need a long compute time.

.. tip:: Set ``batch`` to 0 if you only want to evaluate the Gaussian Process model and do not need any experiment suggestions.

The last section (``results_dir``) specifies where the directory with results should be generated.

Run

.. code-block:: bash

    python scripts/run.py --spec_file tutorial-intro.yml

and wait for several minutes.


Interpreting results
^^^^^^^^^^^^^^^^^^^^

Open the directory ``Results/tutorial-intro``. File ``batch.csv`` contains suggestions of 5 points to be tested in
the next experiment. Observe they don't need to be close to each other -- Bayesian optimization needs to balance
*exploitation* (sampling near the maximum already found) and *exploration* (exploring promising areas, which may
contain an undiscovered maximum). Plot ``bo_distance`` illustrates the similarity between different points and
``bo_experiment`` clearly shows in what region of the parameter space they are located.

To see what regions look promising, see plots ``acquisition1d_*``, which show the acquisition function for a model
which uses all data samples and three models built using different cross-validation folds. Note that this suffix
convention applies to other files as well.

Plots ``slice_1d`` and ``slice_2d`` show the mean prediction of the GP in lines and planes passing through the maximum
found by the model.

The optimum estimated using the GP model can be found in ``optima`` files. (It may happen that this file contains
more than one point, what happens if different *contexts* are considered).


A more advanced example
-----------------------

This time we will optimize a more complicated function:

.. math::

    f(x) = 1200 \exp \left( \cos \left(\pi (\log_{10} x-1)\right) + \varepsilon \right), \quad x \in [1, 10^4]

In particular observe that it has two maxima :math:`x_1=10` and :math:`x_2=10^3`. Moreover, if  :math:`\varepsilon` is
a Gaussian noise term, then the error on the observed objective is not Gaussian any longer.

We will generate example values assuming that :math:`x` was fairly uniformly measured in the log-space:

.. code-block:: python

    import numpy as np
    import pandas as pd

    np.random.seed(42)

    n_observations = 50

    logx = 4 * np.random.rand(n_observations)
    eps = np.random.normal(scale=0.2, size=n_observations)

    objective = 1200 * np.exp(np.cos(np.pi * (logx - 1)) + eps)

    collected_data = pd.DataFrame({"x": 10**logx})
    collected_data["objective"] = objective

    collected_data.to_csv("Tutorial/Data/tutorial-intro-2.csv", index=False)

This time we will use more advanced data preprocessing and a different kernel, and we don't plot 2D slices of the
model as we have only one input variable.

.. code-block:: YAML
    :caption: Tutorial/Specs/tutorial-intro-2.yml

    data:
      folder: Tutorial/Data
      files:
        init: tutorial-intro-2.csv
      inputs:
        x:
          lower_bound: 1.0
          upper_bound: 10000.0
          normalise: Full
          log_transform: True
      output_column: objective
      output_settings:
        normalise: Full
        log_transform: True

    model:
      kernel: Matern
      add_bias: True

    training:
      num_folds: 3
      iterations: 100
      upper_bound: 1
      plot_slice1d: True
      plot_slice2d: False

    bayesopt:
      batch: 10

    results_dir: Results/tutorial-intro-2

After we run

.. code-block:: bash

    python scripts/run.py --spec_file tutorial-intro-2.yml

we should get a ``Results/tutorial-intro-2`` directory. This time the Expected Improvement acquisition function will have
two local maxima. After a careful investigation of ``batch.csv`` one can conclude that ABEX recommends measuring
points both near 10 and 1000, i.e. is exploiting both regions that look promising Note that, in ``optima.csv`` only
one maximum is provided.

Problems
^^^^^^^^

  1. What happens if log transform for ``x`` is turned off?

  2. Now assume that log transform for ``x`` input is turned on but normalization and log transform for the objective
     are turned off. What happens?

Next steps
----------

The primary use case for ABEX is the optimization of signaling pathways.
See the following tutorial:

  * :ref:`cell_signalling`

For a more advanced tutorial on creating configuration files see :ref:configs.

ABEX offers more functionalities, including suggesting initial experiments to be collected or running simulated
experiments in the cloud -- should you need any of them, please consult other tutorials.
