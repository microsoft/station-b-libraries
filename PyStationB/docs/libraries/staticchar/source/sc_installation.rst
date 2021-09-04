Getting Started
===============

.. _installation_staticchar:

Installation
------------

We recommend you set up to use all the code in the PyStationB monorepo, as described in "Getting started"
in the monorepo documentation. However if you wish, to install characterization independently of the monorepo, you can do

.. code-block:: bash

    pip3 install -e .

which will install all needed dependencies.

To get started with characterization, see the example notebook in the ``examples/`` directory
and/or the tests in ``tests/notebooks``.
They use example time series included in the ``tests/tests_data/`` directory.
(This is a small data set included only for illustratory purposes).


Running Static Characterization
-------------------------------

From the ``libraries/StaticCharacterization`` subdirectory of ``PyStationB``, you can give the command

.. code-block:: bash

   python staticchar/run.py config1.yml ... configN.yml data_dir1/ ... data_dirN/

where the argument list includes at least one configuration file like those in ``tests/configs``, and at least one data directory
representing an experiment and containing per-well time series csv files. Example data directories are in ``tests/test_data``. 
This script will run every configuration on
every data directory, creating a new directory with a name of the form ``SC_RESULTS/config_name/data_dir_name/NNN`` for every combination,
where ``NNN`` is chosen to be unique to the current run among subdirectories under ``SC_RESULTS/config_name``.
For example, ``python staticchar/run.py tests/configs/gradient.yml tests/test_data/S-shape/`` will create ``SC_RESULTS/gradient/S-shape/001``;
or if there are already subdirectories under ``SC_RESULTS/gradient`` with name ``001`` but not ``002``, then ``SC_RESULTS/gradient/S-shape/002`` will be created.
Within each subdirectory, the following data files and plots are generated.

* ``config.yml``: a copy of the configuration file used to create this directory.
* ``characterizations.csv``: a file with column names of the form ``SampleID,ExperimentID,Well,Condition1,...,ConditionN,Signal1,...,SignalN``,
  for example ``SampleID,ExperimentID,Well,C6HSL,C12HSL,Arabinose,ATC,EYFP,ECFP``.
  Each row in the file represents a single well. The Condition values are taken from the ``_conditions.csv`` file in the data directory,
  and the Signal values are the results of characterization with the specified method (Gradient or Integral).
* ``model_values.csv``: a file with column names ``SampleID,carrying_capacity,growth_rate,lag_time,time_maximal_activity,log_initial_density``,
  with all values except ``SampleID`` coming from the fitted growth model (Gompertz or Logistic).
* ``signals_vs_time.png``: a grid of scatterplots, one per well, showing how each signal and the reference value evolved over time.
  The time period used for characterization is highlighted. A log scale is used for signals.
* ``signals_vs_reference.png`` (gradient method only): as for ``signals_vs_time.png``, but the X axis is the reference signal rather than time, 
  and linear scaling is used.
* ``growth_model.png``: a plot of the growth signal (typically optical density) against time, showing the actual data points and the fitted 
  growth model curve.
* ``value_correlations.png``: a scatterplot, with one point per well, of the characterized value of every signal, including the growth signal, 
  against every other signal and every condition. Signal values (on either axis) are plotted on a log scale, and condition values (always on the X axis)
  on a linear scale. Also shown is the Pearson correlation coefficient (using log values for signals) and its two-tail p value. When p values are
  significant at the 5% level after Bonferroni correction, i.e. when p < 0.05 / number-of-wells, a red colour is used.
* ``rank_correlations.png``: as for ``value_correlations.png``, but this time, rather than linear or log scales, the relative rank of each
  quantity is used, with 0.0 for the smallest and 1.0 for the largest. Where there is a tie, e.g. when the same condition value is used across
  multiple wells, the mean rank for that value is used. Correlations are calculated on the ranks.
* ``integration_SIG.png``, for each signal name ``SIG`` (integral method only). For each well, a time series plot of the signal in question, with
  the region over with the integral is calculated highlighted.