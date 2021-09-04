"""Automatic Biological EXperimentation (ABEX)

There are a few related functionalities of ABEX. The first one is:

Suggestions of new experiments
------------------------------

Having outputs of already collected experiments, we want to suggest a batch of experiments that should be collected.

To make a successful run one uses:

  - An optimization strategy. This can be either Bayesian Optimization (the most mature method for now) or another.
      They are implemented in `abex.optimizers` package.

  - A Config object, which encodes a configuration YAML (see `abex.settings`). For examples of configuration YAMLs,
      see Specs of CellSignalling. Our Configs can encode different experiments with different
      settings and can be treated as a small programming language -- see `abex.expand`.

  - A data set. Our data may come from different sources: be represented by either a CSV file or by some schema in BCKG.
      The `abex.data` submodule implements methods for creating data sets. Data sets and predictions need often to be
      transformed -- e.g. we may want to optimize the problem in the log-space. For different methods of moving between
      optimization spaces, see `abex.transforms`.

  - One also can assess the quality of suggestions/performance of the models using plotting utilities of `abex.plotting`


Optimization of simulators
--------------------------

If a simulator, approximating a real experiment is available (e.g. a biochemical model of a cell or cross-validation
results of a machine learning model), we may want to optimize over many steps.

For these purposes:
  - different optimization strategies are in `abex.bayesopt` and `abex.run`, and `abex.optimizers`.
  - `abex.simulations` implements a `DataLoop`, which optimizes a simulator using a few steps. It shows how to wrap
    a model (`DataGeneratorBase`, `SimulatorBase`) with example implementations given in `abex.simulations.toy_models`.
  - plotting utilities of `abex.plotting` can draw e.g. convergence plots


Optimization of our wet-lab experiments
---------------------------------------

For this we have `abex.cell_signalling` module. It consists of `abex.wetlab`, implementing 'data combinatorics' and
basic Antha integration, and `abex.cell_signalling.simulations`, implementing biochemical cell models
"""
