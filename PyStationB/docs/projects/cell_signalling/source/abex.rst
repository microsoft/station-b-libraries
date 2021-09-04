.. _cell-signalling-configs:

Optimization
============

In this section, we show examples of how to call ABEX for different scenarios.

Example 1: "Wet-lab"
--------------------

This an example of a reasonable Bayesian Optimization config to use for the Cell-signalling wetlab experiments.
It exists in the repository as `CellSignalling/Specs/tutorial-wetlab.yml`.

.. code-block:: yaml

    bayesopt:
      acquisition: MEAN_PLUGIN_EXPECTED_IMPROVEMENT  # Important to use mean-plugin to account for the noise
      batch_strategy: MomentMatchedEI  # Moment-matched EI is the batch acquisition strategy developed by us
      acquisition_optimizer: Random  # Currently only Random is supported for MomentMatchedEI batch strategy
      batch: 30  # Size of the batch
      num_samples: 50000  # Number of random samples to evaluate the acquisition function at. Keep > 30000 if acquisition_optimizer is Random
    data:
      folder: CellSignalling/Data/LaboratoryExperiment  # Adjust adequately
      files:
        1: data_file.csv  # And point to the right files
      inputs:
        ATC:
          log_transform: true
          lower_bound: 0.25
          upper_bound: 5000.0
          normalise: Full
          unit: ng/ml
        Ara:
          log_transform: true
          lower_bound: 0.01
          upper_bound: 200.0
          unit: mM
          normalise: Full
        C_on:
          log_transform: true
          lower_bound: 1.0
          upper_bound: 20000.0
          normalise: Full
          unit: nM
      output_column: Objective
      output_settings:
        log_transform: true
        normalise: None  # Do not normalise output, as this will effectively change the effect of priors at each iteration
    model:
      add_bias: true  # Setting to True helps with exploration
      anisotropic_lengthscale: False  # We saw little benefit on the simulators of setting to True
      kernel: Matern  # Matern generally recommended above RBF for Bayesian Optimization
      priors:
        sum.Mat52.variance:
          distribution: InverseGamma
          parameters: {a: 1.0, b: 1.0}
        sum.Mat52.lengthscale:
          distribution: InverseGamma
          parameters: {a: 3.0, b: 1.0}
        sum.bias.variance:
          distribution: InverseGamma
          parameters: {a: 2.0, b: 1.0}
        Gaussian_noise.variance:
          distribution: InverseGamma
          parameters: {a: 5.0, b: 1.0}
    optimization_strategy: Bayesian
    training:
      compute_optima: true  # Keep as True if you want to inspect the inferred optima at each iteration. Also required for plotting
      hmc: False  # TODO! Currently HMC doesn't work
      iterations: 100 
      num_folds: 4  # Number of cross-validation fold, only affects plots made
      plot_slice1d: true
      plot_slice2d: true
      plot_binned_slices:  # Helpful for visualising the acquired points
        dim_x: 2
        slice_dim1: 0
        slice_dim2: 1
        num_bins: 5
    results_dir: Results/RealWetlabExperiment

.. _cell_signalling_simulator_config:

Example 2: Simulators
---------------------

.. todo:: This section is missing. It should cover how we use simulators and the data loop to investigate different optimization strategies.

This is very similar to the above, with one key difference: *the input bounds are different*!
it is in the repository as `CellSignalling/Specs/tutorial-wetlab-sim.yml`.

.. code-block:: yaml

    bayesopt:
      acquisition: MEAN_PLUGIN_EXPECTED_IMPROVEMENT  # Important to use mean-plugin to account for the noise
      batch_strategy: MomentMatchedEI  # Moment-matched EI is the batch acquisition strategy developed by us
      acquisition_optimizer: Random  # Currently only Random is supported for MomentMatchedEI batch strategy
      batch: 10  # Size of the batch
      num_samples: 50000  # Number of random samples to evaluate the acquisition function at. Keep > 30000 if acquisition_optimizer is Random
    data:
      folder: CellSignalling/Data/LaboratoryExperiment  # Adjust adequately. No need to specify files!
      inputs:
        ATC:
          log_transform: true
          lower_bound: 0.0001  # Note the different bounds!
          upper_bound: 100.0
          normalise: Full
        Ara:
          log_transform: true
          lower_bound: 0.0001
          upper_bound: 100.0
          normalise: Full
        C_on:
          log_transform: true
          lower_bound: 0.01
          upper_bound: 25000.0
          normalise: Full
      output_column: "Crosstalk Ratio"  # Different output column to wetlab!
      output_settings:
        log_transform: true
        normalise: None  # Do not normalise output, as this will effectively change the effect of priors at each iteration
    model:
      add_bias: true  # Setting to True helps with exploration
      anisotropic_lengthscale: False  # We saw little benefit on the simulators of setting to True
      kernel: Matern  # Matern generally recommended above RBF for Bayesian Optimization
      priors:
        sum.Mat52.variance:
          distribution: InverseGamma
          parameters: {a: 1.0, b: 1.0}
        sum.Mat52.lengthscale:
          distribution: InverseGamma
          parameters: {a: 3.0, b: 1.0}
        sum.bias.variance:
          distribution: InverseGamma
          parameters: {a: 2.0, b: 1.0}
        Gaussian_noise.variance:
          distribution: InverseGamma
          parameters: {a: 5.0, b: 1.0}
    optimization_strategy: Bayesian
    training:
      compute_optima: true  # Keep as True if you want to inspect the inferred optima at each iteration. Also required for plotting
      hmc: False  # TODO! Currently HMC doesn't work
      iterations: 100 
      num_folds: 4  # Number of cross-validation fold, only affects plots made
      plot_slice1d: true
      plot_slice2d: true
      plot_binned_slices:  # Helpful for visualising the acquired points
        dim_x: 2
        slice_dim1: 0
        slice_dim2: 1
        num_bins: 5
    results_dir: Results/RealWetlabExperiment

    # Simulator specific options (only keep if running run_cell_signalling_loop.py)

    init_design_strategy: Random
    multimodal: True  # Keep as True, otherwise the simulator is quite boring and all methods converge really quickly
    simulator_noise: 1.0
    num_init_points: 10
    incorporate_growth: False  # Keep False, the growth model used is unrealistic
    heteroscedastic_noise: False  # Whether the simulator should generate samples with heteroscedastic noise

Important options for Bayes. Opt.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 - ``acquisition: MEAN_PLUGIN_EXPECTED_IMPROVEMENT`` - using the mean plug-in version of Expected Improvement is very useful when applying Bayesian Optimization to noisy problems, and is essentially a requirement to get ``MomentMatchedEI`` to work on noisy problems.
 - ``priors`` - Setting the priors to sensible options is really crucial. See :ref:_setting_a_prior for more detail.
 - ``batch`` and ``num_init_points`` - usually, you would want these to be the same (but not always, hence we allow the flexibility). If using config expansion (:ref:expanding_configs), make sure to vary these two parameters together.

Example 3: Zoom Optimization
----------------------------

To run zoomopt, ``bayesopt`` and ``optimization_strategy`` options can be replaced with the following:

.. code-block:: yaml

  zoomopt:
    batch: 10  # Specify a batch-size[
    design: Random
    shrinking_factor: ['@sf', 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.99999999]  # An example of multiple decent shrinking factors
    shrink_per_iter: True  # Keep set to True


Usually, shrinking factor of around ``0.9`` or ``0.95`` performs the best. It's very useful to keep the ``shrink_per_iter`` option set to ``True``, as it means the optimal shrinking factor will be largely batch-size independent.
