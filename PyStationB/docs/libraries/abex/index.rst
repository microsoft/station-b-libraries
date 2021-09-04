Automated Biological EXperimentation
====================================

Automated Biological Experimentation (ABEX) uses Gaussian Process models and Bayesian optimization to automatically design experiments for defined objectives.
Typically, Bayesian optimization is suitable for expensive-to-evaluate functions, a property that can almost always be attributed to the design of experiments for biological systems.
Measurements of a biological system in a laboratory can be conducted for a range of *conditions*. We often do not fully understand the mechanisms through which those conditions
affect the measurable behaviour of the biological system. Therefore, an emulator of the input-output behaviour is constructed as a Gaussian Process. This is then used to evaluate
an acquisition function that rewards prospective new evaluations of the system when there is some probability that a new optimum might be found.

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   source/introduction
   source/configs
   source/initialization
   source/zoomopt
   source/simulators
   source/azure_ml
   source/plotting
   source/tips

.. toctree::
   :maxdepth: 2
   :caption: Development

   source/architecture