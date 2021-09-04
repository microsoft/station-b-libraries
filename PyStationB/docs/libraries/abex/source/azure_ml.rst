.. _azure_ml:

AzureML
=======

This tutorial describes how to submit runs to AzureML, retrieve, and analyse them.
It specifically describes how to do so with the :ref:`simulators` workflow, so make sure to check out that (short) tutorial first.

Running a Cell-Signalling experiment (such as the one described in :ref:`simulators`) on AzureML is as simple as appending ``--submit_to_aml`` to the run command:

.. code-block:: bash

    python projects/CellSignalling/cellsig_sim/scripts/run_cell_signalling_loop.py --spec_file projects/CellSignalling/Specs/simulator_spec.yml --num_iter 15 --num_runs 100 --strategy Bayesian --enable_multiprocessing --plot_simulated_slices --submit_to_aml


.. note::

    The same ``--submit_to_aml`` flag can be used with the regular ABEX run script: ``scripts/run.py`` 

Setting up AzureML
-------------------

All Azure ML configuration options to connect to our ABEX AzureML cluster are stored in ``azureml-args.yml``, which is part of the git repository. As such, no set-up should be required — submission to AzureML with ``--submit_to_aml`` should just work from the get-go.


Retrieving the results
----------------------

To check the progress of the experiment, you can head to the `Azure ML Website <https://ml.azure.com/>`_ and check on its progress.

Once it's marked as `completed`, you can download the generated files using the `Azure Storage Explorer <https://azure.microsoft.com/en-us/features/storage-explorer/>`_ . 
In the storage explorer, head to the your blobstore:

There, all the files generated during the run will be stored in the ``outputs`` sub-folder within a folder with the same name as the `Run ID` on the Azure ML Website. 
To analyse the generated data further, download the ``outputs`` folder onto your computer:


Generated files
---------------
After downloading the results, there will be two key folders corresponding to each run: the data folder and the results folder. 
If multiple config expansions were present, the results directory might look like this:

.. todo:: Update the names of the "iter" subdirectories

.. code-block:: bash

    expanded_option1/
    ├── run_seed00/
    │   ├── iter1
    |   |   ├── batch.csv
    |   |   └── optima.csv
    |   |   ...
    │   └── iter10
    |   |   ├── batch.csv
    |       └── optima.csv
    ├── run_seed01/
    │   ├── iter1
    |   |   ├── batch.csv
    |   |   └── optima.csv
    |   |   ...
    │   └── iter10
    |   |   ├── batch.csv
    |       └── optima.csv
    |  ...
    └── config.yml
    expanded_option2/
    ├── ...
    ...

And the data folder might look like

.. code-block:: bash

    expanded_option1/
    ├── run_seed00/
    │   ├── init_random_design_data.csv
    │   ├── experiment_outcomes_batch_000001.csv
    │   ├── ...
    │   └── experiment_outcomes_batch_000010.csv
    ├── run_seed01/
    │   ├── init_random_design_data.csv
    │   ├── experiment_outcomes_batch_000001.csv
    │   ├── ...
    │   └── experiment_outcomes_batch_000010.csv
    ├── ...
    expanded_option2/
    ├── ...
    ...

The path to the data and results folders would have been specified in the config in the ``data:⏎ folder:`` and ``results_dir:`` fields respectively.

Analysing convergence of multiple configurations
------------------------------------------------

To see how to use the plotting utilities to compare different configurations on the simulator, see :ref:`comparing_simulator_runs`.
