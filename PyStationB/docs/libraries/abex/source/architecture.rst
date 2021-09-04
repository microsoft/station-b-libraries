Architecture
------------

The main functionality of ABEX is provided by the command

.. code-block:: bash

    python scripts/run.py spec_file.yml

We also have individual scripts for different purposes (e.g. suggesting initial batches to be collected or visualising
the results) in the ``scripts`` directory.

There are also scripts specific to the cell signalling project, in ``CellSignalling/scripts``.

However, ``abex`` *is a Python module and is intended to allow easy development of other scripts*.

Structure
^^^^^^^^^

Most of the ABEX code is structured as follows:

  - ``run``: the submodule responsible for running the main utility of ABEX,

  - ``optimizers``: this submodule contains the optimization strategies, which suggest new batches of samples to be collected,

  - ``simulations``: this allows running multiple optimization steps if a simulator is available,

  - ``data``: we have a special data structure, for representation of the training and test parameters,

  - ``transforms``: the data structure in ``data`` also normalises the data using this submodule,

  - ``settings``: parsed YAML files are stored in ``Config`` object, defined in this module. Changes to this modules
    should be documented in :ref:configs,

  - ``expand``: parser which extends YAML language, to produce different versions of a basic config,

  - ``optim``: *probably* used to analyse the predictions of a GP model, finding points suspected to be maximizing the objective,

  - ``plotting``: visualisation utilities,

  - ``constants``: global constants shared between different submodules,

  - ``common_util``: mostly logging utilities,

  - ``space_designs``: a submodule with different methods of sampling space (e.g. for suggesting initial batches).


Task-specific submodules
^^^^^^^^^^^^^^^^^^^^^^^^

Although ABEX is supposed to be task-agnostic, we store most of our reusable code for our primary task as its
submodules:

  - ``cell_signalling``: this submodule stores the code specific to our internal wet-lab experiments.
    In particular it implements mathematical models of cells and methods that allow moving between optimization and
    experimental space (see :ref:`mathematical_modeling`),

.. warning::

    These two modules may be removed from the future, officially released versions of ABEX.
    If we publicly release the code of ABEX, this code should be pruned from the Git history as well.
