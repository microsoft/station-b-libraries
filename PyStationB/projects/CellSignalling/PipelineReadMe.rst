Cell Signalling
===============

This repository implements a semi-automated pipeline for the collection & analysis of batches corresponding to different algorithms.


Installation
------------
.. note::

  It is a good practice to use a dedicated virtual environment (e.g. virtualenv or conda) for installation.

First, clone the repository using the HTTPS protocol and install all the relevant libraries in the ``python`` folder:

.. code-block:: bash

    git clone https://github.com/microsoft/station-b-libraries
    git submodule update --init --recursive



To pull the data from BCKG you need to set up connection string:

.. code-block:: bash

    export BCKG_PRODUCTION_CONNECTION_STRING = "<the BCKG connection string>"

It is handy to store this in a configuration file.

Running the experiments
-----------------------

Initializing an experimental track
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We structure experiments into *experimental tracks*. An experimental track is basically a set of tags which distinguish
it from any other experimental track. Right now we use the following four tags:

.. code-block:: yaml

    Batch: 5  # An integer, in principle should correspond to the batch size
    Project: CellSignalling  # This tag is shared for all our cell signalling experiments. It shouldn't be modified
    Track_name: BayesOptDefault  # This name can be arbitrary
    TrackNumber: 2  # If we want to have many experimental tracks with the same batch and track_name, we can distinguish them with this

For example, the following experimental tracks are different:

  - ``(5, CellSignalling, BayesOptDefault, 1)``
  - ``(10, CellSignalling, BayesOptDefault, 1)``
  - ``(10, CellSignalling, BayesOpt, 1)``
  - ``(10, CellSignalling, BayesOpt, 2)``

These tags uniquely determine the experimental track in BCKG. However, in this pipeline we use human-readable names.

.. footnotes::

    Each of these tags are associated with a generated experiment in BCKG. Apart from that, we can also tag specific samples
    (an important example of this is the PairingID, currently referred to as "ObservationID"). This is called the *level* of the tag.


For example, I can create an experimental track named ``InitialExploration``, which will live in the ``Experiments/``
directory:

.. code-block:: bash

    python scripts/init.py --names InitialExploration

and then set it the tags uniquely determining it in pyBCKG:

.. code-block:: bash

    cd Experiments/InitialExploration
    vim data_config.yml  # Use the editor of your choice to set the experiment-level tags :)

In fact, the ``data_config.yml`` for each experimental track contains not only the tags, but also an important entry

.. code-block:: yaml

    initial_experiment_id: AAAAAA-BBBBBB-...

To get the very first batch of suggestions in this experimental track, we need to initialize the model with some
data. This data can be shared across many experimental tracks: for example, we may have already have an experiment in BCKG
which worked very well and we may want to use it as the initial data for many experimental tracks.

This experiment ID is something we need to know from BCKG.

We set the tags and the initial experiment. Now we need to configure ABEX. Assuming you are already in ``Experiments/InitialExploration``,
open the file ``abex_config.yml``. This is the usual ABEX config, so you can set the hyperparameters of BayesOpt/ZoomOpt.

.. warning::

    Leave ``files:`` empty and do not modify the ``folder:``.
    The ``results_dir`` field is set automatically at each ABEX run.
    The names of inputs and the output should not be modified.

In the future, you may want to create many experimental tracks at once:

.. code-block:: bash

    python scripts/init.py --names Track AnotherTrack February

.. note:: Remember to set the configuration files for each of the tracks you initialized.

.. tip::

    It's a good practice to keep a ReadMe file in the experiment track directory,
    which explains the motivation behind this track and acts as an experimental logbook, e.g.:
    "At iteration 5, I didn't see any improvement over iteration 4. If iteration 6 won't be better,
    I'll assume this is the optimum."

Running an iteration
^^^^^^^^^^^^^^^^^^^^

If we have a few experimental tracks initialized, we can *run an interation*, what means that we can run ABEX on them
and then merge the outputs into one DoE, ready to be uploaded to Antha. This DoE is used by the robot to do the pipetting
and later by BCKG to tag different experiments appropriately.

So, how do we create a plate layout with many experimental tracks?

This is as easy as:

.. code-block:: bash

    python scripts/run_iteration.py --iteration 1 --names Track AnotherTrack YetDifferentTrack

The script will download the data from the previous iteration (as iteration is 1, it will download the initial experiments, as we set
in the data configs), run the ABEX and then merge all suggestions in the DoE.

.. note::

    The iteration must be the same for all experimental tracks. It's easy to change the way of running the experiments
    (so that each experiment uses a different iteration), but it's not *currently* supported.
    If you need to do this, we suggest generating separate DoEs (running the script multiple times, for each experiment)
    and then concatenating them in Excel.

As a result, we get a DoE containing all the tags. When it's uploaded to Antha, actually performed, and uploaded to pyBCKG,
we can run

.. code-block:: bash

    python scripts/run_iteration.py --iteration 2 --names Track AnotherTrack YetDifferentTrack

what will download the data from iteration 1 and run ABEX to suggest the DoE for the second iteration.


High-level overview
^^^^^^^^^^^^^^^^^^^

1. Do the initial experiment (e.g. random design) and upload it to BCKG.
2. Initialize the experimental tracks that will use that experiment.
3. Run the first iteration.
4. Upload the DoE to Antha, schedule a run, upload the data to BCKG.
5. Run the second iteration.
6. Upload the DoE to Antha ...
7. Run the third ...


Contribution
------------
We use the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_.

ABEX and pyBCKG APIs are quickly evolving. When they both reach stable 1.0 versions, this pipeline
will need to be restructured as well.

The code is at 0.0.1 version and is far from perfect. Any contributions increasing the code quality are very welcome.

Currently we don't have automated pipelines enforcing code policies, but they may arrive over the time.
