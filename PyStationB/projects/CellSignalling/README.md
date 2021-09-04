# ABEX: Cell Signalling
In this directory we keep the scripts and configuration files specific to cell signalling (our internal wet-lab) experiments.

## Pipeline
Install all required submodules (PyBCKG, Emukit, ABEX, this directory) with `pip` (or add them to PYTHONPATH).

Then export the connection string:

    export BCKG_PRODUCTION_CONNECTION_STRING=...

To initialize a new experimental track run

    python cellsig_pipeline/scripts/init.py -n first_experiment`

A new directory `Experiments/first_experiment` should appear. Modify the configuration files inside. For the `data_config.yml`, you can use the following:
```
initial_experiment_id: de791878-af55-48a8-a356-a0e2da52e0cd
tags:
- level: Experiment
  name: TrackName
  value: FinalTrack
- level: Experiment
  name: Batch
  value: 30
- level: Experiment
  name: TrackNumber
  value: 1
- level: Experiment
  name: Project
  value: CellSignalling
```
which corresponds to an experimental track with several experiments, already present in BCKG.

In the ABEX config we suggest changing the batch size (the default batch size is 0, that is no new experimental samples are proposed) to e.g. 5 and turning off HMC
(both changes will simply speed up the test run, compared to running HMC on 30 new points).

When the files are configured, use

    python cellsig_pipeline/scripts/run_iteration.py --iteration 1 -n first_experiment

to run the first iteration. A new Antha DoE (XLSX) file will be generated. Now, it's the time to do the real experiment and upload the data to BCKG to run the next
iteration. (However, if you used the experimental track described above, you can run another iteration, as the real experimental data -- although corresponding to
a different ABEX config -- already have been uploaded into BCKG).
