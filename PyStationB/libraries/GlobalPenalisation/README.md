# Global Penalization

A repository accompanying the paper on global penalization (using Clark's moment-matching algorithm). Provides implementation of the moment-matched approximation of qEI as well as benchmarks against other Bayesian Optimization acquisition functions.

# Getting Started
Installation on Linux is straightforward. 

1. Create a new conda environment:

   conda env create

2. Install the code in editable mode

   pip3 install -e .

4. To develop the code (e.g. to be able to run unit tests), install additionally:

   pip3 install -r requirements_dev.txt

N.B. To install on Windows, we recommend using WSL2. Running on native windows would require building `jaxlib`, as binaries are not currently available for Windows.

## Code profiling
Use the scripts in the `profiling/` directory.

## Running benchmarks

To run a benchmark on one of the scenarios:

    python run_benchmark.py {scenario name} --batch_method {batch bayesion optimisation method name} --batch_size 5 --num_batches 40 --num_experiments 100

For example:

    python run_benchmark.py branin --batch_method sequential-moment-matched-ei --batch_size 5 --num_batches 40 --num_experiments 100

To see what scenarios and Batch Bayes. Opt. methods are available, run:

    python run_benchmarks.py --help

Note: Some scenarios allow for specifying the dimensionality of the input with a `--scenario_dim` flag, e.g. cosines-additive, cosines-symmetric and cosines-additive-symmetric.

### Generated files

Each run of `run_benchmarks.py` will save the relevant information from the run in an appriopriately named sub-directory in `ExperimentResults`. For instance, running on the `branin` scenario with `local-penalization` batch method with a batch-size of `5` will save the results in

    ExperimentResults/branin/batch_method=local-penalization_batch_size=5

### Plotting results
`run_benchmarks.py` will save the data generated during the benchmark runs. To analyse these, you can visualise convergence of multiple configurations against one another to compare their relative performance.

To visualise convergence of several configurations, run `plot_convergence_comparison.py` with the names of the results directories of the runs that you want to compare. For example, say that you want to compare benchmark runs that were saved to `ExperimentResults/cosines-symmetric/batch_method=local-penalization_batch_size=5_dim=4` and `ExperimentResults/cosines-symmetric/batch_method=sequential-moment-matched-ei_batch_size=5_dim=4` respectively. To do so, you could run:

    python plot_convergence_comparison.py ExperimentResults/cosines-symmetric/batch_method=local-penalization_batch_size=5_dim=4  ExperimentResults/cosines-symmetric/batch_method=sequential-moment-matched-ei_batch_size=5_dim=4

To see other options, such as changing where to save the plot, or axis scales, run:

    python plot_convergence_comparison.py --help

# Contribute

## Disallowed formats
Remember that Mathematica files should be saved to Git in the `.wl` format rather than `.nb` format. (Which doesn't store inputs and additional metadata. Wolfram Language files `.wl` can be opened and edited using Mathematica as ordinary notebooks). Similarly, try to store the code in `.py` files rather than `.ipynb`.

## Tests
We store unit tests in the `tests/` directory. 

Use `pytest` to run them. 

## Code quality
We also have `black` and `flake8` configured for line length 120. 
Use

    pre-commit install

to install them as pre-commit hooks.


Use `make all` to run all checks.
