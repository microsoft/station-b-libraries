# ABEX
Automated Biological Experimentation (ABEX) uses Gaussian Process models and Bayesian optimization to automatically design experiments for defined objective. 
Typically, Bayesian optimization is suitable for expensive-to-evaluate functions, a property that can almost always be attributed to the design of experiments for biological systems.
Measurements of a biological system in a laboratory can be conducted for a range of *conditions*. We often do not fully understand the mechanisms through those conditions 
affect the measurable behaviour of the biological system. Therefore, an emulator of the input-output behaviour is constructed as a Gaussian Process. This is then used to evaluate
an acquisition function that rewards prospective new evaluations of the system when there is some probability that a new optimum might be found. 

The main case study tackled in this project is:

1. Optimization of biological information processing. This case study attempts to select optimal concentrations of inducing chemical signals to a synthetic gene circuit that perceives those signals and produces fluorescent proteins. The circuits are extensions of the double receiver circuit from Grant et al. (Mol. Syst. Biol. 2016)

## Getting started

### Installation

The installation process is described in great detail in our 
[installation guide](docs/source/getting_started/installation.rst).

### Contribute

Visit the [contribution guide](docs/source/development/contributing.rst).

### View the documentation
In the cloned `ABEX` repository:
```
cd docs
make html
```
And open `_build/html/index.html` (from `ABEX/docs`) using a web-browser.
