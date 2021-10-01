# Python Libraries

This repository is for all Python (and some partly-Python) Station B code. It is organized into a number of subrepositories,
located in the [libraries](libraries/) and [projects](projects/) directories. For details of individual subrepositories,
see their README files.

## Installation

Install [miniconda](https://docs.conda.io/en/latest) if you do not already have it. Then, in either WSL or an
Anaconda command shell, cd to your _Windows_ home directory (not your WSL one, if you are in WSL), and:

``` bash
git clone https://github.com/microsoft/station-b-libraries
cd station-b-libraries/PyStationB
bash scripts/initialize.sh
# (the above command should print out a command to set PYTHONPATH; copy and paste it into your shell).
conda activate PyStationB
```

If you are running a WSL shell, you can also do:

``` bash
cd docs
make html
```

At this point you should be able to open the documentation tree in a web browser.
Start with the full installation guide at
`PATH_TO_REPOSITORY/station-b-libraries/PyStationB/docs/_build/html/source/getting_started/installation.html`
(adjusting the path as appropriate). You can also view the documentation in PyCharm and other IDEs;
in that environment, the installation guide can be found [here](docs/source/getting_started/installation.html).
