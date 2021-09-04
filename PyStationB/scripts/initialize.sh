#!/usr/bin/env bash

if [ -z "$(which conda)" ]
then echo Please install miniconda before running this script.
     exit 1
fi
if [ "$(basename $PWD)" != "PyStationB" ]
then echo Please run this script from the PyStationB directory.
     exit 1
fi
git submodule update --init --recursive
# The cache purge is advisable because environment creation/update can otherwise hang during the
# installation of torch, which is needed by Emukit.
pip cache purge
if [ -n "$(conda env list | grep '^PyStationB ')" ]
then
    echo Updating conda environment PyStationB, this may take a few minutes...
    conda env update
else
    echo Creating conda environment PyStationB, this may take a few minutes...
    conda env create
    if [ -z "$(conda env list | grep '^PyStationB ')" ]
    then echo Conda environment creation failed, bailing out
         exit 1
    fi
    echo Conda environment created and activated, installing pre-commit hooks...
    pre-commit install
fi
echo All looks good. Copy and paste these commands in your shell:
echo conda activate PyStationB
echo '# In WSL or Linux:'
echo source scripts/set_pythonpath.sh
echo '# Or in a Windows command shell: set PYTHONPATH to the output of:'
echo python scripts/get_pythonpath.py
echo 'bash scripts/generate_documentation.sh  # optional; takes several minutes to run'
