.. _monorepo-installation:

Installation
============

First, clone the repository and ``cd`` into it.

.. tip:: It is best to do this in your Windows home directory even if you are running WSL (Ubuntu). 

.. code-block:: bash

    git clone https://github.com/microsoft/station-b-libraries
    cd station-b-libraries/PyStationB

.. warning:: Cloning via SSH does not work properly with our submodules. Please, use HTTPS.

The next step is to install all the other Python packages we need. For dependencies management we use
`miniconda <https://docs.conda.io/en/latest/>`_. Assuming you have already installed it, do this in a
WSL or CMD shell:

.. code-block:: bash

    bash scripts/initialize.sh

.. warning::  
   If running this script you gives a syntax error because of '\\r' characters, run
   ``python scripts/dos2unix.py scripts/*.sh`` and then try again.

.. warning:: If you have trouble running "git commit" after doing this, you can remove the effect of
             the pre-commit with ``pre-commit uninstall``.


Development environments
------------------------
PyCharm
~~~~~~~

If you use `PyCharm <https://www.jetbrains.com/pycharm/>`_, you should set it up as follows:

1. Build the station-b python libraries.
2. File / Settings / Python Interpreter / (Gear icon) / Add...
3. Choose "Conda environment", and select the Python interpreter for the PyStationB environment
4. (Gear icon) / Show All, and select your interpreter again
5. Click on the tree icon ("show paths for the selected interpreter")
6. From a Windows command prompt, run ``python scripts\get_pythonpath.py``, and add ("+" symbol) directory that is listed.

VS Code
~~~~~~~

If you use VS Code, run this from a Windows command prompt: ``python scripts\get_pythonpath.py -j``
and add the line of json that it prints to ``station-b-libraries\PyStationB\.vscode\settings.json``. 

