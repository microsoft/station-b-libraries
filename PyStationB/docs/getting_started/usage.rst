Usage
=====

The repository consists of multiple *subrepositories* or *subrepos*. These were originally separate repositories
but have been combined into one for the reasons, and approximately in the way, described
`here <https://medium.com/opendoor-labs/our-python-monorepo-d34028f2b6fa>`__. In brief: directories under
``projects/`` and ``libraries/`` are subrepos. The main distinctions between the two categories are:

* Projects can depend on libraries, but not on other projects.
* Libraries cannot depend on projects.
* Libraries can depend on other libraries as long as there are no circularities.
* Both libraries and projects must have a working ``setup.py`` file to allow pip installs.
* We envisage libraries being released separately as open source, but not projects.

If you are planning to add a new project or library to the repository, read this: :ref:`new_subrepos`.

Initializing your environment
-----------------------------

If you are using WSL, you can activate the environment and set the window title with:

.. code-block:: bash

    source scripts/start.sh
             
In a command shell, you need to set your `PYTHONPATH` manually. The required paths can be obtained by doing
```python
python scripts\get_pythonpath.py
```

Tests
-----

A good way to check if everything is working is to run the test suite. You can do this with:

.. code-block:: bash

    make test


Type checking with pyright
~~~~~~~~~~~~~~~~~~~~~~~~~~

The `pyright <https://github.com/Microsoft/pyright>`_ type checker, which we use in our build, is written in
node.js rather than Python and needs special installation if you wish to run it locally (which is recommended).

The commands are as follows. On MacOS and Linux, these need to be run as root; start a root shell with ``sudo bash``.
For Windows command prompt and WSL, they can be run directly.

First try installing pyright directly. It is import to install the right version, as it is under active
development. Look for a line containing ``npm install`` in ``azure-pipelines/build.yml``, and give that
command, omitting the ``sudo`` if you are running on Windows (including WSL). For example:

.. code-block:: bash

    grep 'npm install' azure-pipelines/build.yml
    # The above command prints: "sudo npm install --global pyright@1.1.148", so:
    npm install --global pyright@1.1.148

If this works (e.g. if ``pyright --version`` prints a version number afterwards),
you're done.

If the ``npm`` command is not found, you can install it on Windows by downloading it `here <https://nodejs.org/en/>`_. 
On WSL, Linux or MacOS, install it like this.

.. code-block:: bash

    apt-get install npm
    npm cache clear --force && npm install -g npm

In either case, retry installing pyright afterwards:

.. code-block:: bash

    npm install --global pyright@1.1.148  # use version as determined above

If ``npm`` is found but there is an error message that your version of ``node`` is not recent enough to support pyright, use ``nvm``. 
To install ``nvm``, you can try ``npm install -g nvm``. But sometimes that does not work. An alternative is to use ``curl``:

.. code-block:: bash

    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.34.0/install.sh | bash

Once ``nvm`` is installed, you can proceed as:

.. code-block:: bash

    nvm install stable
    npm install -g pyright@1.1.148  # use version as determined above

If you are still having problems after this, ask a team member for help.

Building the documentation
--------------------------

While RST is a very readable format, it is often more convenient to use an HTML version of the documentation.
After the installation of PyStationB is finished, you can build the documentation using:

.. code-block:: bash

    cd docs
    make html

Then open ``PyStationB/docs/_build/html/index.html`` with a web browser of your choice.

Whenever you want to generate a new version of the documentation (e.g. after you or another contributor introduced some
changes), remember to remove the previous build:

.. code-block:: bash

    cd docs
    make clean
    make html
