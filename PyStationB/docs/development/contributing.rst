.. _contribution_guide:

Contribution guide
==================

Recommended workflow and checklist
----------------------------------

We use `Git workflow with multiple feature branches <https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow>`_.
To start contributing, create a new branch (we recommend the format ``yourusername/tasknumber-featurename``), where
`tasknumber` is the number of the task your pull request (PR) will address. Work items (tasks and higher level)
can be found `here <https://github.com/microsoft/station-b-libraries/issues>`__. If an
appropriate task does not already exist, create one at a sensible place in the tree.
Before you start working on your pull request (PR), read the rest of this page.

When your changes are ready, follow these steps:

* Write tests for all your new or changed code if you have not already done so.
* Run ``make test`` and fix any problems that arise.
* If appropriate, modify the documentation (in the ``docs/`` directory); see "Documentation" below.
* Make sure you have committed all your changes, and ``git push`` your branch
* Create a draft pull request (option "Create as draft" from the "Create" button).
* If you get the message "Merge conflicts", it means the master branch has been updated since you created your
  branch. See "Troubleshooting" below.
* Run a build. This has to be done manually while the PR is a draft; once your PR is published, it will
  be triggered every time you push changes.
* If the build fails on Linux and/or Windows, fix the problems and repeat. This is rare if ``make test`` has run
  successfully, but it can happen; see "Troubleshooting" below for some ideas.
* Fill in an informative title and description for the PR.
* Associate the task whose number you used for the branch name, and possibly others, with the PR.
* Publish the pull request and add one or more reviewers; you may also wish to alert them to your request via Teams.
* The reviewer(s) will review your code, suggesting changes if needed.
* When the reviewers have approved your changes, your PR is ready to be merged.
* We recommend using the "squash merge" option, which helps with Git history readability. You should also usually tick
  to delete the branch, and possibly to complete the work item(s) you attached. If you delete the branch, you
  should also delete it locally with ``git branch -D``.
* Once your PR is merged, ``git pull`` the new version of the master branch to your local machine.

.. _contribution_conventions:

Code quality checks
-------------------
New code must pass the code quality checks carried out by the tests triggered by GitHub actions.
When you publish a new PR, all these tests will be run automatically in the cloud.
Additionally, whenever you do a "git commit" of any python files, they will automatically be run through
`black <https://pypi.org/project/black>`_ and flake8 (see below) as a pre-commit hook.

It is useful to check your code regularly during development. In WSL (Ubuntu), you
have `make <https://en.wikipedia.org/wiki/Make_(software)>`_ available, so you can do:

.. code-block:: bash

    make test

which runs any required tests locally. (See ``Makefile``. The Windows command shell does not have a "make"
command, but there is a script ``make.bat`` which does the same as ``make test``).

.. tip::

   There are three stages at which code can be checked: at commit time with the pre-commit hook
   ("black" and "flake8" checks only); when you run "make"; and in the build. The first two of these are 
   optional. You can remove the pre-commit hook at any time with ``rm .git/hooks/pre-commit`` and 
   reinstate it with ``pre-commit install``; and it is up to you whether and how often you run "make". 
   There is a tradeoff involved: if you run the pre-commit and/or make, you have to wait for them to finish, 
   but you get early feedback on any problems, which is likely to make them easier to resolve. Experiment
   and see what pattern works best for you.

.. tip::

   When you set up your ``PYTHONPATH`` using ``source scripts/start.sh`` (see the installation instructions),
   the paths in it include both the top level of PyStationB and each individual subrepository.
   In the build, a separate, more restricted version of ``PYTHONPATH`` is used to run the tests in each
   subrepository, allowing access only to the subrepository itself and any others listed in the file
   ``internal_requirements.txt`` at the top level of the subrepository. If tests pass on your local machine
   but fail in the build, this may be the reason. You can install a subrepository-specific ``PYTHONPATH``
   by giving the subrepository name to ``scripts/set_pythonpath.sh``, e.g.

   .. code-block::

     source scripts/set_pythonpath.sh ABEX

   If you know you are only going to be working on one subrepository (and possibly its internal_requirements ones)
   you could execute this command beforehand.

Specifically, "make test" will check that your code passes the following steps.

Flake8
^^^^^^

`Flake8 <https://pypi.org/project/flake8/>`_ is a tool helping us to have unified code style across the codebase.
In particular, the maximum line length is 120. All these conventions are implemented in the configuration file ``.flake8``.

To run flake8 just type:

.. code-block::

    flake8

MyPy and Pyright
^^^^^^^^^^^^^^^^

Errors like ``2 + "abcd"`` can be found in C++ or Haskell (compiled statically-typed languages) at compilation time. Python
is an interpreted dynamically-typed language, so that the above expression will raise an exception only when explicitly called
(e.g. this may be a part of a function which wasn't tested!).

To catch these errors, we use `type annotations <https://docs.python.org/3/library/typing.html>`_ and the type
checkers 
`MyPy <http://mypy-lang.org/>`_, and `Pyright <https://github.com/Microsoft/pyright>`, which look for any incompatibilities between the types. 
Please make sure you add type annotations for the arguments and return values of all methods in your code.

The MyPy configuration file is ``mypy.ini``, and there is a configuration stanza for Pyright in `pyproject.toml`.

To run these checks locally, use one or both of:

.. code-block::

    python scripts/mypy_runner.py
    pyright .

.. tip::

    Because Python is a dynamically typed language, type checking is hard and some false positives and false negatives are
    inevitable. To make matters worse, a lot of code in libraries that we make heavy use of, like numpy, matplotlib and pandas, has
    type declarations that are far from complete, and may have return types that vary depending on their arguments and are therefore
    declared in a very general way. It is therefore especially common for spurious type errors to be triggered by code using these
    libraries that is in fact perfectly correct.

    We recommend two strategies here, one preferred one and one fallback, that can be used when you are reasonably sure that
    a reported type error is spurious. The preferred one is to use the ``cast`` function
    from the ``typing`` library; for example, ``return cast(List[str], mylist)`` instead of ``return mylist`` when ``mylist`` has
    been assigned to the result of a library function that may have various return types but we know that on this occasion, the
    result is guaranteed to be of type ``List[str]``.

    The fallback strategy, to be used when ``cast`` is impractical for any reason, is simply to prevent type checkers complaining at all
    about a given line of code, by adding ``# type ignore`` (after a space) to the end of the offending line.

Docstrings
^^^^^^^^^^

Even the cleanest code may be hard to understand without proper documentation. We use
`docstrings <https://www.python.org/dev/peps/pep-0257/>`_
and follow `Google docstring conventions <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

.. tip::

    Try to document everything, including modules and private functions.

    Moreover, if a new public functionality is added (e.g. a function or a class which can be used outside of the module)
    is introduced, it should be added to the module docstring.

    The functionality may also deserve its own tutorial.

We use `interrogate <https://pypi.org/project/interrogate/>`_ to check docstring coverage.
Its configuration is stored in ``pyproject.toml``.

To run this check locally, use

.. code-block:: bash

    interrogate -c pyproject.toml

.. note::

    The field ``fail-under`` in ``pyproject.toml`` specifies the required docstring coverage.
    If your PR improves the docstring coverage, increase this value -- then future PRs can't pass without docstrings!

Unit tests
^^^^^^^^^^

We keep our unit tests in the ``tests/`` directory at top level and within each subrepo, and slower tests in
``slow_tests/``. Slow tests are not checked by the `make` commands or in PR builds, but they are run
as part of a more thorough check on the master branch every weekend.

All code must be tested, except when it isn't, or the build will fail.
"Except when it isn't" means that any untested code must be labelled with "# pragma: no cover",
to alert reviewers and others to its untested status.

.. tip::

    Whenever you want to implement a new functionality, you may want to add a test for it first.
    See `test-driven development <https://en.wikipedia.org/wiki/Test-driven_development>`_ for more information.

To run the unit tests, we use `pytest <https://docs.pytest.org/en/stable/>`_.
In addition, in the CI build we use pytest-cov to
`check the test coverage <https://pypi.org/project/pytest-cov/>`_. We keep configuration information in ``.coveragerc``
files, both at the top level (to cover the top level `tests/` directory) and within every subrepository that has
a `tests/` subdirectory.

.. note::

    The field ``fail_under`` of ``.coveragerc`` specifies the required test coverage as a percentage.
    Generally, this value should be 100.0; we are working towards this. If you create a new subrepository,
    please use this value!

You should get all tests to pass on your local machine before asking for reviews on a PR. So that ``testmon`` can
keep track of both Windows and Linux environments, run tests like this, from the PyStationB directory:

In an Ubuntu (WSL) shell, run:

.. code-block:: bash

    make test

Alternatively, to just run unit tests (rather than flake8 and mypy), run this in WSL:

.. code-block:: bash

    make pytest

This will run tests in WSL, and print a command for you to copy and paste to run them in an Anaconda
(Windows) command shell.

We use the `testmon <http://testmon.org>`_ add-on package for pytest in order to avoid re-running tests that
only access unchanged code. Testmon maintains a database file called ``.testmondata`` to keep track of
what tests access what code. We have a command file ``scripts/run_all_tests.sh`` that maintains one of these files at 
top level and one in every library and project that has tests. They are not part of the git repo and should not be committed - 
the reason for this is that they regularly cause merge conflicts.

When you merge two branches together, typically when doing a ``git merge master`` into your current branch when
someone else has completed a PR, it is in general not possible to predict which tests need to be re-run. You are therefore
recommended to delete all ``.testmondata`` files with a command like

.. code-block:: bash

    rm .testmondata */*/.testmondata

This will cause all tests, rather than just some of them, to be run the next time you call ``scripts/run_all_tests.sh``,
which will take a few minutes. If you are in a hurry, you can leave the ``.testmondata`` files as they are; the downside of 
that, which you may decide is worth it, is that ``scripts/run_all_tests.sh`` may then omit some tests whose results are in
fact no longer valid. In that case, they may fail during the build. If that happens, you should delete the relevant ``.testmondata``
files if you wish your local tests results to be accurate. But since the testmon process is an entirely local one, you can
make whatever choice works best for you.

Copyright notices
^^^^^^^^^^^^^^^^^

Every ``.py`` file in the repository needs to have a copyright notice at the top. These notices are checked and,
if necessary, added, by the command

.. code-block:: bash

    python scripts/check_copyright_notices.py

This script is not run as part of ``make test`` but is run as a step in the PR build, which will fail if
missing copyright notices are detected.

Conda environments and pip requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The way we structure package requirements is a little complex, because of the need to maintain both an environment
that works for the whole repository, and environments that work for each subrepository on its own.

At the top level, ``environment.yml`` defines the ``PyStationB`` conda environment. This file specifies only the
desired versions of python and pip, then delegates everything else to ``requirements_dev.txt``. This in turn
pulls in ``requirements.txt`` and then defines some additional package requirements. The difference between these
two files is (or should be) that ``requirements.txt`` defines everything that is needed at run time, while
``requirements_dev.txt`` also specifies things that are needed for development. The same setup is used within each
subrepository (i.e. each subdirectory of both ``libraries/`` and ``projects/``), with environments named after
the subrepositories.


The top-level requirements lists (both ``requirements.txt`` and ``requirements_dev.txt``) should include all the
contents of their counterparts in the subrepositories, plus anything listed in ``additional_requirements.txt``
and ``additional_requirements_dev.txt`` respectively. If you change any requirements files, you can check this property
still holds by running

.. code-block:: bash

   python scripts/check_requirements.py

If you add a ``-f`` switch, the script will update the top-level requirements files if necessary; otherwise it
will print a message reporting on the current situation. In addition, there is a test,
``tests/test_check_requirements.py``, which runs during the PR build and fails if there is inconsistency.

The version of each package is specified exactly in the requirements files, using ``==``. If ``>=`` is used,
new versions will be used in builds as they are released, and may cause unexpected test failures - however,
if at some point you create a `package` from a subrepository (which is one purpose of maintaining separate
subrepositories), you will probably want to replace ``==`` by ``>=`` to make the package play nicely with the
requirements of other packages.

To update versions in a controlled way, call

.. code-block:: bash

   python scripts/upgrade_requirements.py

This will download the latest versions of all packages and update the requirements files at both top and subrepo
levels; you can then check everything runs correctly and if so, create a PR for them.



Troubleshooting
^^^^^^^^^^^^^^^

Our code base is changing fast. The instructions below are inevitably incomplete and may be out of 
date, so do not hesitate to ask for help if issues are not straightforward to resolve.

If ``make test`` fails in your desktop environment, here are some tips. All commands should be run
from within your ``PyStationB`` directory.

* Have you activated the ``PyStationB`` conda environment? ``conda activate PyStationB`` will do this.

* Is your conda environment up to date? It may well not be if you have just pulled or merged the master
  branch. To update it, do ``conda env update -n``.

* Is your ``PYTHONPATH`` correct? In Linux or WSL, do ``source scripts/set_pythonpath.sh``. In a Windows
  command shell, do ``python scripts/get_pythonpath.py`` and ``set PYTHONPATH=...``, replacing ``...`` by
  the output of the script.

* Is ``pyright`` missing or failing? See the installation page for installation instructions.

If everything looks OK locally (``make test`` shows no issues) but the build fails, here are some things to try.

* If the build cannot be run at all because you get a "Merge conflicts" message in Azure DevOps, you need to resolve
  conflicts and re-push. You can also do this proactively to ensure you are working off the most up to date version
  of the master branch.

  .. code-block:: bash

    git checkout master
    git pull
    git checkout my-branch
    git merge master
    rm .testmondata */*/.testmondata

  The "merge" command will fail, but will tell you which files have conflicts. Edit them to correct the situation
  and "git commit" your changes. Then:

  .. code-block:: bash

    git merge master
    git push

* If ``pyright`` gives different results in the build from on your desktop - possibly in subrepositories you
  have not even modified - it is probably because the latest
  version of ``pyright`` is installed and used in the build, but the one in your local environment has not been 
  updated. To check for this possibility, run the same command as you used to install ``pyright`` initially, 
  e.g. ``npm install -g pyright@1.1.148``, and re-run ``pyright .`` or ``make pyright``.

  Note: in PR builds, though not in slow (weekend) builds, the ``pyright`` step in the build always succeeds
  (returns a zero exit code, and shows green) even if pyright detects errors. This is to prevent pyright false
  alarms from holding up development. However, it is good practice to investigate and either mask out
  (with ``# type: ignore``) or fix any ``pyright`` errors - e.g. after inspecting the weekend build output at the
  start of the week.

* If some library packages are not found, your top-level requirements files may be out of step with those within
  libraries or projects. You can remedy this on desktop with ``python scripts/check_requirements.py -f`` and
  committing and pushing any changes.

* If some files are reported as missing copyright notices, run ``python scripts/check_copyright_notices.py``
  and commit and push any changes.

* If the Linux build passes but the Windows one fails, there may be a Windows-specific issue. Sometimes one of the
  processes in the Windows ``pytest`` step falls over, in which case the Windows build can simply be re-run.
  If this is not the case, start an Anaconda
  shell, run ``conda env update`` to ensure your environment is up to date, then re-run whatever failed in the build.
  
* Rarely, a test may pass on desktop but fail in the build because of an import failure. If this happens, try
  limiting the value of ``PYTHONPATH`` and re-running the test. For example, if an ABEX test fails:

  .. code-block:: bash

    source scripts/set_pythonpath.sh ABEX
    cd libraries/ABEX
    pytest tests/test_file_with_problems.py

  This runs your test with the suitable restricted ``PYTHONPATH``. Fix the situation, then restore the original
  ``PYTHONPATH`` if you wish:

  .. code-block:: bash

    cd ../..
    source scripts/set_pythonpath.sh        \# full PYTHONPATH again

* Tests can take a different amount of time to run in different environments, and a test that passes locally
  may time out in the build. The default timeout is 2 seconds; you can specify a longer time by annotating a test
  with e.g. ``@pytest.mark.timeout(10)`` for 10 seconds.

* A test could be `flaky`: it might exhibit different behaviours on different occasions, for example if it
  generates random numbers without fixing a seed, or if it depends on an outside resource. Flaky tests should
  be rewritten to remove the source of flakiness.

* A test of code that generates plots can fail in the build even when it passes on desktop, because of slightly
  different environments. This is what is happening if you see something like the following in the output of the
  ``pytest`` step of the build:

  .. code-block:: 

    E       AssertionError: assert False
    E        +  where False = figure_found(None, 'test_introduction/plot5_truncated')

  and a few lines later,

  .. code-block::
    
    ----------------------------- Captured stderr call -----------------------------
    CHECK THIS FILE, THEN GIT-ADD OR DELETE IT: tests/data/figures/test_introduction/plot5_truncated/figure006.png
    Copying tests/data/figures/test_introduction/plot5_truncated/figure006.png to directory /home/vsts/work/1/a/Linux/figures/test_introduction/plot5_truncated

  This indicates that a plot file, ``figure006.png``, has been created in a test, and it is different at the binary
  level from all other versions of the file (here, ``figure000.png`` .. ``figure005.png``). It is very likely that
  the differences are invisible to the naked eye and that the new ``png`` file is fine. You can download the file
  from the build output, inspect it, and if appropriate, ``git add`` it to the code base to enable the test
  to pass. For instructions on how to do this, as well as on how to use ``figure_found`` in new tests of plotting
  code, see the docstring of ``figure_found`` in ``libraries/Utilities/psbutils/filecheck.py``.

* The "interrogate" step may fail if you have introduced new code with a smaller than expected proportion of
  docstrings. Add more docstrings! As a last resort, you could reduce the value of ``fail_under`` in
  ``pyproject.toml``, but your reviewer might object to that.

* Similarly, the test coverage check may fail if you have introduced code without enough tests. The expected
  percentage of tested code is specified in the ``fail_under`` field of the ``.coveragerc`` file at top level
  (for ``scripts/``) and in each subrepository. Write more tests! If writing a test is really hard, you may need
  to redesign the code, e.g. by separating logic from input/output. If that is not feasible, you can add
  ``# pragma: no cover`` to untested portions of code; expect your reviewer to question this. As a last resort,
  you could edit the ``.coveragerc`` file; if you do that, expect even more serious questions from your reviewer.

* Note that "# pragma: no cover" can only be attached to portions of code, not whole files. This means you at
  least need to import every non-test file into a test file, even if you do not actually test it. To avoid
  complaints from flake8 about unused imports, you can do something like ``from foo.bar import baz``, and then in a
  test function, ``assert baz is not None``.

Conventions
-----------

Apart from the conventions described above (for which we have automated checks), we recommend
`Google style guide <https://google.github.io/styleguide/pyguide.html>`_.

Documentation
^^^^^^^^^^^^^

To write the documentation, we use `ReStructured Text (RST) <https://en.wikipedia.org/wiki/ReStructuredText>`_ and
`Sphinx <https://www.sphinx-doc.org/>`_.
There is a nice `syntax guide <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_.

Ignoring files in Git
^^^^^^^^^^^^^^^^^^^^^

The convention for local configs, scripts, data folders and other files that you want Git to ignore is the prepend them
with an underscore (e.g. ``_filename.file``).

If not yet ignored by ``.gitignore`` in the sub-directory with that file/folder, add a line to ``.gitignore`` to ignore all
files/folders starting with ``_`` in that sub-directory: ``subdirectory/_*``.
