.. _new_subrepos:

Adding a new subrepository
--------------------------

If you are starting a new piece of work that is not an obvious fit for any of the existing subrepositories
(projects and libraries) in PyStationB, you may need to create a new one. Please consult with other team members before
doing this; it may be that your code can go in an existing subrepository, which will save you some work (see below)
and help minimize complexity.

The first decision is whether to make your contribution a project or a library. The distinction is explained
on the :ref:`monorepo-installation` page. Note in particular that code in a project cannot be accessed (depended on) from any
other project or library. Libraries can be accessed from projects, and from other libraries as long as there are no
loops.

The steps involved are the same, except some are optional for projects:

#. Create a subdirectory of ``projects/`` or ``libraries/`` with a name starting with an upper case letter,
   e.g. ``FooBar``. This directory is your subrepo.

#. At top level in the subrepo, create a directory with the same name, but lower cased, e.g. ``foobar``. This    is your *code directory*, where your Python code will go. You can choose another name, or have more than one such
   directory, as long as the name(s) clearly identify the subrepo. For example, ``foobar_core`` and ``foobar_extra``
   would be fine, but ``code`` would not.

#. Create a file ``__init__.py`` in your code directory. An empty file is OK.

#. Create the following files and directories; the easiest way in most cases is to copy them from another project or
   library, making appropriate edits.

   * Directories (possibly empty) ``tests/`` and, if necessary, ``slow_tests/``, for test code. 
     "Slow" tests are those that are likely to take more than about 30 seconds to run.

   * Directory ``docs/``, files ``conf.py``, ``Makefile``, ``make.bat`` and ``index.rst``. In ``conf.py``, change the value
     of ``project`` to be the (descriptive) name of your project or library, and ``modindex_common_prefix`` to
     be the name(s) of your code directory/ies. Edit ``index.rst`` to have an appropriate heading and a line
     in the toctree for each code directory (prefixed by ``_modules/``). Check all is OK by running ``make html``
     from inside the ``docs/`` directory (or in Windows, run ``make.bat``).

   * File ``environment.yml``, changing the value of the ``name`` field to your project name (``FooBar`` in our example).
  
   * File ``requirements_dev.txt`` should contain the line ``-r requirements.txt``, plus specifications (name and,
     if necessary, version constraints) of any packages required (by this subrepo directly, not by its dependents) at
     development time but not at run time.
   
   * File ``requirements.txt`` should contain specifications of any runtime requirements for this subrepo, *not*
     including any other subrepos in PyStationB.
   
   * File ``internal_requirements.txt`` should be a complete list of all the other subrepos (which must be
     libraries, not projects) depended on by this subrepo, directly *or indirectly*. For example, if ``FooBar``
     needs to import code from the ``ABEX`` library, you need to list ``Emukit`` as well as ``ABEX``, because
     ``Emukit`` is an internal requirement of ``ABEX``.
   
   * File ``setup.py`` is required for libraries but optional for projects. Change the ``name`` and ``description``
     fields as required. If the ``long_description`` is set by reading a README file, ensure that file is also
     created.
   
   * File ``.coveragerc`` is required at the top level of your subrepo if you have a ``tests/`` subdirectory. 
     It should specify the name(s) of your (non-test) code subdirectories under ``source`` in the ``[run]`` section, 
     and a value for ``fail_under`` which should be 100.00 unless you can persuade your code reviewer otherwise.
     Example, for the case where ``psbutils`` contains the non-test code:

     .. code-block::

           [run]
           parallel = true
           source = psbutils

           [report]
           show_missing = true
           precision = 2
           fail_under = 100.00

#. From inside your ``FooBar`` directory, create the ``FooBar`` conda environment:

   .. code-block:: bash

      pip cache purge
      conda env create
      conda activate FooBar

   Your code should run correctly from within this environment. If it does not, modify your three requirements
   files as needed.


#. Return to the top level and check that your subrepository is correctly configured by running tests:

   .. code-block:: bash

      cd ../..
      conda activate PyStationB
      pytest
    