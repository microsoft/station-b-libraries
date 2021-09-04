.. _configs:

Configs
=======

For an overview of how to design configs for the :ref:`cell_signalling` experiments, see: :ref:`cell-signalling-configs`. 
That section also includes an example config, which can act as a guide on what a typical config might look like, and
what fields ought to be filled in.

Otherwise, refer to `abex/settings.py` for detailed documentation on what each field's purpose is.

.. todo::

    Consider using `Sphinx Pydantic <https://sphinx-pydantic.readthedocs.io/en/latest/>`_, to automatically generate a nice sphinx documentation for all the fields.


.. todo::

    Gene ID is still its own unique entry in the config. Make it into a categorical variable.  
    

.. _expanding_configs:

Expanding configs
-----------------

ABEX has a handy short-cut for running under multiple configurations of hyperparameters with only one config file.
This can be done by "expanding" the config.

To give an example, let's say that you would like to compare ABEX outputs using two kernels ("RBF" and "Matern), with otherwise identical configs. 
This can be done with a single config file using the following syntax:

.. code-block:: yaml

    kernel: ["@my_label", "RBF", "Matern"]

where `my_label` can be any string consisting of lower case letters a to z and (except in first position) underscores.
This string will be used when naming the results folders.

In more detail, a resolution is defined as follows:

* A choice list is a Python list of more than two elements, whose first element is a list starting with a string
  whose first character is "@" and is followed by one of the characters a to z and then zero or more occurrences
  of a to z or _.
  Note that in YAML, you write such a list in the form :code:`["@x", foo, bar]`, i.e. the :code:`"@x"` has to be in
  quotes because :code:`@` is not a legal first character for a YAML value.

* A resolution of a choice list is a resolution of one of its non-initial elements.

* A resolution of any other list is a list of resolutions of its elements.

* A resolution of a dictionary is another dictionary, with each value replaced by one of its resolutions.

* The (single) resolution of any other object - anything that is not a list or a dictionary - is the object itself.

If two choice lists have the same initial element all the resolutions are constrained to have the corresponding
elements selected. All such choice lists must be the same length.

Examples:
 -        :code:`"foo"` has a single resolution, :code:`"foo"`
 -        :code:`["@x", "a", 2]` has two resolutions, :code:`"a"` and :code:`2`
 -        :code:`{"p": ["@x", "a", 2], "q": ["@y", "b", 3]}` has the same four resolutions: :code:`"@x"` and
          :code:`"@y"` are distinct so all 2x2=4 combinations are valid
 -        :code:`{"p": ["@x", "a", 2], "q": ["@x", "b", 3]}` has two resolutions, because corresponding elements are
          selected: :code:`{"p": "a", "q": "b"}, {"p": 2, "q": 3}`

