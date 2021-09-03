.. _arrayshapes:

===================
Array shape checker
===================

Basics
======

Type hints are an essential part of a Python programmer's toolkit, but they are of limited help in understanding code that uses numpy arrays, Torch tensors and similar arraylike objects.
The reader wants to know not just what datatype these objects are, but how many dimensions they have and how the sizes of those dimensions are related. The 
`torchtyping library <https://reposhub.com/python/deep-learning/patrick-kidger-torchtyping.html>`_
is useful in this regard but (unless extended) works only for Torch tensors and can only express a limited set of relationships.
In any case, we want more than just a way to specify shape relationships for code we understand; we want help in figuring out what those relationships are
for unfamiliar code that may have minimal or no documentation, rather in the way that `monkeytype <https://github.com/instagram/MonkeyType>`_ helps with type inference.

The arrayshapes library allows you to do all these things. Suppose you have the following code:

.. code-block:: python

    def foo(a: np.ndarray, b: np.ndarray) -> np.ndarray:
      c = do_something_clever(...)
      return c

And suppose you know that ``a`` must be two-dimensional, ``b`` must be one-dimensional, and the "something clever" is matrix multiplication, yielding a vector. Then you can write:

.. code-block:: python

    def foo(a: np.ndarray, b: np.ndarray) -> np.ndarray:
      c = do_something_clever(a, b)
      Shapes(a, "I,J")(b, "J")(c, "I")
      return c

As in logic programming, variables such as "I" and "J" are bound to the relevant part of the array shape when they are first encountered, and must keep that value throughout
a chain of calls. The ``Shapes`` call will inspect the shapes of the arrays, if possible binding the variables ``I`` and ``J`` to the specific integer values that it finds. 
If this turns out not to be possible - perhaps the single dimension of ``b`` is not the same as the second dimension of ``a``, or ``do_something_clever`` turns out not to be
matrix multiplication after all, so that ``c`` is not a vector with the expected size - then a ``ValueError`` is raised giving details of the issue. Thus the ``Shapes`` line 
serves two purposes: it checks at run time that arguments are legal, and it documents the expected behaviour of the code.

A disadvantage of the above code is that if ``a`` and ``b`` are not related as specified, ``do_something_clever`` may fail before the ``Shapes`` line is reached.
To avoid this, you can check shapes both before and after the call to ``do_something_clever`` while carrying variables over, like this:

.. code-block:: python

    def foo(a: np.ndarray, b: np.ndarray) -> np.ndarray:
      s = Shapes(a, "I,J")(b, "J")  # check a and b are correctly related
      c = do_something_clever(a, b)
      s(c, "I")  # check we got the kind of result we expected; "I" is already bound on entry
      return c

A ``Shapes`` chain can be indexed, and will return the array argument of the relevant member of the chain. So the above could be shortened to:

.. code-block:: python

    def foo(a: np.ndarray, b: np.ndarray) -> np.ndarray:
      s = Shapes(a, "I,J")(b, "J")
      return s(do_something_clever(a, b), "I")[-1]  # return result of do_something_clever(a, b)

or still further to

.. code-block:: python

    def foo(a: np.ndarray, b: np.ndarray) -> np.ndarray:
      return Shapes(a, "I,J")(b, "J")(do_something_clever(a, b), "I")[-1]

The arrays do not have to be of type ``np.ndarray``. Anything with a ``shape`` attribute (for example, a Torch tensor or a Pandas dataframe) 
will be checked in the same way, and anything without a ``shape`` will be converted to a numpy array by wrapping it in ``np.array(...)``
(of course this will fail for objects that cannot be wrapped in this way).

It is possible to specify data types too. There is an optional third argument to ``Shapes`` calls which can be a numpy ``dtype`` or a list of them.
If this is present, the corresponding array (after conversion to numpy if necessary) must have one of the supplied dtypes. For example, we could say 

.. code-block:: python

    def foo(a: np.ndarray, b: np.ndarray) -> np.ndarray:
      Shapes(a, "I,J", np.float64)(b, "J", [np.float64, np.int64])
      ...

As well as simple variables, you can specify arithmetic expressions with the usual four binary operators. "/" is interpreted as integer division.
For example:

.. code-block:: python

    def foo(a: np.ndarray, b: np.ndarray) -> np.ndarray:
      Shapes(a, "I,J", np.float64)(b, "I*J", [np.float64, np.int64])
      ...


Note the following:

  * Variables like these that match a single dimension (we will see another kind in a moment) must be a single letter.
  * A current limitation is that operator precedence rules are not implemented, so you should use parentheses for expressions with several operators:
    write "``(A+B)*C``" or "``A+(B*C)``" rather than "``A+B*C``".
  * The code cannot solve equations, so the first time a variable occurs, it must be on its own, not in an operator expression. Thus:

.. code-block:: python

   Shapes(a, "I,J", np.float64)(b, "J/2")   # Legal; J is already bound when J/2 is encountered
   Shapes(a, "I,J*2", np.float64)(b, "J")   # Illegal; J is unbound when J*2 is encountered

Splice variables
================

Some code can accept arrays with different number of dimensions. `Splice variables` can handle such situations. Unlike the standard variables we have 
seen so far, they consist of two or more letters, and will match any number of dimensions including zero. For example:

.. code-block:: python

    def foo(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return Shapes(a, "I,J,XX")(b, "J")(do_something_clever(a, b), "I,XX")[-1]

In the case of matrix multiplication, ``XX`` will bind to the empty tuple. If ``a`` has four dimensions, it will bind a two-element tuple
whose values are the sizes of the last two dimensions.

If you are trying to understand new code, you can start by writing a ``Shapes`` expression which is guaranteed to succeed, by assigning each array a distinct 
splice variable:

.. code-block:: python

    def foo(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return Shapes(a, "AA")(b, "BB")(do_something_clever(a, b), "CC")[-1]

This will not gain you any insight in itself, but will prepare data for `shape inference`, which will help you to understand how the shapes
are related and to write more specific constraints.

Note:

  * There can only be one splice variable in any one comma-separated string; thus ``"I,XX,YY,J"`` is not allowed.
  * A splice variable must be at top level, not part of an operator expression; ``"I,XX*2"`` is not allowed.

Shape inference
===============

After running some code that makes calls to annotated functions like ``foo``, if you make one of the following calls,

.. code-block:: python

    constraints = Shapes.infer()

    Shapes.infer(output=True)
    Shapes.infer(output=sys.stdout)

then relationships between variables (splice and/or normal) will be inferred. In the first case, the constraints (a list of strings) will be 
returned; in the other cases, which are synonymous, they will also be printed. A typical use case for the latter would be to make the call
at the end of a test, and run it with ``pytest -s``. This example is based on one of the tests in ``test_arrayshapes.py``:

.. code-block:: python

    def alternator(arr):
        """
        Returns an object consisting of the even-numbered members of arr, whatever its type and dimensionality.
        Raises an exception if there is an odd number of members. But let's pretend this function is a black box or
        too complex to understand, and that all we know is that it takes a single array as its argument and returns
        another array.
        """
        assert len(arr) % 2 == 0
        if isinstance(arr, list):
            return [arr[i] for i in range(0, len(arr), 2)]
        return arr[range(0, len(arr), 2)]

    def test_shape_inference_process():
        arrays = [
            np.zeros((4)),
            [0] * 6,                                         # Shapes works with lists
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),  # Shapes works with pandas DataFrames too
            np.zeros((2, 5)),                                # vary number of dimensions
            np.zeros((4, 3, 2)),                             # vary it again
            np.zeros((3,)),                                  # this will fail as 3 is odd, but that's OK
        ]
        Shapes.clear()  # so we only get the bindings from the Shapes call below, not those from other tests
        for arr in arrays:
            try:
                Shapes(arr, "XX")(alternator(arr), "YY")
            except AssertionError:  # triggered by the "assert" in alternator
                pass
        Shapes.infer(output=True)

The output from this would be something like:

.. code-block:: text

    /path-to-repo/station-b-libraries/python/libraries/Utilities/tmp.py:27: 3 constraints from 5 sets of bindings
    /path-to-repo/station-b-libraries/python/libraries/Utilities/tmp.py:27: dims(YY)=dims(XX)
    /path-to-repo/station-b-libraries/python/libraries/Utilities/tmp.py:27: first(YY)=first(XX)/2
    /path-to-repo/station-b-libraries/python/libraries/Utilities/tmp.py:27: prod(YY)=prod(XX)/2

The "``...tmp.py:27:``" prefix identifies the ``Shapes`` call to which the constraints refer; inference 
will be performed on each distinct ``Shapes`` call that has been made, although in this example there 
is only one. The "3 constraints" are those on the three following lines; they are derived from five sets of
(unique) bindings. There is one set of bindings for each successful call in the iteration; the sixth call, with
``np.zeros((3,))``, triggers the ``AssertionError`` in ``alternator``.

The three constraints can be read as follows:

  * ``YY`` has the same number of dimensions as ``XX``; i.e. ``alternator`` returns something with the same number 
    of dimensions (although not necessarily the same shape) as its input.
  * The first dimension of ``YY`` is always half that of ``XX``.
  * The product of the dimensions of ``YY`` is always half that of ``XX``; therefore it is likely that the dimensions after
    the first are the same between ``XX`` and ``YY``.

Given this information, we can rewrite the ``Shapes`` call as:

.. code-block:: python

    Shapes(arr, "I,AA")(alternator(arr), "I/2,AA")

If we do this and re-run the code, we get:

.. code-block:: text

    /path-to-repo/station-b-libraries/python/libraries/Utilities/tmp.py:27: 0 constraints from 5 sets of bindings

which means there were still 5 successful calls (we did not break anything by assuming equal values of ``AA`` in
both places and a halving of the first dimension) and that no further constraints could be derived. In other words, 
we have converged on the best possible description, given the test that we ran.

Bear in mind some limitations of shape inference:

  * There are many constraints that will not be detected; for example, linear relationships between three or more variables.
  * The reported constraints will only be true of the arguments that have been passed to the ``Shapes`` call in question; they
    may not be true of all possible arguments. For example, ``Shapes(a, "X")`` will trigger a constraint ``X=12`` if it is only
    ever passed arrays ``a`` of shape ``(12,)`` in a particular use case, even if the code in question will handle arrays of any
    size. Thus, exercise judgment before accepting and applying a constraint.
