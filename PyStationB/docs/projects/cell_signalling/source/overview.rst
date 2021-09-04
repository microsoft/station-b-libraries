.. _cell-signalling-introduction:

Introduction
------------

.. note::

    A more detailed description of the biological system is available in the
    `Orthogonal Signaling <https://doi.org/10.15252/msb.20156590>`_ paper.

Imagine a bacterium that starts glowing when you feed it appropriate substance.
In our case we feed the bacteria the *C6* substance and we want to observe blue light (meaning that *CFP*,
the Cyan Fluorescent Protein is produced).
There is also another bacterium which can be fed the *C12* substance and that starts producing *YFP*,
the Yellow Fluorescent Protein.

We would like to combine these two into a "traffic lights" bacterium -- when we feed it *C6* it should start producing
*CFP* and when we feed it *C12* it should start producing *YFP*.

Hence, we take a new *E. Coli* bacterial species and copy the DNA from two other species. But this does not work!
If we feed it *C6* it starts producing **both** *YFP* and *CFP*! (Similarly for *C12*).
This behaviour is what we call *crosstalk* and our aim is to eliminate it by adjusting the levels of *C6* and *C12*
we feed into the bacteria. Consider the following objective function:

.. math::

    objective(x, y) = \frac{CFP(C6=x, C12=0) \cdot YFP(C6=0, C12=y)}{CFP(C6=0, C12=y) \cdot YFP(C6=x, C12=0)}

which we call *signal to crosstalk ratio*. To optimize this function (i.e. find a point :math:`(x, y)` that maximizes
the objective) we use Bayesian optimization and Gaussian Processes.

Two more inputs
^^^^^^^^^^^^^^^

In fact there are more degrees of freedom we can control. There are two proteins -- *LuxR* and *LasR* -- which affect
the production rate of *YFP* and *CFP*. *LuxR* can be implicitly controlled by feeding arabinose (*Ara*),
a type of sugar, to the bacteria, while *LasR* can be controlled by using anhydrotetracycline (*ATC*), a type of enzyme.
Although we do not fully understand the functional dependencies between these,
we expect that they follow a sigmoid-like `Hill function <https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)>`_
dependence.

Hence, our objective depends on *four* quantities:

.. math::

    objective(a, b, x, y) = \frac{CFP(Ara=a, ATC=b, C6=x, C12=0) \cdot YFP(Ara=a, ATC=b, C6=0, C12=y)}{CFP(Ara=a, ATC=b, C6=0, C12=y) \cdot YFP(Ara=a, ATC=b, C6=x, C12=0)}


.. _mathematical_modeling:

Data
----

The measurements of the bacterial circuit are originally time-series fluorescence data. Using the Static Characterization library, 
they are converted from time-series to a representative single value for each fluorescent signal. Specifically, we obtain promoter 
activity values for CFP and YFP. These values are then substituted into the objective function. 

Moving between spaces
^^^^^^^^^^^^^^^^^^^^^

.. todo:: This section is missing. It should cover how to use the combinatorics utilities, allowing one to move between ABEX and experimental spaces.


Models
------

.. _mathematical_modeling_two_spaces:

Two spaces
^^^^^^^^^^

Observe that to calculate :math:`objective(a, b, x, y)` we need to combine *two* experiments -- in both of them we
feed :math:`Ara=a` and :math:`ATC=b` values to the bacteria, but one of them has :math:`C6=x` and no *C12*,
and the other has no *C6* and :math:`C12=y`.

We will refer to the procedure of matching experiments together as *matching* or *combinatorics*.

The latter expression comes from the observation that the relationship between both spaces is quite non-trivial:
for example if we fix *a* and *b* and collect observations corresponding to 4 different values of
:math:`x \in \{0, 10, 100, 1000\}` and 4 different values of :math:`y \in \{0, 20, 200, 2000\}`
(in total this gives 8 experiments), we can combine them to get the values of the objective at :math:`3\cdot 3=9`
points!

.. todo::

    Needs clarifying. First, :math:`x=0` and :math:`y=0` are the same experiment, so that means 7 not 8.
    Second, having done those 7 experiments, we have :math:`4x4=16` values for the objective, including those
    for :math:`x=0` and :math:`y=0` - though admittedly those ones are not of real interest.

On the other hand, if we want to collect 8 *random* observations of the objective
(randomly drawing for all :math:`(a, b, x, y)`), we will need to perform :math:`2\cdot 8=16` experiments.
Employing this combinatorial advantage is hard using well-known batch Bayesian optimization techniques,
so as an alternative to vanilla Bayesian optimization we can consider different related problems.

Three-input cell
^^^^^^^^^^^^^^^^

Our main strategy is to consider a three-dimensional space such that :math:`x=y`. In other words, using :math:`c` for
both :math:`x` and :math:`y`, we optimize the function

.. math::

  objective(a, b, c) = \frac{CFP(Ara=a, ATC=b, C6=c, C12=0) \cdot YFP(Ara=a, ATC=b, C6=0, C12=c)}{CFP(Ara=a, ATC=b, C6=0, C12=c) \cdot YFP(Ara=a, ATC=b, C6=c, C12=0)}

