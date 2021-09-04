Methodology
===========

.. _staticchar-introduction:

Introduction
------------

In synthetic biology we often design genetic circuits and measure their effectiveness. A common method of measuring
gene expression is making the bacteria produce a `fluorescent protein <http://www.scholarpedia.org/article/Fluorescent_proteins>`_.

Experiments are carried out in batches, using multi-well plates, which enables simultaneous characterization of multiple conditions (each well can contain different reagents). 
Then, using a `microplate reader <https://en.wikipedia.org/wiki/Plate_reader>`_ we measure spectrally distinct fluorescent proteins, to give an indication of the activity of the genetic circuit, in response to each condition set.
This results in *time series* data :math:`yellow(t), blue(t), \cdots`.

.. note::

    By convention, we usually name these time series :math:`EYFP(t)`, :math:`ECFP(t)` and so on.
    We don't measure directly the number of proteins, but simply the light
    at specific wavelengths.

We also have the access to the absorbance signal (again, a different one for each well),
which can be used to estimate the cell density as a function of time in each well [4].

As we measure different fluorescence and absorbance signals, instead of e.g. the true cell density, they need to be preprocessed.
In particular, usually there can be a large background signal, which can result from `autofluorescence <https://en.wikipedia.org/wiki/Autofluorescence>`_/absorbance of the growth media or cells. 
One strategy to handle this is to use `blank measurements <https://www.researchgate.net/post/Should_we_subtract_the_media_background_readings_when_measuring_OD_and_fluorescence_with_plate_reader>`_,
but they (in theory) work only if the medium is fixed, which is often not the case when exploring varying conditions.

We usually choose our initial cell densities to be small, so there is no signal from them at the beginning.
More information about data analysis can be found `here <https://openwetware.org/wiki/Endy:Basic_data_analysis_on_a_plate_reader>`_.

.. todo:: Does the media absorbance and fluorescence really change? There is some evidence that it doesn't in fact change in our setting.

Finally, to *characterize* how circuit activity depends on different conditions, static characterization seeks to **summarize the whole time series into a single number**. 

Gradient-based static characterization
--------------------------------------

A particular approach which is based on growth model is *static characterization*, described in [1,2,3].
The method works by calculating the (relative) promoter activity for a promoter driving expression of a fluorescent protein. 
The promoter activity is relative to a control promoter, which constitutively expresses a spectrally distinct fluorescent protein.

The benefits of gradient-based characterization include:
- It doesn't need removing the background fluorescence.
- It quantifies the production rate instead of just emitted light.

As any method, it also has some drawbacks:

- If the cells are not growing, the estimate would be highly imprecise. (We are fitting the curve to an almost symmetric cloud of points,
  so that the slope can be anything -- even negative!)
- The method fits the growth model to the logarithm of normalized cell density. We can't do that in our case as the normalized cell
  density (calculated from the OD signal with subtracted background) is almost zero. (It's possible that it wasn't the case in that experiments
  though).
- This method assumes we know the biochemical parameters of the molecules (as maturation time, degradation rate, ...) are known.
  In practice, it may be hard to get them (and they may possibly depend on conditions).

Integral-based static characterization
--------------------------------------

.. todo::
  Write simple description of integral-based characterization

Growth models
-------------

In the gradient-based method, and optionally in the integral method, a time-window is selected from the full time-series. 
The simplest choice is to always pick the same time-points, e.g. between 4 and 8 hours after initiating the experiment.
However, this ignores the fact that different circuits or different conditions will lead to differences in the growth of the cell culture.
As we are attempting to characterize gene expression in each cell using bulk measurements, variations in cell growth will inevitably introduce a bias.
Therefore, it is desirable to account for these differences and select a time-window based on the state of the culture.

The choice taken in `charmeleon` is to centre the time window based on the time of maximal culture growth rate. 
The window is defined as :math:`(t_m + )`
To identify the time of maximal growth, :math:`t_m`, 

The `bacterial growth <https://en.wikipedia.org/wiki/Bacterial_growth>`_ can be modelled with the sigmoidal/logistic/S-shaped curve,
sometimes also followed by a death phase. Some characterization methods work using the parameters of the fitted curve, in particular
using the inflection point or the initial cell density.

.. note::

    The inflection point of the S-shaped cell density curve corresponds to the time of maximal growth. If we however take the logarithm
    (and get an S-shaped curve as well), its inflection point will correspond to the time of *maximal growth per cell*.

Many growth models are summarized in [5], which is a paper I find very important -- I would suggest reading it at this stage.

Another good paper is

Yoav Ram, Eynat Dellus-Gur, Maayan Bibi et al. *Predicting microbial growth in a mixed culture from growth curve data*,
PNAS July 16, 2019 **116** (29) 14698-14707; `DOI <https://doi.org/10.1073/pnas.1902217116>`_,

which accompanies `Curveball <https://curveball.yoavram.com/>`_, a Python module for growth model fitting.

.. note::

    These papers work via fitting the growth curve to the logarithm of (normalized) cell density. In our case the initial cell
    density is not exactly known, close to zero and it changes between experiments.

.. note::

    We aren't using Curveball at the moment, but it may be handy to use at some point.


Applications
------------
There are many characterizations of the same data possible and they may depend on the purpose of a given genetic device:

- In biomanufacturing, we may be interested in the total amount of protein produced at the end of the process, e.g. after 10 hours.
  Note that this objective implicitly optimizes for cell growth. Also, to give consistent results, one needs to fix the initial cell density
  to be the same for all experiments.

- We may want to have a device that works perfectly (e.g. a biosensor) even if it's hard to grow it. Then we would optimize the ratio of
  produced protein to the cell density.

- As the evaluation at specific time may be noisy, we could average it over some interval (i.e. integrate and rescale),

  
References
----------

1. Yordanov B, Dalchau N, Grant PK, Pedersen M, Emmott S, Haseloff J and Phillips A. *A Computational Method for Automated Characterization of Genetic Components*, ACS Synthetic Biology 2014 **3** (8): 578-588. `DOI <https://pubs.acs.org/doi/abs/10.1021/sb400152n>`__

2. Rudge TR, Brown JR, Federici F, Dalchau N, Phillips A, Ajioka JW and Haseloff J. *Characterization of Intrinsic Properties of Promoters*, ACS Synth. Biol. 2016, **5** (1): 89–98; `DOI <https://doi.org/10.1021/acssynbio.5b00116>`__

3. Brown JR. *A design framework for self-organised Turing patterns in microbial populations*; PhD thesis.

4. Beal J, Farny NG, Haddock-Angelli T, et al. *Robust estimation of bacterial cell count from optical density*. Commun Biol **3** : 512 (2020). `DOI <https://doi.org/10.1038/s42003-020-01127-5>`__

5. Zwietering MH, Jongenburger I, Rombouts FM and van 't Riet K. *Modeling of the Bacterial Growth Curve*, Appl. Environ. Microbiol. 1990 Jun; **56** (6): 1875–1881.   `Link <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC184525/>`__

