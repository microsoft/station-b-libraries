# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Provides implementations of different cell models:
    - LatentCell, four-input cell using relative LuxR/LasR levels, C6_on, and C12_on
    - FourInputCell, four-input cell using Ara, ATC, C6_on, and C12_on
    - ThreeInputCell, three-input cell using Ara, ATC, C_on. (Which corresponds to the four input cell
      with C6_on=C12_on).

To pass from LuxR/LasR to Arabinose/ATC, we use Hill functions, implemented as callable classes:
  - HillFunction.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from abex.simulations import SimulatorBase
from emukit.core import ContinuousParameter, ParameterSpace
from psbutils.type_annotations import NDFloat, NumpyCallable
from scipy.special import expit


class LatentCell(SimulatorBase):
    """Model of a Receiver cell, as in `Grant et al., _Orthogonal intercellular signaling for programmed
    spatial behavior_, 2016 <https://www.embopress.org/doi/full/10.15252/msb.20156590>`_.

    It uses 'latent' input variables LuxR and LasR. They are in principle non-measurable.

    This model is deterministic and noise-free at present.

    Note that the range on LuxR and LasR is probably too wide.
    """

    def __init__(self):
        # For the following values see Table S6 of the SI to _Orthogonal Signaling
        self.biochemical_parameters: Dict[str, float] = {
            # Stoichiometry of HSL molecules
            "n": 0.797,
            # Affinities of dimerizations
            "KR6": 2.076e-4,
            "KR12": 4.937e-7,
            "KS6": 1.710e-8,
            "KS12": 8.827e-3,
            # Basal transcription rates
            "a0_76": 0.086,
            "a0_81": 0.264,
            # Transcription rates with regulators bound
            "a1_R": 18.47,
            "a1_S": 8.24,
            # Binding and tetrametrization constants
            "KGR_76": 8.657e-2,
            "KGR_81": 3.329e-3,
            "KGS_76": 4.788e-4,
            "KGS_81": 4.249e-1,
        }

        self._parameter_space = ParameterSpace(
            [
                # LuxR and LasR are in units relative to the 2016 experiment. They are, in principle, non-measurable
                # in a direct manner.
                # C6_on and C12_on are expressed in nM.
                ContinuousParameter("LuxR", 0.0001, 10000),
                ContinuousParameter("LasR", 0.0001, 10000),
                ContinuousParameter("C6_on", 1, 20000),
                ContinuousParameter("C12_on", 1, 20000),
            ]
        )

    @property
    def parameter_space(self) -> ParameterSpace:
        return self._parameter_space

    def _objective(self, X: np.ndarray) -> np.ndarray:
        luxr, lasr, c6_on, c12_on = X.T
        return self._signal_to_crosstalk_ratio(luxr=luxr, lasr=lasr, c6_on=c6_on, c12_on=c12_on)[:, None]

    def _fraction_block(self, c6: np.ndarray, c12: np.ndarray, k6: float, k12: float) -> np.ndarray:
        """In `production_rate` there appear four very similar fractions. This method calculates them."""
        n: float = self.biochemical_parameters["n"]

        numerator = (k6 * c6) ** n + (k12 * c12) ** n
        denominator = (1 + k6 * c6 + k12 * c12) ** n

        return numerator / denominator  # type: ignore

    def _production_rate(
        self,
        luxr: np.ndarray,
        lasr: np.ndarray,
        c6: np.ndarray,
        c12: np.ndarray,
        kgr: float,
        kgs: float,
        a0: float,
    ) -> np.ndarray:
        """Production (transcription and translation together) rate of a protein controlled by gene G is given by
        eq. (27) in C.2 of SI to Grant et al., _Orthogonal signaling_, 2016.

        Args:
            luxr: LuxR concentration, relative to reference concentration. (I.e. values between 0.1 and 100).
            lasr: LasR concentration, relative to reference concentration. (I.e. values between 0.1 and 100).
            c6: 3OC6HSL concentration in nM
            c12: 3OC12HSL concentration in nM
            kgr: affinity of considered gene G for LuxR regulator
            kgs: affinity of considered gene G for LasR regulator
            a0: basal transcription rate of gene G

        Returns:
            production rate of gene G

        Notes
        -----
        Remember that `kgr`, `kgs` and `a0` depend on the gene G to be expressed.
        """
        # Change variable names
        r, s = luxr, lasr

        fraction_block_r = self._fraction_block(
            c6=c6,
            c12=c12,
            k6=self.biochemical_parameters["KR6"],
            k12=self.biochemical_parameters["KR12"],
        )
        fraction_block_s = self._fraction_block(
            c6=c6,
            c12=c12,
            k6=self.biochemical_parameters["KS6"],
            k12=self.biochemical_parameters["KS12"],
        )

        # Fractions must be multiplied by appropriate factors (binding and tetramizeration constants and LuxR/LasR
        # concentrations)
        block_r: np.ndarray = kgr * r ** 2 * fraction_block_r
        block_s: np.ndarray = kgs * s ** 2 * fraction_block_s

        denominator: np.ndarray = 1 + block_r + block_s

        a1_r: float = self.biochemical_parameters["a1_R"]
        a1_s: float = self.biochemical_parameters["a1_S"]
        numerator: np.ndarray = a0 + a1_r * block_r + a1_s * block_s  # type: ignore

        return numerator / denominator

    def _production_rate_yfp(self, luxr: np.ndarray, lasr: np.ndarray, c6: np.ndarray, c12: np.ndarray) -> np.ndarray:
        """We measure Las81 by presence of eYFP, which is fluorescent.

        Args:
            luxr: LuxR concentration, relative to reference concentration. (I.e. values between 0.1 and 100).
            lasr: LasR concentration, relative to reference concentration. (I.e. values between 0.1 and 100).
            c6: 3OC6HSL concentration in nM.
            c12: 3OC12HSL concentration in nM.

        Returns:
            np.ndarray: production rate of YFP

        Notes
        -----
        Remember that this is production rate instead of ratiometric response. (Ratiometric response measures the ratio
        of production rates. The standard reference protein is mRFP1, i.e. the chromosomal RFP activity.
        However, assuming constant conditions, these values are proportional, so the additional factor cancels out when
        we take the signal-to-crosstalk ratio.
        """
        return self._production_rate(
            luxr=luxr,
            lasr=lasr,
            c6=c6,
            c12=c12,
            kgr=self.biochemical_parameters["KGR_81"],
            kgs=self.biochemical_parameters["KGS_81"],
            a0=self.biochemical_parameters["a0_81"],
        )

    def _production_rate_cfp(self, luxr: np.ndarray, lasr: np.ndarray, c6: np.ndarray, c12: np.ndarray) -> np.ndarray:
        """We measure Lux76 by presence of eCFP, which is fluorescent.

        Args:
            luxr: LuxR concentration, relative to reference concentration. (I.e. values between 0.1 and 100).
            lasr: LasR concentration, relative to reference concentration. (I.e. values between 0.1 and 100).
            c6: 3OC6HSL concentration in nM.
            c12: 3OC12HSL concentration in nM.

        Returns:
            production rate of CFP

        Notes
        -----
        Remember that this is production rate instead of ratiometric response. (They are proportional in constant
        conditions -- they are related by chromosomal RFP activity).
        """
        return self._production_rate(
            luxr=luxr,
            lasr=lasr,
            c6=c6,
            c12=c12,
            kgr=self.biochemical_parameters["KGR_76"],
            kgs=self.biochemical_parameters["KGS_76"],
            a0=self.biochemical_parameters["a0_76"],
        )

    def _get_on_off_production_rates(
        self, luxr: np.ndarray, lasr: np.ndarray, c6_on: np.ndarray, c12_on: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Helper function for getting the 4 target fluorescent protein concentrations of interest."""
        signal_off = np.zeros_like(c6_on)

        cfp_c6 = self._production_rate_cfp(luxr=luxr, lasr=lasr, c6=c6_on, c12=signal_off)
        cfp_c12 = self._production_rate_cfp(luxr=luxr, lasr=lasr, c6=signal_off, c12=c12_on)

        yfp_c6 = self._production_rate_yfp(luxr=luxr, lasr=lasr, c6=c6_on, c12=signal_off)
        yfp_c12 = self._production_rate_yfp(luxr=luxr, lasr=lasr, c6=signal_off, c12=c12_on)
        return cfp_c6, cfp_c12, yfp_c6, yfp_c12

    def _signal_to_crosstalk_ratio(
        self, luxr: np.ndarray, lasr: np.ndarray, c6_on: np.ndarray, c12_on: np.ndarray
    ) -> np.ndarray:
        """Signal to crosstalk ratio as given by eq. (2) of _Orthogonal signaling_.

        Args:
            luxr: LuxR concentration, relative to reference concentration.
            lasr: LasR concentration, relative to reference concentration.
            c6_on: 3OC6HSL concentration, when this signal is sent. Concentration given in nM.
            c12_on: 3OC12HSL concentration, when this signal is sent. Concentration given in nM.

        Returns:
            np.ndarray: signal to crosstalk ratio
        """
        cfp_c6, cfp_c12, yfp_c6, yfp_c12 = self._get_on_off_production_rates(
            luxr=luxr, lasr=lasr, c6_on=c6_on, c12_on=c12_on
        )
        return self._crosstalk_ratio_from_signals(cfp_c6=cfp_c6, cfp_c12=cfp_c12, yfp_c6=yfp_c6, yfp_c12=yfp_c12)

    def _crosstalk_ratio_from_signals(
        self, cfp_c6: np.ndarray, cfp_c12: np.ndarray, yfp_c6: np.ndarray, yfp_c12: np.ndarray
    ) -> np.ndarray:
        """Compute the signal to cross-talk ration from the fluorescent signal measurements.

        Args:
            cfp_c6 (np.ndarray): Ratiometric CFP signal measurement (in response to C6)
            cfp_c12 (np.ndarray): Ratiometric CFP signal measurement (in response to C12)
            yfp_c6 (np.ndarray): Ratiometric YFP signal measurement (in response to C6)
            yfp_c12 (np.ndarray): Ratiometric YFP signal measurement (in response to C12)

        Returns:
            np.ndarray: Signal to cross-talk ratio
        """
        return (cfp_c6 * yfp_c12) / (cfp_c12 * yfp_c6)  # type: ignore


@dataclass
class HillFunction:
    """This callable object implements a Hill function with prescribed parameters.

    Attributes:
        n (float): Hill coefficient
        K (float): ligand concentration producing half occupation
        scale (float): Hill function is rescaled by this value

    Notes:
        See `wikipedia <https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)>`_.
    """

    n: float  # Hill coefficient
    K: float  # ligand concentration producing half occupation
    scale: float = 1  # if there is a need to rescale the output

    def __call__(self, x: NDFloat) -> NDFloat:
        """Applies hill function to the inputs.

        Returns
            array of the same shape (or float)
        """
        adc: float = self.K ** self.n  # the apparent dissociation constant
        xn = x ** self.n
        return self.scale * xn / (adc + xn)  # type: ignore


class FourInputCell(SimulatorBase):
    """This cell is controlled by:
        - Ara (arabinose) which induces LuxR (in range 0.01-100 mM),
        - ATC (anhydrotetracycline) which induces LasR (in range 0.001-10 ng/ml),
        - C6 and C12 as usual (in range 1-25000 nM).

    We move from Ara and ATC space to LuxR and LasR using Hill functions.
    """

    # Bounds for the growth factor curve - these will determine the space over which the growth factor changes
    _growth_factor_bounds = ((1e-2, 1e2), (1e-3, 1e1), (1, 2.5e4), (1, 2.5e4))
    # If heteroscedastic noise specified, it will increase smoothly from 1 to _max_heteroscedastic_noise_multiplier
    # with a mid-point of the transition being at location _heteroscedastic_noise_transition.
    _max_heteroscedastic_noise_multiplier = 10.0
    _heteroscedastic_noise_transition = np.array((90, 8, 140000, 3000))

    def __init__(
        self,
        use_growth_in_objective: bool = True,
        noise_std: Optional[float] = None,
        luxr_transfer_func: NumpyCallable = None,
        lasr_transfer_func: NumpyCallable = None,
        growth_penalty_func: NumpyCallable = None,
        heteroscedastic: bool = False,
    ):
        """
        Attributes:
            use_growth_in_objective (bool): if True, objective is signal to crosstalk ratio multiplied by penalty
                that grows with the input concentrations
            noise_std: The standard deviation of the multiplicative noise on the output. No noise added if None.
            luxr_transfer_func (Callable): conversion from Ara to LuxR
            lasr_transfer_func (Callable): conversion from ATC to LasR
            growth_penalty_function (Callable): A 1D function that penalizes high values of a given input.
            heteroscedastic (bool): Whether to vary the noise variance across the input space (with a determinstic
                function)
        """
        self.latent_cell = LatentCell()
        self._parameter_space = ParameterSpace(
            [
                ContinuousParameter("Ara", 1e-4, 1e4),
                ContinuousParameter("ATC", 1e-4, 1e4),
                ContinuousParameter("C6_on", 1e-3, 2.5e5),
                ContinuousParameter("C12_on", 1e-3, 2.5e5),
            ]
        )

        self._luxr_transfer_func = luxr_transfer_func or self._default_hill_luxr()
        self._lasr_transfer_func = lasr_transfer_func or self._default_hill_lasr()
        self._growth_penalty = growth_penalty_func or self._default_growth_penalty

        self._incorporate_growth: bool = use_growth_in_objective
        self._noise_std = noise_std
        self._is_heteroscedastic = heteroscedastic

    @staticmethod
    def _default_hill_luxr() -> HillFunction:
        return HillFunction(n=2, K=20, scale=100)

    @staticmethod
    def _default_hill_lasr() -> HillFunction:
        return HillFunction(n=2, K=5, scale=10)

    @staticmethod
    def _default_growth_penalty(x: np.ndarray) -> np.ndarray:
        """Assume that `x` is normalized to [0, 1]. We are looking for a function such that f(x)=1 for small
        values, and then decreases to f(1)=0."""
        return 1 - HillFunction(n=10, K=0.5, scale=1)(x)

    def _growth_factor(self, X: np.ndarray) -> np.ndarray:
        """Returns growth factors. Shape (n_samples, 1)."""
        # Estimate maximal values of the controllable and latent parameters
        max_ara, max_atc, max_c6_on, max_c12_on = [upper for lower, upper in self._growth_factor_bounds]
        max_luxr, max_lasr = self._luxr_transfer_func(max_ara), self._lasr_transfer_func(max_atc)

        # Rescale the latent space variables to 'relative' scale, i.e. [0, 1].
        ara, atc, c6_on, c12_on = X.T
        luxr, lasr = self._luxr_transfer_func(ara), self._lasr_transfer_func(atc)

        rel_luxr, rel_lasr = luxr / max_luxr, lasr / max_lasr
        # Assume growth factor depends on the number of produced LuxR and LasR proteins...
        growth_factor_proteins = self._growth_penalty((rel_luxr + rel_lasr) / 2.0)
        # ... and that there is also some toxicity associated to C6/C12 signals.
        growth_factor_c6 = self._growth_penalty(c6_on / max_c6_on)
        growth_factor_c12 = self._growth_penalty(c12_on / max_c12_on)

        # Take a harmonic mean of growth factors of both cells
        growth_factor = growth_factor_proteins * np.sqrt(growth_factor_c6 * growth_factor_c12)

        return growth_factor[:, None]

    def _heteroscedastic_noise_std_multiplier(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Returns a coefficient between [1, self._max_heteroscedastic_noise_multiplier] by which to multiply
        the noise standard deviation if heteroscedastic noise is specified.

        This is just an arbitrary function chosen so that the noise variance increases past the predetermined
        threshold.

        Args:
            x: The array of input concentrations of shape [n_inputs, 4] (with columns representing Ara, ATC, C6_on and
                C12_on concentrations).

        Return:
            Array of shape [n_inputs, 1] to multiply the noise standard deviation by.
        """
        # Calculate a normalised logit. This ensures that sigmoid interpolates from  roughly 0.047 to 0.95
        # over one order of magnitude difference in input
        logits = 6 * (np.log10(x) - np.log10(self._heteroscedastic_noise_transition))  # type: ignore
        # Smoothly increase the noise multiplier from 1 to _max_heteroscedastic_noise_multiplier based on the value
        # of the sigmoid
        noise_multiplier = (self._max_heteroscedastic_noise_multiplier - 1.0) * expit(logits) + 1
        return noise_multiplier.mean(axis=1, keepdims=True)

    @property
    def parameter_space(self) -> ParameterSpace:
        return self._parameter_space

    def _get_outputs(self, X: np.ndarray) -> np.ndarray:
        """
        Pass from Ara/ATC space to LuxR/LasR space and use the LuxR/LasR cell to get the
        ratiometric fluorescent protein observations (outputs).
        """
        ara, atc, c6_on, c12_on = X.T

        luxr, lasr = self._get_latent_concentrations(ara=ara, atc=atc)

        # Get output signals from the "latent" cell
        cfp_c6, cfp_c12, yfp_c6, yfp_c12 = self.latent_cell._get_on_off_production_rates(
            luxr=luxr, lasr=lasr, c6_on=c6_on, c12_on=c12_on
        )

        if self._incorporate_growth:
            # Apply growth factor to the signals (non-crosstalk terms)
            cfp_c6 *= np.sqrt(self._growth_factor(X)).ravel()
            yfp_c12 *= np.sqrt(self._growth_factor(X)).ravel()

        outputs = np.stack((cfp_c6, cfp_c12, yfp_c6, yfp_c12), axis=1)

        # Apply noise to each of the outputs if noise specified
        if self._noise_std:  # pragma: no cover
            if self._noise_std < 0.0:
                raise ValueError("Standard deviation cannot be negative.")  # pragma: no cover
            # Get the paramaters for the lognormal samples so that the standard deviation of the cross-talk ratio is
            # equal to self._noise_std. Note, the mean of the noise on each output won't be exactly 1.0, but the
            # mean of the noise on the objective will be 1.0 as desired.
            mean, sigma = lognormal_natural_scale_param_to_log_scale(1.0, self._noise_std)
            # 4 is the number of outputs in the simulator, so the variance per output should be
            # total_variance / num_outputs. Sqrt all of this for sigma.
            per_output_sigma = sigma / 2.0
            if self._is_heteroscedastic:
                # Increase noise standard deviation more in some regions of the input space
                per_output_sigma = per_output_sigma * self._heteroscedastic_noise_std_multiplier(X)  # pragma: no cover

            lognormal_samples = np.random.lognormal(mean=mean, sigma=per_output_sigma, size=outputs.shape)
            outputs *= lognormal_samples
        return outputs

    def _get_latent_concentrations(self, ara: np.ndarray, atc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pass from Ara/ATC space to LuxR/LasR space.

        Returns:
            Tuple[np.ndarray, np.ndarray]: LuxR and LasR activity within the cell
        """
        luxr: np.ndarray = self._luxr_transfer_func(ara)
        lasr: np.ndarray = self._lasr_transfer_func(atc)
        return luxr, lasr

    def _objective(self, X: np.ndarray) -> np.ndarray:
        """Pass from Ara/ATC space to LuxR/LasR space and use the LuxR/LasR cell."""
        outputs = self._get_outputs(X)
        cfp_c6, cfp_c12, yfp_c6, yfp_c12 = outputs.T
        objective = self.latent_cell._crosstalk_ratio_from_signals(
            cfp_c6=cfp_c6, cfp_c12=cfp_c12, yfp_c6=yfp_c6, yfp_c12=yfp_c12
        )

        return objective


class ThreeInputCell(FourInputCell):
    """A three-input thin wrapper around a four input cell, assuming C6_on = C12_on.

    This cell is controlled by:
        - Ara (arabinose) which induces LuxR (in range 0.01-100 mM),
        - ATC (anhydrotetracycline) which induces LasR (in range 0.001-10 ng/ml),
        - C_on, which is the concentration of both C12 and C6 used to calculate signal-to-crosstalk
            (in range 1-25000 nM).
    """

    def __init__(
        self,
        use_growth_in_objective: bool = True,
        noise_std: Optional[float] = None,
        luxr_transfer_func: NumpyCallable = None,
        lasr_transfer_func: NumpyCallable = None,
        growth_penalty_func: NumpyCallable = None,
        heteroscedastic: bool = False,
    ):
        """
        Args:
            use_growth_in_objective (bool): if True, objective is signal to crosstalk ratio multiplied by penalty
                that grows with the input concentrations
            noise_std: The standard deviation of the multiplicative noise on the output. No noise added if None.
            luxr_transfer_func (Callable): conversion from Ara to LuxR
            luxr_transfer_func (Callable): conversion from ATC to LasR
            growth_penalty_function (Callable): A 1D function that penalizes high values of a given input.
            heteroscedastic (bool): Whether to vary the noise variance across the input space (with a deterministic
                function)
        """
        self.four_input_cell = FourInputCell(
            use_growth_in_objective=use_growth_in_objective,
            noise_std=noise_std,
            luxr_transfer_func=luxr_transfer_func,
            lasr_transfer_func=lasr_transfer_func,
            growth_penalty_func=growth_penalty_func,
            heteroscedastic=heteroscedastic,
        )
        # Override the parameter space to a 3 input one.
        self._parameter_space = ParameterSpace(
            [
                ContinuousParameter("Arabinose", 1e-4, 1e4),
                ContinuousParameter("ATC", 1e-4, 1e4),
                ContinuousParameter("Con", 1e-3, 2.5e5),
            ]
        )

    def _get_outputs(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        ara, atc, c = X.T

        X_new = np.vstack((ara, atc, c, c)).T

        return self.four_input_cell._get_outputs(X_new)

    def _objective(self, X: np.ndarray) -> np.ndarray:
        """Separate C into C6 and C12 and use the internally-stored four-input cell."""
        ara, atc, c = X.T

        X_new = np.vstack((ara, atc, c, c)).T

        return self.four_input_cell._objective(X_new)


def lognormal_from_mean_and_std(
    mean: NDFloat,
    std: NDFloat,
    size: Union[int, Sequence[int]],
) -> np.ndarray:
    """Generate samples from a log-normal distribution with a given mean and standard deviation.

    A useful utility, as the numpy.random.lognormal() function taken the mean and standard deviation of the
    underlying normal distribution as parameters. This function calculates these parameters from the desired mean
    and std. of the final log-normal.

    Formulae taken from: https://en.wikipedia.org/wiki/Log-normal_distribution#Generation_and_parameters

    Args:
        mean (float): The mean of the log-normal variable
        std (float): The standard deviation of the log-normal variable
        size (Union[int, Sequence[int]]): [description]
    Return:
        np.ndarray: Samples from a log-normal distribution with the specified size

    Raises:
        ValueError: If either mean or standard deviation parameters are not positive.
    """
    normal_mean, normal_sigma = lognormal_natural_scale_param_to_log_scale(mean, std)
    return np.random.lognormal(mean=normal_mean, sigma=normal_sigma, size=size)


def lognormal_natural_scale_param_to_log_scale(mean: NDFloat, std: NDFloat) -> Tuple[NDFloat, NDFloat]:
    """
    Converts from the parametrization of a lognormal distribution by the mean and standard deviation in the
    natural space (i.e. the actual mean and standard deviation of the log-normal variable) to the parametrization
    in the log-space (i.e. to the mean and standard deviation parameters of the underlying normal distribution).

    Raises:
        ValueError: If either mean or standard deviation parameters are not positive.
    """
    if not np.all(mean > 0):
        raise ValueError("Mean of a log-normal variable must be positive.")  # pragma: no cover
    if not np.all(std > 0):
        raise ValueError("Standard deviation must be positive.")  # pragma: no cover
    normal_mean = np.log((mean ** 2) / np.sqrt((mean ** 2) + (std ** 2)))
    normal_sigma = np.sqrt(np.log(1 + (std ** 2) / (mean ** 2)))
    return normal_mean, normal_sigma  # type: ignore
