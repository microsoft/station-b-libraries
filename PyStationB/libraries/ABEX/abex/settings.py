# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Settings, a module allowing to describe what data to use, model to build and optimization step to perform.

Exports:
    OptimizerConfig, the main configuration file providing enough information to make any optimization run we support
    simple_load_config, creates a new OptimizerConfig from a YAML file. The simplest function possible without side
        effects.
    make_config_from_dict, appends some information and dumps it into the results directory.
        It's *not* an alternative to ``OptimizerConfig(**dictionary)``.
    load, a high-level function wrapping ``make_config_from_dict`` and ``simple_load_config``. Should not be used for
        configs containing resolutions (using the "@" syntax)
    load_resolutions, it's the ``load`` method for configs with resolutions
"""
import copy
import enum
import json
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union, cast

import GPy
import pydantic
import yaml
from abex.constants import SLSQP
from abex.data_settings import DataSettings, ParameterConfig, FloatIntStr
from abex.expand import expand_structure_with_resolutions
from abex.gpy import InverseGamma

from abex.space_designs import DesignType
from emukit.bayesian_optimization.acquisitions import EntropySearch, ExpectedImprovement, NegativeLowerConfidenceBound
from emukit.bayesian_optimization.acquisitions.expected_improvement import MeanPluginExpectedImprovement
from emukit.core.acquisition.acquisition import Acquisition
from psbutils.type_annotations import PathOrString

# TODO: we want this to be just Union[float, str] but pydantic has (or had) trouble with that. Check and if possible
# TODO: remove the "int".

# When searching recursively downwards to look for config files, visit at most this number of directories by default.
MAX_DIRECTORIES_TO_SEARCH_FOR_CONFIG = 1000

BATCH_CSV = "batch.csv"
BATCH_FOLD_CSV_BASE = "batch_fold{}.csv"

RESULTS_DIR = "Results"


class NonNegativeInt(pydantic.ConstrainedInt):
    """Type representing a non-negative integer.

    Although pydantic's conint may be considered a cleaner solution (less new types), MyPy doesn't properly handle it
    right now.
    """

    ge = 0


class OpenUnitInterval(pydantic.ConstrainedFloat):
    """Type representing a float from the open interval (0, 1)."""

    gt = 0
    lt = 1


class AcquisitionClass(Enum):
    """Enumeration of supported acquisition functions used for Bayesian optimization"""

    EXPECTED_IMPROVEMENT = ExpectedImprovement
    MEAN_PLUGIN_EXPECTED_IMPROVEMENT = MeanPluginExpectedImprovement
    UCB = NegativeLowerConfidenceBound
    ENTROPY_SEARCH = EntropySearch


class KernelFunction(enum.Enum):
    """Enumeration of supported kernels for the Gaussian process covariance function"""

    RBF = GPy.kern.RBF
    RQ = GPy.kern.RatQuad
    Matern = GPy.kern.Matern52


class PriorDistribution(enum.Enum):
    """Enumeration of supported prior distributions"""

    GAMMA = GPy.priors.Gamma
    INVERSEGAMMA = InverseGamma
    UNIFORM = GPy.priors.Uniform
    GAUSSIAN = GPy.priors.Gaussian


class OptimizationStrategy(str, Enum):
    """
    Implemented optimization strategies. "Implemented" means the value of each member of the enum should be
    the value of strategy_name in a subclass of OptimizerBase.
    """

    ZOOM = "Zoom"
    BAYESIAN = "Bayesian"


class BatchAcquisitionStrategy(Enum):
    """Implemented optimization strategies."""

    LOCAL_PENALIZATION = "LocalPenalization"
    MMEI = "MomentMatchedEI"


class AcquisitionOptimizer(Enum):
    """Implemented optimization strategies."""

    GRADIENT = "Gradient"
    RANDOM = "Random"


class PriorConfig(ParameterConfig):
    """A class to collect settings for a continuous input parameter.

    Attributes:
        lower_bound, upper_bound: The constraints on the inputs to be passed to the Bayesian Optimization procedure.
            Defined in the original input-space (these will be passed through pre-processing transforms). If not given
            the lower_bound and upper_bound will be inferred as the minimum and maximum of the values in the data
            provided.
        default_condition: NaN values for this input will be replaced with this value
        drop_if_nan: If set to True, if the value of the input is NaN, the entire row of data will be dropped
    """

    distribution: str
    parameters: Dict[str, float] = pydantic.Field(default_factory=dict)

    @pydantic.validator("distribution")
    def validate_distribution_name(cls, value: str) -> str:  # pragma: no cover
        """Validates if provided distribution is implemented in ABEX.

        See `pydantic validators <https://pydantic-docs.helpmanual.io/usage/validators/>`_ for more information.
        """
        return validate_enum(value.upper(), PriorDistribution)

    def get_prior(self) -> GPy.priors.Prior:  # pragma: no cover
        """Return a GPy prior"""
        prior_constructor = PriorDistribution[self.distribution.upper()].value
        return prior_constructor(**self.parameters)


class BinnedSlicesConfig(pydantic.BaseModel):
    """Settings for a binned slices plot to make (see `abex.plotting.composite_core.slices1d_with_binned_data).

    Attributes:
        dim_x: The index of the dimension to put on the x-axis (slices of the objective will be plotted against
            this input)
        slice_dim1: The index of the 1st dimension to make the bins for
        slice_dim2: The index of the 2nd dimension to make the bins for
        num_bins: The number of bins to divide each of slice_dim1 and slice_dim2 dimensions into. num_bins**2
            subplots will be made as results
    """

    dim_x: int = 0
    slice_dim1: int = 1
    slice_dim2: int = 2
    num_bins: int = 4


class TrainSettings(pydantic.BaseModel):
    """
    A class to collect configuration options associated with model training and analysis

    Attributes:
        iterations (int): Number of iterations used by the optimizer of the model parameters.
            If hmc is True, the number of hmc samples will be (iterations / 5).
        hmc (bool): Use Hamiltonian Monte Carlo (HMC) to compute a posterior over the kernel hyperparameters
        num_folds (int): Number of folds to use for cross-validation
        compute_optima (bool): Compute the optimum of the mean function. Must be True
            if plot_slice1d or plot_slice2d is on
        optim_samples (int): Number of random samples to initialize the optimizer
        optim_method (str): Optimization method (passed to scipy.optimize)
        plot_slice1d (bool): Create plots of the input-output behaviour of the GP for each input. compute_optima
            must be on if set to True.
        plot_slice2d (bool): Create plots of the input-output behaviour of the GP for each pair of inputs.
            compute_optima must be on if set to True.
        slice_cols (int): Maximum number of columns in multi-panel slice plot figures
    """

    # TODO: Separate into Train and Plotting Settings?
    optim_samples: int = 10
    optim_method: str = SLSQP
    plot_slice1d: bool = True
    plot_slice2d: bool = False
    plot_binned_slices: Optional[BinnedSlicesConfig] = None
    compute_optima: bool = True  # Compute optima checks if plot_slice1d (and 2d) are set to True, so it goes after them
    slice_cols: Optional[int] = None
    hmc: bool = True
    iterations: int = 100
    num_folds: int = 5

    @pydantic.validator("compute_optima")
    def compute_optima_specified_if_plotting_slices(cls, value: bool, values: Dict) -> bool:  # pragma: no cover
        """If plot_slices1d or 2d is True, compute_optima must be True."""
        if "plot_slice1d" in values and values["plot_slice1d"]:
            if not value:  # pragma: no cover
                raise pydantic.ValidationError(
                    "If plot_slice1d set to True, compute_optima must also be True."
                )  # type: ignore
        if "plot_slice2d" in values and values["plot_slice2d"]:
            if not value:  # pragma: no cover
                raise pydantic.ValidationError(
                    "If plot_slice2d set to True, compute_optima must also be True."
                )  # type: ignore
        return value


class ModelSettings(pydantic.BaseModel):
    """
    A class to collect configuration options that define the model

    Attributes:
        kernel (str): The kernel to be used for the covariance function
        add_bias (bool): Whether to add a bias kernel
        anisotropic_lengthscale (bool): Whether to use anisotropic length-scales (ARD)
        priors: If specified, priors will be placed on the model parameters named in this dictionary.
            The names for the parameters to use should be those og GPy model hyperparameters (e.g.
            'kern.RBF.lengthscale'). The allowed hyperparameters will change depending on the model specification
            (choice of kernel, inclusion of bias, etc.). To get the parameter names from a GPy model one can
            run model.parameter_names()

            Example:
                priors={"Gaussian_noise.variance": PriorConfig(distribution="Gamma")}

        fixed_hyperparameters: A dictionary specifying whether to fix any of the model hyperparameters
            to specific values. These hyperparameters won't be optimized as a result.
            The dictionary points from parameter names to the value (or multiple values in case of
            vector hyperparameters) to fix them to.
            Just as for priors, the hyperparameters names are the GPy model hyperparameter names
            (obtained when one runs model.parameter_names() on the GPy model).

            Example:
                fixed_hyperparameters={"Gaussian_noise.variance": 1e-5, "kern.RBF.lengthscale": [5.0, 1.0, 1.0]}
            for a GPy model with an anisotropic-lengthscale RBF kernel with 3 input dimensions.
    """

    kernel: str = KernelFunction.RBF.name
    add_bias: bool = False  # Recommended setting to True if the output is to be logged
    anisotropic_lengthscale: bool = False
    priors: Dict[str, PriorConfig] = pydantic.Field(default_factory=dict)
    fixed_hyperparameters: Dict[str, Union[float, List[float]]] = pydantic.Field(default_factory=dict)

    def get_kernel(self) -> GPy.kern.Kern:
        """Return a GPy kernel"""
        return KernelFunction[self.kernel].value

    @pydantic.validator("kernel")
    def validate_kernel(cls, value: str) -> str:
        """Validates if provided kernel name is implemented in ABEX.

        See `pydantic validators <https://pydantic-docs.helpmanual.io/usage/validators/>`_ for more information.
        """
        return validate_enum(value, KernelFunction)


class BayesOptSettings(pydantic.BaseModel):
    """
    A class to collect configuration options associated with Bayesian Optimization

    Attributes:
        batch (int): The batch-size for a generated experiment. Non-negative. (Set to 0 for fast model evaluation).
        num_samples (int): Number of sample points in optimization of local penalization acquisition
        num_anchor (int): Number of anchor points in optimization of local penalization acquisition
        lipschitz_constant (float): Controls the strength of repulsion in local penalization
        context (dict): The context of Bayesian optimization (fixed input values)
    """

    batch: NonNegativeInt = cast(NonNegativeInt, 0)
    num_samples: pydantic.PositiveInt = cast(pydantic.PositiveInt, 1000)
    num_anchors: pydantic.PositiveInt = cast(pydantic.PositiveInt, 10)
    lipschitz_constant: Optional[pydantic.PositiveFloat] = None
    # pydantic seems to have some hard to reproduce bugs when Union[float, str] is used. It seems to work fine with
    # Union[float, int, str], though.
    context: Dict[str, FloatIntStr] = pydantic.Field(default_factory=dict)
    acquisition: str = AcquisitionClass.MEAN_PLUGIN_EXPECTED_IMPROVEMENT.name
    acquisition_optimizer: AcquisitionOptimizer = AcquisitionOptimizer.GRADIENT
    batch_strategy: BatchAcquisitionStrategy = BatchAcquisitionStrategy.LOCAL_PENALIZATION
    nonmyopic_batch: bool = False

    def get_acquisition_class(self) -> Type[Acquisition]:
        """Return the type of acquisition function"""
        return AcquisitionClass[self.acquisition].value

    @pydantic.validator("acquisition")
    def validate_acquisition(cls, value: str) -> str:  # pragma: no cover
        """Validates if provided acquisition (an arbitrary str) represents acquisition class that can be used
        in ABEX.

        See `pydantic validators <https://pydantic-docs.helpmanual.io/usage/validators/>`_ for more information.
        """
        return validate_enum(value, AcquisitionClass)

    @pydantic.validator("batch_strategy")
    def no_gradients_with_mmqei(cls, value: bool, values: Dict) -> bool:
        """If batch_strategy is MM-qEI, then acquisition_optimizer must be RANDOM."""
        if "acquisition_optimizer" in values and values["acquisition_optimizer"] is AcquisitionOptimizer.GRADIENT:
            if value is BatchAcquisitionStrategy.MMEI:  # pragma: no cover
                raise pydantic.ValidationError(
                    "If batch_strategy is MomentMatchedEI, then acquisition_optimizer cannot be Gradient."
                )  # type: ignore
        return value


class ZoomOptSettings(pydantic.BaseModel):
    """
    A class to collect configuration options associated with "zooming in" optimization (usual approach in biology).

    Attributes:
        batch (int): The batch-size for a generated experiment. Should be at least 1.
        design (DesignType): how the points are sampled from a specified area (e.g. this can be a
            uniform grid or e.g. a Latin or Sobol one). Defaults to Latin.
        n_step (Optional[int]): which iteration of optimization is this? (E.g. 1 if only an initial batch was collected,
            2 if zoom optimization was already used once, etc.). Should be at least 1, if given. Defaults to None; in
            that case the number of steps is estimated using the data set and batch size
        shrinking_factor (float): space volume is shrunk by this factor. Must be in range (0, 1). Defaults to 0.5.
        shrink_per_iter (bool): whether the sampling space volume is shrunk by the shrinking_factor per iteration
            (otherwise it will be shrunk by that factor per batch). This parameterisation of the shrinking factor
            should be more robust to batch-size.
    """

    batch: pydantic.PositiveInt = cast(pydantic.PositiveInt, 1)
    design: DesignType = DesignType.LATIN
    n_step: Optional[pydantic.PositiveInt] = None
    shrinking_factor: OpenUnitInterval = cast(OpenUnitInterval, 0.5)  # We require 0 < shrinking_factor < 1.
    shrink_per_iter: bool = False


class OptimizerConfig(pydantic.BaseModel):
    """A class to collect all optimizer configuration options

    Attributes:
        data (abex.data_settings.DataSettings): Settings associated with data loading/pre-processing
        model (ModelSettings): Settings associated with the model definition
        training (TrainSettings): Settings associated with training the model
        bayesopt (BayesOptSettings): Settings associated with Bayesian optimization and experiment generation
        zoomopt (ZoomOptSettings or None): Settings associated with experiment generation via "zoom in" optimization
        seed (int): Seed for random-number generators
        results_dir (Path): Path to store results files
        optimization_strategy: either "Zoom" or "Bayesian", specifies which algorithm should be used
        resolution_spec: string representing the way "@" variables have been resolved. Form is "name1ind1name2ind2...",
            where each "nameNindN" pair is the name of a choice variable (after the "@") and the index (from 1)
            of its value.
        init_design_strategy: Design type to use to generate the initial batch (when used for simulation)
        num_init_points: number of initial points (when used for simulation; None if not simulation)
    """

    data: DataSettings = pydantic.Field(default_factory=DataSettings)
    model: ModelSettings = pydantic.Field(default_factory=ModelSettings)
    training: TrainSettings = pydantic.Field(default_factory=TrainSettings)
    bayesopt: BayesOptSettings = pydantic.Field(default_factory=BayesOptSettings)
    zoomopt: Optional[ZoomOptSettings] = None
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN

    seed: Optional[int] = None
    results_dir: Path = pydantic.Field(default=Path(RESULTS_DIR))
    resolution_spec: str = ""

    init_design_strategy: DesignType = DesignType.RANDOM
    num_init_points: Optional[int] = None

    @classmethod
    def get_consistent_simulator_fields(cls) -> List[str]:
        """
        Returns an approximation to the "consistent" simulator fields of the class - those that should be the same
        over all runs in a simulation. Assumes all fields in cls that are not in OptimizerConfig are simulator fields,
        and that the only simulator field in OptimizerConfig to be returned is num_init_points. init_design_strategy
        is not returned, because it is allowed to change between runs in a simulation (i.e. it is a simulator field,
        but it does not have to be consistent).
        """
        non_simulator_fields = set(OptimizerConfig.__fields__.keys()).difference(["num_init_points"])
        cls_fields = set(cls.__fields__.keys())
        return sorted(cls_fields.difference(non_simulator_fields))

    def get_simulator(self) -> Any:
        """
        Returns an instance of SimulatorBase - not declared here to avoid circular import.
        Should only be called when simulation is being done.
        """
        raise NotImplementedError()  # pragma: no cover

    @property
    def experiment_batch_path(self) -> Path:
        """Get the path to the file in which the experiment batch from the optimization procedure is/will be saved."""
        assert (  # TODO: Now `results_dir` is Path, not Optional[Path]. Do we still need this?
            self.results_dir is not None
        ), "A results directory must be specified to generate a batch"
        return Path(self.results_dir) / BATCH_CSV

    def experiment_batch_path_for_fold(self, fold: int) -> Path:  # pragma: no cover
        """Get the path to the file in which the batch from the BO procedure with model for a given cross-validation
        fold is/will be saved.
        """
        assert (  # TODO: Now `results_dir` is Path, not Optional[Path]. Do we still need this?
            self.results_dir is not None
        ), "A results directory must be specified to generate a batch"
        return Path(self.results_dir) / BATCH_FOLD_CSV_BASE.format(fold)

    def with_adjustments(self, extension: str, file_dict: Dict[str, str], create: bool = True):
        """
        Returns a version of self with results_dir extended by extension and data.files replaced by file_dict.
        Args:
            extension: relative path for results_dir
            file_dict: new value of data.files
            create: create extended dir if it does not exist
        """
        extended_dir = self.results_dir / extension
        if create:
            extended_dir.mkdir(parents=True, exist_ok=True)
        data_settings = self.data.copy(update={"results_dir": extended_dir, "files": file_dict})
        return self.copy(update={"results_dir": extended_dir, "data": data_settings})


def load(
    config_path_or_name: str,
    seed: int = 0,
    config_class: Type[OptimizerConfig] = OptimizerConfig,
) -> OptimizerConfig:  # pragma: no cover
    """Load a config from a yaml file (assuming that it doesn't have multiple resolutions.
    For reference see load_resolution()).

    Args:
        config_path_or_name (str): Path to the yaml config file, or name of the config file that will be looked up.
        seed (Optional[int], optional): Random seed. If given, the 'seed' will be set to this values
            in the resulting dictionary. Defaults to None.
        config_class: The config class to instantiate with the config dictionary into a config object

    Returns:
        OptimizerConfig (there is an implicit assumption there is only one resolution of the config)
    """
    config = next(load_resolutions(config_path_or_name, seed, 1, config_class))[0][1]
    return config


def load_config_from_path_or_name(config_path_or_name: PathOrString) -> Tuple[Path, Dict]:
    """Loads a yaml file the name of or path to which is given in config_path_or_name into a dictionary.

    Args:
        config_path_or_name (PathOrString): name or path of a config
    Returns:
        path of config, and Dict representing the config
    """
    file_path = Path(config_path_or_name)
    if not file_path.exists():
        raise FileNotFoundError(  # pragma: no cover
            f"Cannot find config file {config_path_or_name} in current directory {Path.cwd()}"
        )
    loader: Callable
    if file_path.suffix in [".yml", ".yaml"]:
        loader = yaml.safe_load
    elif file_path.suffix == ".json":  # pragma: no cover
        loader = json.load
    else:  # pragma: no cover
        raise ValueError(f"Spec file name must end with .yml, .yaml or .json, not: {file_path}")
    with open(file_path) as f:
        return file_path, loader(f)


class CustomDumper(yaml.SafeDumper):
    """The default dumper of pyYAML saves Paths and Enums as Python objects. (They are not human-readable and probably
    won't parse properly). We need to implicitly convert them to other formats.

    Note:
        Whenever you see unexpected output, as '&id001' or '!!python/object/apply', extend this dumper with
        an appropriate data format.
    """

    def represent_data(self, data):
        if isinstance(data, Enum):  # We want to store Enum values (not keys), as they are parsed by pydantic.
            return self.represent_data(data.value)
        elif isinstance(data, Path):  # We need to convert Paths to strings.
            return self.represent_data(str(data))

        return super().represent_data(data)


def make_config_from_dict(config_dct: Dict, config_class: Type[OptimizerConfig]) -> OptimizerConfig:
    """Instantiate the config class from a loaded yaml structure, make sure the results directory is created,
    and dump a yaml representation of the created config there.

    Args:
        config_dct (Dict): Dictionary to construct the config object from
        config_class: The config class to instantiate with the config dictionary into a config object

    Returns:
        ConfigType: An instance of the ConfigType config
    """
    config = config_class(**config_dct)
    # Make the results directory if it doesn't exist yet
    config.results_dir.mkdir(parents=True, exist_ok=True)
    # Save config. We use the `dict()` method of pydantic `BaseModel`.
    with open(config.results_dir / "config.yml", "w") as outp:
        yaml.dump(config.dict(), outp, Dumper=CustomDumper)
    return config


def _get_seed_dirname_format(max_value: int) -> str:
    n_chars = len(str(max_value))
    return "seed{:0" + str(n_chars) + "d}"


def load_resolutions(
    config_path_or_name: str,
    seed: int = 0,
    num_runs: int = 1,
    config_class: Type[OptimizerConfig] = OptimizerConfig,
    max_resolutions: Optional[int] = None,
    resolution_spec: Optional[str] = None,
) -> Generator[List[Tuple[Dict[str, Any], OptimizerConfig]], None, None]:
    """Creates a generator of configs from a single config file (with possibly multiple resolutions).

    Handles the following steps:
        1. Loads a config as dictionary from a file specified by config_path_or_name. If config_path_or_name is not a
            path pointing to a config, assume it's a name and try to recursively find such config.
        2. (Possibly) expand the structure of the config into multiple resolutions, if multiple resolutions specified:
            For example, if the config includes a field:
                parameter: ['@p', 'a', 'b']
            This function will yield two configs with "parameter: 'a'" and "parameter: 'b'" (order not guaranteed)
        3. For each config resolution, adjust the results_dir and data.folder directories to point to
           unique sub-directories of those specified (or implied) in the config, and yield the config. The
           subdirectories will be those in the config extended by the selection string (var1val1var2val2 etc) or
           "fixed" if no variables, and then by "seedNN" where NN is the seed value.

    Args:
        config_path_or_name (str): [description]
        seed: minimum seed value to iterate from
        num_runs: number of seeds (starting from "seed") to iterate over
        config_class: The config class to instantiate with the config dictionary into a config object
        max_resolutions: if not None, only return at most this number of resolutions.
        resolution_spec: if not None, only return a config corresponding to that spec.
    Yields:
        a sequence of lists of pairs, where each pair is:
           - the dictionary from which a Config is built
           - the Config itself
    """
    # First load the config as a vanilla structure, independent of the config class.
    yaml_file_path, config_dct = load_config_from_path_or_name(config_path_or_name)
    # If results_dir is not specified in the config itself, use the default RESULTS_DIR plus file path stem.
    if "results_dir" in config_dct:
        config_dct["results_dir"] = Path(config_dct["results_dir"])  # pragma: no cover
    else:
        # Adjust base results directory to point to a sub-directory specific to this optimization experiment
        config_dct["results_dir"] = Path(RESULTS_DIR) / yaml_file_path.stem
    if "simulation_folder" in config_dct["data"]:  # pragma: no cover
        yield [({}, make_config_from_dict(config_dct, config_class=config_class))]
        return
    seed_dirname_fmt = _get_seed_dirname_format(seed + num_runs - 1)
    for sel_spec, resolved in generate_resolutions(config_dct, max_resolutions, resolution_spec):
        pairs = []
        for seed_value in range(seed, seed + num_runs):
            resolved_for_seed = copy.deepcopy(resolved)
            resolved_for_seed["seed"] = seed_value
            sel_spec_dirname = sel_spec or "fixed"
            seed_name = seed_dirname_fmt.format(seed_value)
            resolved_for_seed["results_dir"] = resolved_for_seed["results_dir"] / sel_spec_dirname / seed_name
            resolved_for_seed["data"]["simulation_folder"] = (
                resolved_for_seed["data"]["folder"] / sel_spec_dirname / seed_name
            )
            pairs.append((resolved_for_seed, make_config_from_dict(resolved_for_seed, config_class=config_class)))
        yield pairs


def generate_resolutions(
    config_dct: Dict[str, Any],
    max_resolutions: Optional[int],
    resolution_spec: Optional[str],
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """
    Given a config in the form of a dictionary, yields a sequence of resolved configs and their resolution specs.
    See load_resolutions for more details.
    """
    # Then expand (resolve) the config in every possible way (up to max_resolutions).
    configs_and_resolutions = expand_structure_with_resolutions(config_dct, max_resolutions=max_resolutions)
    for resolved, res_spec in configs_and_resolutions:
        if resolution_spec in [None, res_spec]:
            # Update results directory if there are resolutions, i.e. if res_spec is a non-empty string.
            if res_spec and "results_dir" in resolved:
                resolved["results_dir"] = Path(resolved["results_dir"]) / res_spec
            if "data" in resolved and "folder" in resolved["data"]:
                resolved["data"]["folder"] = Path(resolved["data"]["folder"])
            resolved["resolution_spec"] = res_spec
            yield res_spec, resolved


def simple_load_config(path: PathOrString, config_class: Type[OptimizerConfig] = OptimizerConfig) -> OptimizerConfig:
    """Load config from file. This is the simplest method possible, which does not postprocess it in any ways.

    Args:
        path: location of the YAML file
        config_class: The config class to instantiate with the config dictionary into a config object

    Returns:
        config object, containing only the information stored in the YAML file (and default values)
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    result = config_class(**data)
    result.data.config_file_location = Path(path)
    return result


def validate_enum(name: str, enum_class: Type[enum.Enum]) -> str:
    """Validates if provided attribute value (an arbitrary str name) represents a valid Enum name that can be used
    to retrieve an Enum value
    """
    allowed_names: List[str] = [_.name for _ in enum_class]

    if name not in allowed_names:
        raise ValueError(
            f"{name} not recognized as an option for {enum_class.__name__}."  # pragma: no cover
            f" Allowed: {allowed_names}."
        )

    return name
