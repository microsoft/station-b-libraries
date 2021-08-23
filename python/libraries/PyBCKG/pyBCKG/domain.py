# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import dacite
import pandas
from dacite import Config


@dataclass
class Concentration:
    value: float
    units: str

    def to_str(self):
        return f"{self.value}{self.units}"


class TimeUnit(Enum):
    Hours = "h"
    Minutes = "min"
    Seconds = "sec"


@dataclass
class Time:
    value: float
    units: TimeUnit

    def convert_to(self, unit: TimeUnit):
        if unit == TimeUnit.Hours:
            if self.units == TimeUnit.Hours:
                return self
            elif self.units == TimeUnit.Minutes:
                return Time(self.value / 60.0, unit)
            elif self.units == TimeUnit.Seconds:
                return Time(self.value / 3600.0, unit)
        elif unit == TimeUnit.Minutes:
            if self.units == TimeUnit.Hours:
                return Time(self.value * 60.0, unit)
            elif self.units == TimeUnit.Minutes:
                return self
            elif self.units == TimeUnit.Seconds:
                return Time(self.value / 60.0, unit)
        elif unit == TimeUnit.Seconds:
            if self.units == TimeUnit.Hours:
                return Time(self.value * 3600.0, unit)
            elif self.units == TimeUnit.Minutes:
                return Time(self.value * 60.0, unit)
            elif self.units == TimeUnit.Seconds:
                return self


@dataclass
class Position:
    row: int
    col: int


class DNAType(Enum):
    SourceLinearDNA = "Linear DNA (source)"
    SourcePlasmidDNA = "Plasmid DNA (source)"
    AssembledPlasmidDNA = "Plasmid DNA (assembled)"
    GenericPlasmidDNA = "Plasmid DNA (generic)"


class ChemicalType(Enum):
    Other = "Other"
    Antibiotic = "Antibiotic"
    SmallMolecule = "Small Molecule"
    Media = "Media"


@dataclass(frozen=True)
class Reagent:
    guid: str
    name: str
    notes: str
    barcode: Optional[str]


@dataclass(frozen=True)
class DNA(Reagent):
    dna: str
    _type: DNAType

    def to_dict(self):
        json = {
            "guid": self.guid,
            "name": self.name,
            "notes": self.notes,
            "dna": self.dna,
            "_type": self._type.value,
            "barcode": self.barcode,
        }
        return json

    @staticmethod
    def from_dict(json):
        return dacite.from_dict(data_class=DNA, data=json, config=Config(cast=[DNAType]))


@dataclass(frozen=True)
class Chemical(Reagent):
    _type: ChemicalType

    def to_dict(self):
        json = {
            "guid": self.guid,
            "name": self.name,
            "notes": self.notes,
            "_type": self._type.value,
        }
        return json

    @staticmethod
    def from_dict(json):
        return dacite.from_dict(data_class=Chemical, data=json, config=Config(cast=[ChemicalType]))


@dataclass
class Condition:
    guid: str
    reagent: Reagent
    concentration: Concentration
    time: Optional[Time]


@dataclass
class PlateReaderFilter:
    midpoint: float
    width: Optional[float]


FLUORESCENCE_KEY = Tuple[float, Optional[float], float, Optional[float]]
SIGNAL_MAP_KEY = Union[str, FLUORESCENCE_KEY, float]
SIGNAL_MAP = Dict[SIGNAL_MAP_KEY, str]


# Should add Enum for units
@dataclass
class Signal:
    guid: str

    def to_label(self, label_map: Optional[SIGNAL_MAP] = None) -> str:
        """Returns a human-readable label for the signal.
        The user can specify a dictionary `label_map` that maps signal meta-data to any string.
        For a generic Signal, the key must be the `guid` of the signal.
        Sub-classes of Signals inherit this property but maybe overriden."""
        if label_map is not None:
            if self.guid in label_map:
                return label_map[self.guid]
        return self.guid


class SignalType(Enum):
    TITRE = "Titre"
    CELL_DIAMETER = "CellDiameter"
    AGGREGATION = "Aggregation"
    CELL_COUNT = "CellCount"
    TRANSFECTION_EFFICIENCY = "TransfectionEfficiency"
    PLATE_READER_TEMPERATURE = "PlateReaderTemperature"
    PLATE_READER_LUMINESCENCE = "PlateReaderLuminescence"
    PLATE_READER_ABSORBANCE = "PlateReaderAbsorbance"
    PLATE_READER_FLUORESCENCE = "PlateReaderFluorescence"
    GENERIC = "Generic"


@dataclass
class PlateReaderFluorescence(Signal):
    emission: PlateReaderFilter
    excitation: PlateReaderFilter
    gain: Optional[float]

    def to_label(self, label_map: Optional[SIGNAL_MAP] = None) -> str:
        key = (
            self.excitation.midpoint,
            self.excitation.width,
            self.emission.midpoint,
            self.emission.width,
        )
        if label_map is not None:
            if key in label_map:
                return label_map[key]
            elif self.guid in label_map:
                return label_map[self.guid]
        return f"Fluorescence{key}"

    def get_type(self) -> SignalType:
        return SignalType.PLATE_READER_FLUORESCENCE


@dataclass
class PlateReaderAbsorbance(Signal):
    wavelength: float
    correction: Optional[float]
    gain: Optional[float]

    def to_label(self, label_map: Optional[SIGNAL_MAP] = None) -> str:
        if label_map is not None:
            if self.wavelength in label_map:
                return label_map[self.wavelength]
            elif self.guid in label_map:
                return label_map[self.guid]
        return f"Absorbance({self.wavelength})"

    def get_type(self) -> SignalType:
        return SignalType.PLATE_READER_ABSORBANCE


@dataclass
class PlateReaderLuminescence(Signal):
    def to_label(self, label_map: Optional[SIGNAL_MAP] = None) -> str:
        if label_map is not None:
            if self.guid in label_map:
                return label_map[self.guid]
        return "Lum"

    def get_type(self) -> SignalType:
        return SignalType.PLATE_READER_LUMINESCENCE


@dataclass
class PlateReaderTemperature(Signal):
    def to_label(self, label_map: Optional[SIGNAL_MAP] = None) -> str:
        if label_map is not None:
            if self.guid in label_map:
                return label_map[self.guid]
        return "Temp"

    def get_type(self) -> SignalType:
        return SignalType.PLATE_READER_TEMPERATURE


@dataclass
class Titre(Signal):
    def to_label(self, label_map: Optional[SIGNAL_MAP] = None) -> str:
        if label_map is not None:
            if self.guid in label_map:
                return label_map[self.guid]
        return "Titre"

    def get_type(self) -> SignalType:
        return SignalType.TITRE


@dataclass
class GenericSignal(Signal):
    name: str

    def to_label(self, label_map: Optional[SIGNAL_MAP] = None) -> str:
        if label_map is not None:
            if self.name in label_map:
                return label_map[self.guid]
            elif self.guid in label_map:
                return label_map[self.guid]
        return self.name

    def get_type(self) -> SignalType:
        return SignalType.GENERIC


@dataclass
class CellDiameter(Signal):
    def to_label(self, label_map: Optional[SIGNAL_MAP] = None) -> str:
        if label_map is not None:
            if self.guid in label_map:
                return label_map[self.guid]
        return "Cell diam."

    def get_type(self) -> SignalType:
        return SignalType.CELL_DIAMETER


@dataclass
class Aggregation(Signal):
    def to_label(self, label_map: Optional[SIGNAL_MAP] = None) -> str:
        if label_map is not None:
            if self.guid in label_map:
                return label_map[self.guid]
        return "Agg."

    def get_type(self) -> SignalType:
        return SignalType.AGGREGATION


class CellCountType(Enum):
    Live = "Live"
    Dead = "Dead"
    Total = "Total"


@dataclass
class CellCount(Signal):
    cellCountType: CellCountType

    def to_label(self, label_map: Optional[SIGNAL_MAP] = None) -> str:
        key = f"{self.cellCountType.value} cells"
        if label_map is not None:
            if key in label_map:
                return label_map[key]
            elif self.guid in label_map:
                return label_map[self.guid]
        return key

    def get_type(self) -> SignalType:
        return SignalType.CELL_COUNT


@dataclass
class TransfectionEfficiency(Signal):
    gene: str

    def to_label(self, label_map: Optional[SIGNAL_MAP] = None) -> str:
        key = f"Trfxn. Eff. ({self.gene})"
        if label_map is not None:
            if key in label_map:
                return label_map[key]
            elif self.guid in label_map:
                return label_map[self.guid]
        return key

    def get_type(self) -> SignalType:
        return SignalType.TRANSFECTION_EFFICIENCY


@dataclass
class TimeSeries:
    data: pandas.Series
    signal: Signal


class MeasureType(Enum):
    MEAN = "Mean"
    MEDIAN = "Median"
    MODE = "Mode"
    STANDARD_DEVIATION = "SD"
    VARIANCE = "Variance"


@dataclass
class Observation:
    guid: str
    value: float
    replicate: str
    observed_at: datetime
    signal: Signal
    units: Optional[str]
    measure: Optional[MeasureType]
    measuredBy: Optional[str]


class Compartment(Enum):
    Plasmid = "Plasmid"
    Chromosome = "Chromosome"
    Cytosol = "Cytosol"


@dataclass
class CellEntity:
    compartment: Compartment
    entity: Reagent


@dataclass
class Cell:
    guid: str
    name: str
    notes: Optional[str]
    entities: List[CellEntity]


@dataclass
class SampleDevice:
    cell: Cell
    cell_density: Optional[float]
    preseeding_density: Optional[float] = 2.00e5


@dataclass
class Sample:
    guid: str
    physical_plate_name: Optional[str]
    physical_well: Optional[Position]
    virtual_well: Optional[Position]
    device: SampleDevice
    conditions: List[Condition]
    observations: List[Observation]


class ExperimentOperationType(Enum):
    AnthaExecuted = "AnthaExecuted"
    AnthaBundleUploaded = "AnthaBundleUploaded"
    AnthaLayoutUploaded = "AnthaLayoutUploaded"
    BacterialStocksInnoculated = "BacterialStocksInnoculated"
    OvernightStocksDiluted = "OvernightStocksDiluted"
    PlateReaderStarted = "PlateReaderStarted"
    InputPlatePrepared = "InputPlatePrepared"
    PlateIncubated = "PlateIncubated"
    ColoniesPicked = "ColoniesPicked"
    ExperimentStarted = "ExperimentStarted"
    ExperimentFinished = "ExperimentFinished"
    ResultsProcessed = "ResultsProcessed"
    Induction = "Induction"


@dataclass
class ExperimentOperation:
    guid: str
    _type: ExperimentOperationType
    timestamp: datetime

    def to_dict(self):
        json = {
            "guid": self.guid,
            "_type": self._type.value,
            "timestamp": self.timestamp,
        }
        return json


@dataclass
class Experiment:
    guid: str
    name: str
    notes: Optional[str]
    operations: List[ExperimentOperation]
    signals: List[Signal]
    samples: List[Sample]


@dataclass
class BuildExperiment(Experiment):
    pass


@dataclass
class TestExperiment(Experiment):
    pass
