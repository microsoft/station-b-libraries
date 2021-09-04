# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Submodule responsible for parsing YAML configuration files into well-defined pydantic objects.

Exports:
    DataSettings, a class used to identify the data forming an experimental track
    Tag, a class representing tags we can add to experiments and samples
    load_settings, method to read (and parse) YAML configuration files
    dump_settings, method to save a pydantic object into a YAML configuration file
"""
import enum
import pathlib
from typing import Any, Callable, List, Optional, Set, Union

import pydantic
import yaml


class TagLevel(enum.Enum):
    """Whether a tag is associated to an experiment or to a specific sample."""

    EXPERIMENT = "Experiment"
    SAMPLE = "Sample"


class Tag(pydantic.BaseModel):
    """
    Attributes:
        name: tag name, e.g. "Batch"
        value: tag value, e.g. "30"
        level: whether the tag is associated on experiment or sample level

    Example:
        We want to associate with an experiment the information that we used a batch of 30. The BCKG format for this
        is "Batch:30" and we can parse it using the ``decode`` method.
        If we however want to inform BCKG that it should create such tag, we need to add this information as a column
        "Experiment:Batch" (see ``column_name`` method).
    """

    name: str
    value: Union[int, float, None, str] = None
    level: Optional[TagLevel] = None

    def encode(self) -> str:  # pragma: no cover
        """As in BCKG tags are strings (not tuples name/value), we want to save this as a string."""
        assert self.value is not None

        return f"{self.name}:{self.value}"

    @classmethod
    def decode(cls, s: str, level: Optional[TagLevel] = None) -> "Tag":  # pragma: no cover
        """This decodes a BCKG string into a Tag."""
        name, value = s.split(":")
        return cls(name=name, value=value, level=level)

    @property
    def column_name(self) -> str:  # pragma: no cover
        """We append tags in additional columns. This is the name format of such column."""
        assert self.level is not None
        return f"{self.level.value}:{self.name}"


class DataSettings(pydantic.BaseModel):
    """This class is used to parse the data configuration files.

    Attributes:
        initial_experiment_id: the experiment ID (as in BCKG) of the experiment to be pulled in at first iteration
        tags: experiment tags to be used to pull in the data at second, third, ... iterations (tags identify the
            experimental track)
    """

    initial_experiment_id: str
    tags: List[Tag]

    @property
    def set_of_tags(self) -> Set[str]:  # pragma: no cover
        return set(tag.encode() for tag in self.tags)


def load_settings(yaml_path: Union[pathlib.Path, str], class_factory: Callable) -> Any:  # pragma: no cover
    """A generic method to load settings.

    Args:
        yaml_path: location of the YAML configuration file
        class_factory: a function generating pydantic objects storing configuration

    Returns:
        an object of type returned by class factory
    """
    with open(yaml_path) as file_handler:
        data_dict: dict = yaml.safe_load(file_handler)
        return class_factory(**data_dict)


def dump_settings(yaml_path: Union[pathlib.Path, str], config: pydantic.BaseModel) -> None:  # pragma: no cover
    """Dumps pydantic model ``config`` to location ``yaml_path``."""
    with open(yaml_path, "w") as output:
        yaml.dump(config.dict(), output, Dumper=CustomDumper, default_flow_style=False)


class CustomDumper(yaml.SafeDumper):
    """The default dumper of pyYAML saves Paths and Enums as Python objects. (They are not human-readable and probably
    won't parse properly). We need to implicitly convert them to other formats.

    Note:
        Whenever you see unexpected output, as '&id001' or '!!python/object/apply', extend this dumper with
        an appropriate data format.
    """

    def represent_data(self, data):  # pragma: no cover
        if isinstance(data, enum.Enum):  # We want to store Enum values (not keys), as they are parsed by pydantic.
            return self.represent_data(data.value)
        elif isinstance(data, pathlib.Path):  # We need to convert Paths to strings.
            return self.represent_data(str(data))

        return super().represent_data(data)
