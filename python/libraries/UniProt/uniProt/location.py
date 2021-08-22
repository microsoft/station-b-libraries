# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Set, Optional
from uniProt.rdfgraph import LocationRDF
from uniProt.api import get_location_graph


@dataclass
class Location:
    """The `Location` class hold relevant metadata associated with sub-cellular locations from UniProt."""

    uniprot_id: str
    label: Optional[str]
    comment: Optional[str]
    part_of: Set[str]
    subclass: Optional[str]

    @staticmethod
    def from_location_rdf(g: LocationRDF):
        """Helper function to return a `Location` object from a `LocationRDF` object (RDF Graph)."""
        return Location(uniprot_id=g.id, label=g.label, comment=g.comment, part_of=g.part_of, subclass=g.subclass)


def get_location(uniprot_id: str) -> Location:  # pragma: no cover
    """Queries the UniProt database for a subcellular location with the id `uniprot_id`
    and returns a `Location` object"""
    g: LocationRDF = get_location_graph(uniprot_id)
    return Location.from_location_rdf(g)
