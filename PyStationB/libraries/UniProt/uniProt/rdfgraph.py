# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import List, Set, Optional
from rdflib import URIRef, Graph
from rdflib.namespace import SKOS, RDF, RDFS
from parse import parse, Match


FUNCTION_ANNOTATION_REF = URIRef("http://purl.uniprot.org/core/Function_Annotation")
SEQUENCE_REF = URIRef("http://purl.uniprot.org/core/sequence")
FULL_NAME_REF = URIRef("http://purl.uniprot.org/core/fullName")
CATEGORY_REF = URIRef("http://purl.uniprot.org/core/category")
RECOMMENDED_NAME_REF = URIRef("http://purl.uniprot.org/core/recommendedName")
CELLULARCOMPONENT = URIRef("http://purl.uniprot.org/core/cellularComponent")
CELLULAR_COMPONENT = URIRef("http://purl.uniprot.org/core/Cellular_Component")
PART_OF = URIRef("http://purl.uniprot.org/core/partOf")
CLASSIFIED_WITH_REF = URIRef("http://purl.uniprot.org/core/classifiedWith")
OBO_DEFINITION = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")
OBO_NAMESPACE_REF = URIRef("http://www.geneontology.org/formats/oboInOwl#hasOBONamespace")
LOCATION_FORMAT = "http://purl.uniprot.org/locations/{}"
UNIPROT_FORMAT = "http://purl.uniprot.org/uniprot/{}"
KEYWORD_FORMAT = "http://purl.uniprot.org/keywords/{}"
ONTOBEE_FORMAT = "http://purl.obolibrary.org/obo/{}"


def get_location_id(url: str) -> str:
    """Parses the ID of the Location from the UniProt URL"""
    loc_id = parse(LOCATION_FORMAT, url)
    if loc_id is not None and not isinstance(loc_id, Match):
        if len(loc_id.spans) != 1:  # pragma: no cover
            raise ValueError(f"Format error: {url}")
        return loc_id[0]
    raise ValueError(f"{url} not a recognized format")  # pragma: no cover


class RDFGraph(Graph):
    """Subclass of the `rdflib.Graph` class.
    This class also contains an `id` property which can either be the UniProt or GO ID."""

    def __init__(self, id: str):
        super().__init__()
        self.id = id


class UniProtRDF(RDFGraph):
    """Subclass of the `RDFGraph` class.
    This class contains helper functions to extract protein properties from RDF graphs."""

    @property
    def recommended_name(self) -> Optional[str]:
        """Returns the preferred/recommended name"""
        uniprot_uri = URIRef(UNIPROT_FORMAT.format(self.id))
        subject_list = [o for s, p, o in self if s == uniprot_uri and p == RECOMMENDED_NAME_REF]
        for s, p, o in self:
            if s in subject_list and p == FULL_NAME_REF:
                return o.value
        return None  # pragma: no cover

    @property
    def all_names(self) -> List[str]:
        """Returns all the listed names"""
        return [o.value for s, p, o in self if p == FULL_NAME_REF]

    @property
    def gene(self) -> Optional[str]:
        """Returns the gene name"""
        for s, p, o in self:
            if p == SKOS.prefLabel:
                return o.value
        return None  # pragma: no cover

    @property
    def isoforms(self) -> List[str]:
        """Returns the list of protein isoforms"""
        uniprot_uri = URIRef(UNIPROT_FORMAT.format(self.id))
        seq_subjects = [o for s, p, o in self if p == SEQUENCE_REF and s == uniprot_uri]
        isoforms = [o.value for s, p, o in self if s in seq_subjects and p == RDF.value]
        return isoforms

    @property
    def description(self) -> Optional[str]:
        """Returns the description of the protein from the RDF Graph `g`"""
        subject_list = [s for s, p, o in self if o == FUNCTION_ANNOTATION_REF]
        for s, p, o in self:
            if s in subject_list and p == RDFS.comment:
                return o.value
        return None  # pragma: no cover

    @property
    def subcellular_locations(self) -> List[str]:
        """Returns the list of location IDs"""
        locations = [get_location_id(o.toPython()) for s, p, o in self if p == CELLULARCOMPONENT]
        return locations

    @property
    def keywords(self) -> List[str]:
        """Returns the uris for the keywords"""
        keywords = [o.toPython() for s, p, o in self if p == CLASSIFIED_WITH_REF]
        return keywords


class LocationRDF(RDFGraph):
    """Subclass of the `RDFGraph` class.
    This class contains helper functions to
    extract location properties from RDF graphs."""

    @property
    def label(self) -> Optional[str]:
        """Returns the preferred label of the Location"""
        for s, p, o in self:
            if p == SKOS.prefLabel:
                return o.value
        return None  # pragma: no cover

    @property
    def subclass(self) -> Optional[str]:
        """Returns the UniProt Location Id of the Super Class"""
        for s, p, o in self:
            if p == RDFS.subClassOf and o != CELLULAR_COMPONENT:
                return get_location_id(o.toPython())
        return None

    @property
    def part_of(self) -> Set[str]:
        """Returns a list of UniProt Location Id that this location is a part of."""
        part_of_list: Set[str] = set()
        for s, p, o in self:
            if p == PART_OF:
                part_of_list.add(get_location_id(o.toPython()))
        return part_of_list

    @property
    def comment(self) -> Optional[str]:
        """Returns the description of this location"""
        for s, p, o in self:
            if p == RDFS.comment:
                return o.value
        return None  # pragma: no cover


class KeywordRDF(RDFGraph):
    """Subclass of the `RDFGraph` class.
    This class contains helper functions to extract
    keyword properties from RDF graphs.."""

    @property
    def label(self) -> Optional[str]:
        """Returns the preferred label of the Keyword"""
        for s, p, o in self:
            if p == SKOS.prefLabel:
                return o.value
        return None  # pragma: no cover

    @property
    def comment(self) -> Optional[str]:
        """Returns the description of the Keyword"""
        for s, p, o in self:
            if p == RDFS.comment:
                return o.value
        return None  # pragma: no cover

    @property
    def category(self) -> Optional[str]:
        """Returns the URI of the Keyword category"""
        for s, p, o in self:
            if p == CATEGORY_REF:
                return o.toPython()
        return None  # pragma: no cover


class OntoBeeRDF(RDFGraph):
    """Subclass of the `RDFGraph` class.
    This class contains helper functions to
    extract GO keyword properties from RDF graphs."""

    @property
    def label(self) -> Optional[str]:
        """Returns the preferred label of the Keyword"""
        for s, p, o in self:
            if s == URIRef(ONTOBEE_FORMAT.format(self.id)) and p == RDFS.label:
                return o.value
        return None  # pragma: no cover

    @property
    def definition(self) -> Optional[str]:
        """Returns the definition of the keyword"""
        for s, p, o in self:
            if p == OBO_DEFINITION:
                return o.value
        return None  # pragma: no cover

    @property
    def obo_namespace(self) -> Optional[str]:
        """Returns one of 3 namespaces (if applicable):
        `cellular_component`,`molecular_function `, or `biological_process`"""
        for s, p, o in self:
            if p == OBO_NAMESPACE_REF:
                return o.value
        return None  # pragma: no cover
