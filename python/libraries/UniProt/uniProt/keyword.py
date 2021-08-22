# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from enum import Enum
from typing import Optional
from dataclasses import dataclass
from uniProt.rdfgraph import KeywordRDF, OntoBeeRDF
from uniProt.api import get_keyword_graph, get_ontobee_graph


TECHNICAL_TERM_UNIPROT = "http://purl.uniprot.org/keywords/9990"
PTM_UNIPROT = "http://purl.uniprot.org/keywords/9991"
MOLECULAR_FUNCTION_UNIPROT = "http://purl.uniprot.org/keywords/9992"
LIGAND_UNIPROT = "http://purl.uniprot.org/keywords/9993"
DOMAIN_UNIPROT = "http://purl.uniprot.org/keywords/9994"
DISEASE_UNIPROT = "http://purl.uniprot.org/keywords/9995"
DEVELOPMENTAL_STAGE_UNIPROT = "http://purl.uniprot.org/keywords/9996"
CODING_SEQUENCE_DIVERSITY_UNIPROT = "http://purl.uniprot.org/keywords/9997"
CELLULAR_COMPONENT_UNIPROT = "http://purl.uniprot.org/keywords/9998"
BIOLOGICAL_PROCESS_UNIPROT = "http://purl.uniprot.org/keywords/9999"
BIOLOGICAL_PROCESS_OBO = "biological_process"
MOLECULAR_FUNCTION_OBO = "molecular_function"
CELLULAR_COMPONENT_OBO = "cellular_component"


class KeywordType(Enum):
    """Enum specifying the category of a Keyword."""

    BIOLOGICAL_PROCESS = "Biological Process"
    CELLULAR_COMPONENT = "Cellular Component"
    CODING_SEQUENCE_DIVERSITY = "Coding Sequence Diversity"
    DEVELOPMENTAL_STAGE = "Developmental Stage"
    DISEASE = "Disease"
    DOMAIN = "Domain"
    LIGAND = "Ligand"
    MOLECULAR_FUNCTION = "Molecular Function"
    POST_TRANSLATIONAL_MODIFICATION = "Post Translational Modification"
    TECHNICAL_TERM = "Technical Term"

    @staticmethod
    def from_uniprot_uri(uri: str):
        """Helper function that returns a `KeywordType` (Keyword Category) from the UniProt URL."""
        if uri == DOMAIN_UNIPROT:
            return KeywordType.DOMAIN
        elif uri == BIOLOGICAL_PROCESS_UNIPROT:
            return KeywordType.BIOLOGICAL_PROCESS
        elif uri == CELLULAR_COMPONENT_UNIPROT:
            return KeywordType.CELLULAR_COMPONENT
        elif uri == CODING_SEQUENCE_DIVERSITY_UNIPROT:
            return KeywordType.CODING_SEQUENCE_DIVERSITY
        elif uri == DEVELOPMENTAL_STAGE_UNIPROT:
            return KeywordType.DEVELOPMENTAL_STAGE
        elif uri == DISEASE_UNIPROT:
            return KeywordType.DISEASE
        elif uri == LIGAND_UNIPROT:
            return KeywordType.LIGAND
        elif uri == MOLECULAR_FUNCTION_UNIPROT:
            return KeywordType.MOLECULAR_FUNCTION
        elif uri == PTM_UNIPROT:
            return KeywordType.POST_TRANSLATIONAL_MODIFICATION
        elif uri == TECHNICAL_TERM_UNIPROT:
            return KeywordType.TECHNICAL_TERM
        else:
            raise ValueError(f"{uri} not recognized as Keyword type")  # pragma: no cover

    @staticmethod
    def from_obo_ns(obo_namespace: str):
        """Helper function that returns a `KeywordType` (Keyword Category) from the Ontobee URL."""
        if obo_namespace == BIOLOGICAL_PROCESS_OBO:
            return KeywordType.BIOLOGICAL_PROCESS
        elif obo_namespace == MOLECULAR_FUNCTION_OBO:
            return KeywordType.MOLECULAR_FUNCTION
        elif obo_namespace == CELLULAR_COMPONENT_OBO:
            return KeywordType.CELLULAR_COMPONENT
        else:
            raise ValueError(f"{obo_namespace} not a OBO NS recognized keyword type.")  # pragma: no cover


class KeywordDatabase(Enum):
    """Enum specifying the database/repository of a Keyword."""

    UNIPROT = "UniProt"
    ONTOBEE = "Ontobee"


@dataclass
class Keyword:
    """A Keyword is a way of categorizing proteins based on various properties."""

    id: str
    label: Optional[str]
    database: KeywordDatabase
    keyword_type: Optional[KeywordType]
    description: Optional[str]

    @staticmethod
    def from_keyword_rdf(g: KeywordRDF):
        """Helper function to get a Keyword from a `KeywordRDF`object (RDFGraph)."""
        _type = None if g.category is None else KeywordType.from_uniprot_uri(g.category)
        return Keyword(
            id=g.id, label=g.label, database=KeywordDatabase.UNIPROT, keyword_type=_type, description=g.comment
        )

    @staticmethod
    def from_onto_rdf(g: OntoBeeRDF):
        """Helper function to get a Keyword from a `OntoBeeRDF`object (RDFGraph)."""
        _type = None if g.obo_namespace is None else KeywordType.from_obo_ns(g.obo_namespace)
        return Keyword(
            id=g.id, label=g.label, database=KeywordDatabase.ONTOBEE, keyword_type=_type, description=g.definition
        )


def get_keyword(id: str, database: KeywordDatabase) -> Keyword:  # pragma: no cover
    """Query the relevant `database` (Ontobee or UniProt) for a keyword by its `id`"""
    if database == KeywordDatabase.UNIPROT:
        g = get_keyword_graph(id)
        return Keyword.from_keyword_rdf(g)
    elif database == KeywordDatabase.ONTOBEE:
        g = get_ontobee_graph(id)
        return Keyword.from_onto_rdf(g)
    else:
        raise ValueError(f"Database {database.value} not recognized.")
