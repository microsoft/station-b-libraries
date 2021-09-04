# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from dataclasses import dataclass
from typing import List, Optional, Tuple

from parse import Match, parse
from uniProt.api import get_protein_graph
from uniProt.keyword import Keyword, KeywordDatabase, get_keyword
from uniProt.location import Location, get_location
from uniProt.rdfgraph import UniProtRDF

OBO_KEYWORD_FORMAT = "http://purl.obolibrary.org/obo/{}"
UNIPROT_KEYWORD_FORMAT = "http://purl.uniprot.org/keywords/{}"


def get_id_db_from_uri(url: str) -> Tuple[str, KeywordDatabase]:
    """Helper function to parse and return the Id and Database of a keyword from a URL"""
    obo_parse = parse(OBO_KEYWORD_FORMAT, url)
    if obo_parse is not None and not isinstance(obo_parse, Match):
        if len(obo_parse.spans) != 1:  # pragma: no cover
            raise ValueError(f"Format error: {url}")
        return (obo_parse[0], KeywordDatabase.ONTOBEE)
    else:
        uniprot_parse = parse(UNIPROT_KEYWORD_FORMAT, url)
        if uniprot_parse is not None and not isinstance(uniprot_parse, Match):
            if len(uniprot_parse.spans) != 1:  # pragma: no cover
                raise ValueError(f"Format error: {url}")
            return (uniprot_parse[0], KeywordDatabase.UNIPROT)
    raise ValueError(f"{url} not a recognized format")  # pragma: no cover


def get_keyword_from_uri(url: str) -> Keyword:  # pragma: no cover
    """Helper function to get a `Keyword` object from a URL"""
    (id, db) = get_id_db_from_uri(url)
    return get_keyword(id, db)


@dataclass
class Protein:
    """A class to programmatically access relevant Protein metadata from UniProt."""

    uniprot_id: str
    recommended_name: Optional[str]
    all_names: List[str]
    gene: Optional[str]
    description: Optional[str]
    isoforms: List[str]
    locations: List[Location]
    keywords: List[Keyword]

    @staticmethod
    def from_uniprot_rdf(g: UniProtRDF):  # pragma: no cover
        """Helper function to get a `Protein` object from a `UniProtRDF` object (RDF Graph)"""
        locations = [get_location(l_id) for l_id in g.subcellular_locations]
        keywords = [get_keyword_from_uri(url) for url in g.keywords]
        return Protein(
            uniprot_id=g.id,
            recommended_name=g.recommended_name,
            all_names=g.all_names,
            gene=g.gene,
            description=g.description,
            isoforms=g.isoforms,
            locations=locations,
            keywords=keywords,
        )


def get_protein(uniprot_id: str) -> Protein:  # pragma: no cover
    """Queries the UniProt database for a protein with the id `uniprot_id`
    and returns a `Protein` object"""
    g: UniProtRDF = get_protein_graph(uniprot_id)
    return Protein.from_uniprot_rdf(g)
