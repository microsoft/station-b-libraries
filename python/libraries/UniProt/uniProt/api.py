# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from typing import List

import requests
from uniProt.mapping import LocationIDs, TaxonomyIDs
from uniProt.rdfgraph import KeywordRDF, LocationRDF, OntoBeeRDF, UniProtRDF


class QueryType(Enum):
    """Helper Enum to specify the type of GET request sent to UniProt REST API."""

    PROTEIN = "uniprot"
    LOCATION = "locations"
    KEYWORD = "keywords"


UNIPROT_QUERY = "https://www.uniprot.org/uniprot/?query="
RDF_QUERY = "https://www.uniprot.org/{}/{}.rdf"
ONTOBEE_QUERY = "http://www.ontobee.org/ontology/GO?iri=http://purl.obolibrary.org/obo/{}"


@dataclass
class QueryParameters:
    """Class to specify query parameters for UniProt. If `reviewed`
    is set to `True`, only UniProt entries that have been reviewed by the curators of
    UniProt will be returned."""

    reviewed: bool
    taxonomies: List[TaxonomyIDs]
    locations: List[LocationIDs]

    @property
    def query_string(self) -> str:  # pragma: no cover
        """Converts the QueryParameters into a URL string.
        This URL is based on the GET request format of the UniProt REST API."""
        query_string = UNIPROT_QUERY + "reviewed:" + ("yes" if self.reviewed else "no")
        if len(self.taxonomies) > 0:
            t_query = [f"organism:{t.value}" for t in self.taxonomies]
            t_string = "+OR+".join(t_query)
            query_string += f"+AND+({t_string})"
        if len(self.locations) > 0:
            t_query = [f"location:{t.value}" for t in self.locations]
            t_string = "+OR+".join(t_query)
            query_string += f"+AND+locations:({t_string})"
        query_string += "&format=list"
        return query_string


def get_rdf_graph(query: str, id: str) -> StringIO:  # pragma: no cover
    """Helper function that returns the result of a GET request specified in the `query` url."""
    res = requests.get(query)
    if res.status_code != 200:
        raise LookupError(f"Entry with id {id} not found.")
    res_io = StringIO(res.text)
    return res_io


def get_protein_graph(uniprot_id: str) -> UniProtRDF:  # pragma: no cover
    """Queries the UniProt database for a protein with the id `uniprot_id`
    and returns a `UniProtRDF` Graph"""
    query = RDF_QUERY.format(QueryType.PROTEIN.value, uniprot_id)
    res_io = get_rdf_graph(query, uniprot_id)
    g = UniProtRDF(uniprot_id)
    return g.parse(res_io, format="xml")


def get_location_graph(uniprot_id: str) -> LocationRDF:  # pragma: no cover
    """Queries the UniProt database for a location with the id `uniprot_id`
    and returns a `LocationRDF` Graph"""
    query = RDF_QUERY.format(QueryType.LOCATION.value, uniprot_id)
    res_io = get_rdf_graph(query, uniprot_id)
    g = LocationRDF(uniprot_id)
    return g.parse(res_io, format="xml")


def get_keyword_graph(uniprot_id: str) -> KeywordRDF:  # pragma: no cover
    """Queries the UniProt database for a keyword with the id `uniprot_id`
    and returns a `KeywordRDF` Graph"""
    query = RDF_QUERY.format(QueryType.KEYWORD.value, uniprot_id)
    res_io = get_rdf_graph(query, uniprot_id)
    g = KeywordRDF(uniprot_id)
    return g.parse(res_io, format="xml")


def get_ontobee_graph(id: str) -> OntoBeeRDF:  # pragma: no cover
    """Queries the Gene Ontology database for a keyword with the `id`
    and returns a `OntoBeeRDF` Graph"""
    query = ONTOBEE_QUERY.format(id)
    res_io = get_rdf_graph(query, id)
    g = OntoBeeRDF(id)
    return g.parse(res_io, format="xml")


def query_proteins(params: QueryParameters) -> List[str]:  # pragma: no cover
    """Queries the UniProt database for `QueryParameters` specified in `params`
    and returns a List of UniProt IDs that match the query."""
    res = requests.get(params.query_string)
    if res.status_code != 200:
        raise LookupError(f"Malformed query with parameters: {params}")
    return res.text.split("\n")
