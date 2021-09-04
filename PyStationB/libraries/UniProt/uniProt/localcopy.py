# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from uniProt.keyword import Keyword, KeywordDatabase
import os
import pathlib
from io import StringIO
from dataclasses import dataclass
from typing import TypeVar
from uniProt.rdfgraph import UniProtRDF, LocationRDF, OntoBeeRDF, KeywordRDF
from uniProt.protein import Protein, get_id_db_from_uri
from uniProt.location import Location
from uniProt.api import QueryType, get_rdf_graph, RDF_QUERY, ONTOBEE_QUERY


KEYWORD_RDF_GRAPHS = TypeVar("KEYWORD_RDF_GRAPHS", OntoBeeRDF, KeywordRDF)


def save_local_copy(fp: pathlib.Path, res_io: StringIO):  # pragma: no cover
    """Helper function to save the contents of a file to a folder location"""
    with fp.open("w") as fd:
        fd.write(res_io.getvalue() + "\n")


@dataclass
class LocalCopy:
    """A class to help search/download a local folder to see if a UniProt entry exists before querying UniProt."""

    def __init__(self, filepath: pathlib.Path, save_copy: bool):
        (
            """Constructor for LocalCopy. `filepath` is the path to a local folder.
        If `filepath` is not a valid folder, an error is returned.
        If `save_copy` is set to `True`, a local copy of the UniProt entry is saved as an RDF XML file."""
        )
        if not filepath.is_dir():  # pragma: no cover
            raise ValueError(f"{filepath} is not a valid directory.")
        self.filepath = filepath
        self.save_copy = save_copy
        self.location_dir = self.filepath / "location"
        self.location_dir.mkdir(exist_ok=True, parents=True)
        self.uniprot_dir = self.filepath / "uniprot"
        self.uniprot_dir.mkdir(exist_ok=True, parents=True)
        self.keyword_dir = self.filepath / "keyword"
        self.keyword_dir.mkdir(exist_ok=True, parents=True)
        self.keyword_ontobee_dir = self.keyword_dir / "ontobee"
        self.keyword_ontobee_dir.mkdir(exist_ok=True, parents=True)
        self.keyword_uniprot_dir = self.keyword_dir / "uniprot"
        self.keyword_uniprot_dir.mkdir(exist_ok=True, parents=True)

    def process_keyword_query(self, local_fp: str, query: str, id: str, g: KEYWORD_RDF_GRAPHS) -> KEYWORD_RDF_GRAPHS:
        (
            """Checks if the keyword with `id` exists in the localcopy. If the file exists, the file is read and parsed.
        If not the relevant database is queried for the keyword. A copy of the entry is stored in `RDF` format if
        `save_copy` is set to `True`."""
        )
        if os.path.isfile(local_fp):
            g.parse(local_fp, format="xml")
        else:  # pragma: no cover
            res_io = get_rdf_graph(query, id)
            if self.save_copy:
                save_local_copy(pathlib.Path(local_fp), res_io)
            g.parse(res_io, format="xml")
        return g

    def get_keyword(self, url) -> Keyword:
        """Get a `Keyword` from a URL by first checking to see if the keyword already exists in localcopy."""
        (id, database) = get_id_db_from_uri(url)
        if database == KeywordDatabase.ONTOBEE:
            local_fp = os.path.join(self.keyword_ontobee_dir, f"{id}.xml")
            query = ONTOBEE_QUERY.format(id)
            o_rdf: OntoBeeRDF = self.process_keyword_query(local_fp, query, id, OntoBeeRDF(id))
            return Keyword.from_onto_rdf(o_rdf)
        elif database == KeywordDatabase.UNIPROT:
            local_fp = os.path.join(self.keyword_uniprot_dir, f"{id}.xml")
            query = RDF_QUERY.format(QueryType.KEYWORD.value, id)
            k_rdf: KeywordRDF = self.process_keyword_query(local_fp, query, id, KeywordRDF(id))
            return Keyword.from_keyword_rdf(k_rdf)
        raise ValueError(f"{database} not recognized for keywords.")  # pragma: no cover

    def get_location(self, uniprot_id: str) -> Location:
        (
            """Get a `Location` with id `uniprot_id`
        by first checking to see if the location already exists in localcopy."""
        )
        g = LocationRDF(uniprot_id)
        local_fp = os.path.join(self.location_dir, f"{uniprot_id}.xml")
        if os.path.isfile(local_fp):
            g.parse(local_fp, format="xml")
        else:  # pragma: no cover
            query = RDF_QUERY.format(QueryType.LOCATION.value, uniprot_id)
            res_io = get_rdf_graph(query, uniprot_id)
            if self.save_copy:
                save_local_copy(pathlib.Path(local_fp), res_io)
            g.parse(res_io, format="xml")
        return Location.from_location_rdf(g)

    def get_protein_from_rdf(self, g: UniProtRDF):
        """Helper function that checks to see if location and keyword already exist in localcopy."""
        recommended_name = g.recommended_name
        all_names = g.all_names
        gene = g.gene
        description = g.description
        isoforms = g.isoforms
        locations = [self.get_location(l_id) for l_id in g.subcellular_locations]
        keywords = [self.get_keyword(url) for url in g.keywords]
        return Protein(
            uniprot_id=g.id,
            recommended_name=recommended_name,
            all_names=all_names,
            gene=gene,
            description=description,
            isoforms=isoforms,
            locations=locations,
            keywords=keywords,
        )

    def get_protein(self, uniprot_id: str) -> Protein:
        """Get a `Protein` with id `uniprot_id` by first checking to see if the protein already exists in localcopy."""
        g = UniProtRDF(uniprot_id)
        local_fp = os.path.join(self.uniprot_dir, f"{uniprot_id}.xml")
        if os.path.isfile(local_fp):
            g.parse(local_fp, format="xml")
        else:  # pragma: no cover
            query = RDF_QUERY.format(QueryType.PROTEIN.value, uniprot_id)
            res_io = get_rdf_graph(query, uniprot_id)
            if self.save_copy:
                save_local_copy(pathlib.Path(local_fp), res_io)
            g.parse(res_io, format="xml")
        return self.get_protein_from_rdf(g)
