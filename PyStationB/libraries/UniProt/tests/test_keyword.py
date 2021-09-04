# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import pytest
from pathlib import Path
from uniProt.keyword import Keyword, KeywordDatabase, KeywordType
from uniProt.localcopy import LocalCopy


ROOT_DIR = Path(__file__).parent.parent
TEST_FILES_DIR = ROOT_DIR / "tests" / "test_files"
lc = LocalCopy(TEST_FILES_DIR, save_copy=True)


@pytest.mark.timeout(10)
def test_apoptosis():
    apoptosis_url = "http://purl.uniprot.org/keywords/53"
    apoptosis_description = (
        "Protein involved in apoptotic programmed cell death. Apoptosis is characterized by "
        + "cell morphological changes, including blebbing, cell shrinkage, nuclear fragmentation, chromatin "
        + "condensation and chromosomal DNA fragmentation, and eventually death. Unlike necrosis, apoptosis "
        + "produces cell fragments, called apoptotic bodies, that phagocytic cells are able to engulf and "
        + "quickly remove before the contents of the cell can spill out onto surrounding cells and cause "
        + "damage. In general, apoptosis confers advantages during an organism's life cycle."
    )
    apoptosis: Keyword = lc.get_keyword(apoptosis_url)
    assert apoptosis.id == "53"
    assert apoptosis.database == KeywordDatabase.UNIPROT
    assert apoptosis.description == apoptosis_description
    assert apoptosis.label == "Apoptosis"
    assert apoptosis.keyword_type == KeywordType.BIOLOGICAL_PROCESS


@pytest.mark.timeout(10)
def test_membrane():
    membrane_url = "http://purl.obolibrary.org/obo/GO_0016020"
    membrane_description = (
        "A lipid bilayer along with all the proteins and protein complexes embedded in it an attached to it."
    )
    membrane: Keyword = lc.get_keyword(membrane_url)
    assert membrane.id == "GO_0016020"
    assert membrane.database == KeywordDatabase.ONTOBEE
    assert membrane.description == membrane_description
    assert membrane.label == "membrane"
    assert membrane.keyword_type == KeywordType.CELLULAR_COMPONENT


@pytest.mark.timeout(10)
def test_uniprot_keywords():
    url_format = "http://purl.uniprot.org/keywords/{}"
    url_map = {i: KeywordType.from_uniprot_uri(url_format.format(i)) for i in range(9990, 10000)}
    assert url_map[9990] == KeywordType.TECHNICAL_TERM
    assert url_map[9991] == KeywordType.POST_TRANSLATIONAL_MODIFICATION
    assert url_map[9992] == KeywordType.MOLECULAR_FUNCTION
    assert url_map[9993] == KeywordType.LIGAND
    assert url_map[9994] == KeywordType.DOMAIN
    assert url_map[9995] == KeywordType.DISEASE
    assert url_map[9996] == KeywordType.DEVELOPMENTAL_STAGE
    assert url_map[9997] == KeywordType.CODING_SEQUENCE_DIVERSITY
    assert url_map[9998] == KeywordType.CELLULAR_COMPONENT
    assert url_map[9999] == KeywordType.BIOLOGICAL_PROCESS


@pytest.mark.timeout(10)
def test_obo_keywords():
    biological_process = "biological_process"
    molecular_function = "molecular_function"
    cellular_component = "cellular_component"
    assert KeywordType.from_obo_ns(biological_process) == KeywordType.BIOLOGICAL_PROCESS
    assert KeywordType.from_obo_ns(molecular_function) == KeywordType.MOLECULAR_FUNCTION
    assert KeywordType.from_obo_ns(cellular_component) == KeywordType.CELLULAR_COMPONENT
