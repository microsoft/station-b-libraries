# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path

import pytest
from uniProt.keyword import KeywordDatabase
from uniProt.localcopy import LocalCopy
from uniProt.protein import OBO_KEYWORD_FORMAT, UNIPROT_KEYWORD_FORMAT, Protein, get_id_db_from_uri

ROOT_DIR = Path(__file__).parent.parent
TEST_FILES_DIR = ROOT_DIR / "tests" / "test_files"
lc = LocalCopy(TEST_FILES_DIR, save_copy=True)


@pytest.mark.timeout(10)
def test_vsvg_from_uniprot():
    vsvg_seq = (
        "MKCLLYLAFLFIGVNCKFTIVFPHNQKGNWKNVPSNYHYCPSSSDLNWHNDLIGTAIQVKMPKSHKAIQAD"
        + "GWMCHASKWVTTCDFRWYGPKYITQSIRSFTPSVEQCKESIEQTKQGTWLNPGFPPQSCGYATVTDAEAVI"
        + "VQVTPHHVLVDEYTGEWVDSQFINGKCSNYICPTVHNSTTWHSDYKVKGLCDSNLISMDITFFSEDGELSS"
        + "LGKEGTGFRSNYFAYETGGKACKMQYCKHWGVRLPSGVWFEMADKDLFAAARFPECPEGSSISAPSQTSVD"
        + "VSLIQDVERILDYSLCQETWSKIRAGLPISPVDLSYLAPKNPGTGPAFTIINGTLKYFETRYIRVDIAAPI"
        + "LSRMVGMISGTTTERELWDDWAPYEDVEIGPNGVLRTSSGYKFPLYMIGHGMLDSDLHLSSKAQVFEHPHI"
        + "QDAASQLPDDESLFFGDTGLSKNPIELVEGWFSSWKSSIASFFFIIGLIIGLFLVLRVGIHLCIKLKHTKK"
        + "RQIYTDIEMNRLGK"
    )
    vsvg_id = "P03522"
    vsvg: Protein = lc.get_protein(vsvg_id)
    assert vsvg.uniprot_id == vsvg_id
    assert vsvg.recommended_name == "Glycoprotein"
    assert vsvg.gene == "G"
    assert len(vsvg.isoforms) == 1
    assert vsvg.isoforms[0] == vsvg_seq
    assert len(vsvg.all_names) == 1


@pytest.mark.timeout(10)
def test_keyword_db_from_url():
    uniprot_id = "2"
    obo_id = "GO_0019062"
    uniprot_keyword = UNIPROT_KEYWORD_FORMAT.format(uniprot_id)
    obo_keyword = OBO_KEYWORD_FORMAT.format(obo_id)
    (u_id, u_db) = get_id_db_from_uri(uniprot_keyword)
    (o_id, o_db) = get_id_db_from_uri(obo_keyword)
    assert uniprot_id == u_id
    assert KeywordDatabase.UNIPROT == u_db
    assert obo_id == o_id
    assert KeywordDatabase.ONTOBEE == o_db
