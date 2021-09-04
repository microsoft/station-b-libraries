# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import pytest
from pathlib import Path
from uniProt.location import Location
from uniProt.localcopy import LocalCopy


ROOT_DIR = Path(__file__).parent.parent
TEST_FILES_DIR = ROOT_DIR / "tests" / "test_files"
lc = LocalCopy(TEST_FILES_DIR, save_copy=True)


@pytest.mark.timeout(10)
def test_mitochondrion():
    mitochondrion_id = "173"
    mitochondrion: Location = lc.get_location(mitochondrion_id)
    assert mitochondrion.uniprot_id == mitochondrion_id
    assert mitochondrion.label == "Mitochondrion"
    assert len(mitochondrion.part_of) == 0


@pytest.mark.timeout(10)
def test_nucleus_membrane():
    nucleus_membrane_id = "182"
    nucleus_membrane: Location = lc.get_location(nucleus_membrane_id)
    assert nucleus_membrane.uniprot_id == nucleus_membrane_id
    assert nucleus_membrane.label == "Nucleus membrane"
    assert len(nucleus_membrane.part_of) == 2
    assert "147" in nucleus_membrane.part_of
    assert "178" in nucleus_membrane.part_of
    assert "162" == nucleus_membrane.subclass
