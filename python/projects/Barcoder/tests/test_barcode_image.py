# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import pytest
from pathlib import Path
import pandas as pd
from psbutils.filecheck import figure_found
from barcoder.draw import draw_plate, PlateShape
from barcoder.well_barcode import get_plate_data


ROOT_DIR = Path(__file__).parent.parent
TEST_DIR = ROOT_DIR / "tests"


@pytest.mark.timeout(10)
def test_barcoder():
    test_file = TEST_DIR / "plate_from_bckg.csv"
    barcode_df = pd.DataFrame(pd.read_csv(test_file))
    barcode_data = barcode_df.to_dict("records")
    barcode_image = draw_plate(plate_shape=PlateShape.Well96, plate_details=barcode_data)  # type: ignore
    barcode_fp = TEST_DIR / "barcode.png"
    barcode_image.write_image(str(barcode_fp.resolve()))  # type: ignore
    assert figure_found(barcode_fp, "test_barcoder")
    barcode_fp.unlink(missing_ok=False)


@pytest.mark.timeout(10)
def test_get_plate_data():
    assert get_plate_data is not None
