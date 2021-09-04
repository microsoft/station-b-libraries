# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This module is intended to get metadata from BCKG for barcoded reagents."""
import pyBCKG.domain as domain
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Set
from pyBCKG.azurestorage.api import AzureConnection


def get_plate_data(filepath: Path, conn: AzureConnection) -> List[Dict[str, Any]]:  # pragma: no cover
    """Returns a list of dictionaries contains details of the barcoded reagents.
    `filepath`: The file path of the barcoded file.
    `conn`: An instance of `pyBCKG.azurestorage.api.AzureConnection`"""
    barcoded_df: pd.DataFrame = pd.DataFrame(pd.read_csv(filepath))
    barcodes: Set[str] = {str(barcode) for barcode in list(pd.Series(barcoded_df["Barcode"])) if str(barcode) != "nan"}
    reagents = conn.get_reagents_by_barcode(barcodes)
    barcode_map = barcoded_df.to_dict("records")
    reagent_barcodes = {r.barcode: r for r in reagents}
    new_df_list = []
    for entry in barcode_map:
        full_entry = dict(entry)
        if entry["Barcode"] in reagent_barcodes:
            reagent: domain.Reagent = reagent_barcodes[entry["Barcode"]]
            full_entry["Name"] = reagent.name
            full_entry["SampleID"] = reagent.guid
            if isinstance(reagent, domain.DNA) or isinstance(reagent, domain.Chemical):
                full_entry["Type"] = reagent._type.value
        new_df_list.append(full_entry)
    return new_df_list
