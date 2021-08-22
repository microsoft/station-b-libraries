# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This module contains code to draw interactive plots of barcoded plates."""
import plotly.graph_objects as go
import string
from enum import Enum
from typing import List, Dict, Any


PLATE_BORDER_COLOR = "RoyalBlue"
EMPTY_WELL_FILL = "DarkRed"
EMPTY_WELL_BORDER = "Red"
REAGENT_WELL_FILL = "PaleTurquoise"
REAGENT_WELL_BORDER = "LightSeaGreen"


class PlateShape(Enum):
    """An enum that describes the dimensions of a plate."""

    Well24 = (4, 6)
    Well96 = (8, 12)
    Well384 = (16, 24)
    Well1536 = (32, 48)

    @property
    def row_indices(self):
        """Returns the list of row headers of the plate as a list.
        For example: a `Well24` plate which has 4 rows will return `[A, B, C, D]`"""
        if self == PlateShape.Well1536:  # pragma: no cover
            return [x for x in string.ascii_uppercase] + [f"A{x}" for x in string.ascii_uppercase[:6]]
        else:
            return [x for x in string.ascii_uppercase[: self.value[0]]]

    @property
    def col_indices(self):
        """Returns the list of column headers of the plate as a list.

        For example: a `Well24` plate which has 6 columns will return `[1, 2, 3, 4, 5, 6]`"""
        return list(range(1, self.value[1] + 1))


def draw_plate(plate_shape: PlateShape, plate_details: List[Dict[str, Any]]) -> go.Figure:  # type: ignore
    """Returns an interactive figure (of type: `plotly.graph_objects.Figure`) of a barcoded plate.
    The shape of the plate is specified by the `plate_shape` argument.
    Each element in `plate_details` is dictionary that contains details of the barcoded reagents."""
    fig: go.Figure = go.Figure()  # type: ignore
    rdict = {entry["Well"].upper(): entry for entry in plate_details}

    (r, c) = plate_shape.value
    row_indices = plate_shape.row_indices
    row_indices.reverse()
    col_indices = plate_shape.col_indices

    # Set axes properties
    fig.update_xaxes(range=[0, c + 1], fixedrange=True)  # type: ignore
    fig.update_yaxes(range=[0, r + 1], fixedrange=True)  # type: ignore

    # Add Plate
    fig.add_shape(  # type: ignore
        type="rect",
        x0=0.25,
        y0=0.25,
        x1=c + 0.75,
        y1=r + 0.75,
        line=dict(color=PLATE_BORDER_COLOR),
    )

    label_x = []
    label_y = []
    label_list = []

    for r_index in range(0, r):
        for c_index in range(0, c):

            well = f"{row_indices[r_index]}{col_indices[c_index]}"
            well_details = rdict[well]
            sample_id = well_details.get("SampleID", None)
            barcode = well_details.get("Barcode", None)
            if barcode is None or str(barcode) == "nan":
                fig.add_shape(  # type: ignore
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=(c_index + 0.75),
                    y0=(r_index + 0.75),
                    x1=(c_index + 1.25),
                    y1=(r_index + 1.25),
                    line_color=REAGENT_WELL_BORDER,
                )
            else:
                if sample_id is None or str(sample_id) == "nan":
                    fig.add_shape(  # type: ignore
                        type="circle",
                        xref="x",
                        yref="y",
                        x0=(c_index + 0.75),
                        y0=(r_index + 0.75),
                        x1=(c_index + 1.25),
                        y1=(r_index + 1.25),
                        line_color=EMPTY_WELL_BORDER,
                        fillcolor=EMPTY_WELL_FILL,
                        opacity=0.5,
                    )
                    label_x.append(c_index + 1)
                    label_y.append(r_index + 1)
                    label_list.append(f"Barcode: {barcode}<br>No reagent in BCKG with this barcode.")
                else:
                    name = well_details["Name"]

                    fig.add_shape(  # type: ignore
                        type="circle",
                        xref="x",
                        yref="y",
                        x0=(c_index + 0.75),
                        y0=(r_index + 0.75),
                        x1=(c_index + 1.25),
                        y1=(r_index + 1.25),
                        line_color=REAGENT_WELL_BORDER,
                        fillcolor=REAGENT_WELL_FILL,
                    )
                    label_x.append(c_index + 1)
                    label_y.append(r_index + 1)
                    _typeval: str = f"<br>Type: {well_details['Type']}" if "Type" in well_details else ""
                    label_list.append(f"Barcode: {barcode}<br>Name: {name}{_typeval}<br>Sample ID:{sample_id}")

    # Create scatter trace of text labels
    fig.add_trace(  # type: ignore
        go.Scatter(  # type: ignore
            x=label_x,
            y=label_y,
            text=label_list,
            mode="text",
            opacity=0,
        )
    )

    fig.update_layout(  # type: ignore
        xaxis=dict(tickmode="array", tickvals=list(range(1, c + 1)), ticktext=col_indices),
        yaxis=dict(tickmode="array", tickvals=list(range(1, r + 1)), ticktext=row_indices),
        width=650,
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
