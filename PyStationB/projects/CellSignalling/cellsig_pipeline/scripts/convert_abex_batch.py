# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This simple script converts a batch in the ABEX format to the DoE file for Antha (but without adding any
tag information).

Use it as:

.. code-block::

  python scripts/convert_abex_batch.py batch_from_ABEX.csv
"""
import argparse
import itertools

import cellsig_pipeline.steps as steps
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    """Returns the names of the experiments to be created."""
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=str, help="The CSV with columns defined in combinatorics config.")
    parser.add_argument("--output", type=str, default="GeneratedBatch.xlsx")
    parser.add_argument("--output_plot", type=str, default="GeneratedBatchPlot.png")

    return parser.parse_args()


def generate_doe(batch: pd.DataFrame) -> pd.DataFrame:
    """Splits batch from signal_merged to signal1 and signal2. Uses Antha-specific conventions, but
    doesn't add any tags."""
    doe = steps._batch_to_bare_doe(batch)
    return doe


def visualise_batch(batch: pd.DataFrame) -> plt.Figure:
    """A simple figure visualising the batch. (For visual assessment)."""

    def makeplot(ax: plt.Axes, str1: str, str2: str) -> None:
        ax.scatter(batch[str1], batch[str2])
        ax.set_xlabel(str1)
        ax.set_ylabel(str2)
        ax.set_xscale("log")
        ax.set_yscale("log")

    combinations = list(itertools.combinations(batch.columns, 2))
    n: int = len(combinations)

    fig, axs = plt.subplots(1, n, figsize=(n * 5, 5))

    for ax, (name1, name2) in zip(axs.ravel(), combinations):
        makeplot(ax, name1, name2)  # type: ignore # auto

    return fig


if __name__ == "__main__":
    plt.style.use("dark_background")

    args = parse_args()

    # Generate an Antha DoE file
    batch = pd.read_csv(args.batch)
    generate_doe(batch).to_excel(args.output, index=False)  # type: ignore # auto

    # Visualise the batch
    fig = visualise_batch(batch)  # type: ignore # auto
    fig.savefig(args.output_plot)

    # Print the statistics
    for name in batch.columns:  # type: ignore # auto
        col = batch[name]  # type: ignore # auto
        print(
            f"{name[:3]}:\t{col.min():.2f}-{col.max():.2f}"  # type: ignore # auto
            f"\t({col.max() / col.min():.0f} fold)"  # type: ignore # auto
        )  # type: ignore # auto
