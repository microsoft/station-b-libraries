# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import enum
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from abex.plotting.core import PLOT_SCALE
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pandas.core.groupby import SeriesGroupBy


def plot_convergence(
    df: pd.DataFrame,
    objective_col: str,
    batch_num_col: str,
    run_col: str,
    ax: Optional[Axes] = None,
    yscale: PLOT_SCALE = "symlog",
) -> Tuple[Optional[Figure], Axes]:
    """Makes a plot illustrating convergence over multiple batches for (possibly) multiple runs. I.e. this function
    plots the best value of the objective observed so far (running best) as a function of the batch number.
    It also overlays the distribution of points in each batch alongside (as a boxplot)

    Args:
        df: A DataFrame with at least those columns specified by objective_col, batch_num_col and run_col containing
            the points observed during an optimization run.
        objective_col: The column in the dataframe corresponding to observed objective
        batch_num_col: Column in dataframe corresponding to batch number (iteration) for each observation
        run_col: Column specifying which optimization run each point corresponds to
        ax (optional): Axis to plot on. Create a new one if None.
        yscale (optional): Scale for the y-axis (objective). Defaults to "symlog".

    Returns:
        Figure and axis objects with the plot(Return None instead of figure if an axis supplied as argument)
    """
    # Create figure and axis if one not given as argument to this function
    fig, ax = plt.subplots(figsize=(14, 7)) if ax is None else (None, ax)
    # Only keep relevant columns
    df = df[[run_col, batch_num_col, objective_col]].sort_values(by=[run_col, batch_num_col])  # type: ignore # auto

    unique_run_names = df[run_col].unique().tolist()  # type: ignore
    palette = get_color_palette(unique_run_names)
    with sns.axes_style("whitegrid"):
        # Plot the point
        sns.boxplot(
            x=batch_num_col,
            y=objective_col,
            hue=run_col,
            whis=[0, 100],
            data=df,
            dodge=True,
            zorder=10,
            width=0.8,
            palette=palette,
        )

        sns.stripplot(
            x=batch_num_col,
            y=objective_col,
            hue=run_col,
            data=df,
            dodge=True,
            zorder=20,
            linewidth=1,
            edgecolor="gray",
            palette=palette,
        )

        # Find the maximum for each batch
        grouped_df = df.groupby([run_col, batch_num_col])
        max_per_batch_df = grouped_df.max()
        max_per_batch_df = max_per_batch_df.reset_index()  # type: ignore
        for run in max_per_batch_df[run_col].unique():
            # Aggregate by calculating the running maximum (examples have been sorted by batch num. before)
            max_per_batch_df[max_per_batch_df[run_col] == run] = max_per_batch_df[
                max_per_batch_df[run_col] == run
            ].cummax()

        sns.lineplot(
            x=batch_num_col,
            y=objective_col,
            data=max_per_batch_df,
            hue=run_col,
            linewidth=2.6,
            alpha=0.7,
            drawstyle="steps-post",
            palette=palette,
            zorder=2,
        )
        ax.set_yscale(yscale)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        sns.despine(left=True, bottom=True)

        # Remove duplicate legend:
        num_runs = len(unique_run_names)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[:num_runs], labels[:num_runs])
    return fig, ax  # type: ignore # auto


class ConvergencePlotStyles(enum.Enum):
    LINE = "line"
    BOXPLOT = "boxplot"


def get_max_per_batch(df: pd.DataFrame, run_names: List[str], run_col: str, seed_col: str, batch_num_col: str):
    """
    Find the maximum for each batch in order to plot max found so far

    Args:
        df:
        run_names:
        run_col:
        seed_col:
        batch_num_col:

    Returns:

    """
    grouped_df = df.groupby([run_col, seed_col, batch_num_col])
    max_per_batch_df = grouped_df.max()
    max_per_batch_df = max_per_batch_df.reset_index()  # type: ignore
    for run in run_names:
        run_idxs = max_per_batch_df[run_col] == run
        for seed in max_per_batch_df[run_idxs][seed_col].unique():
            seed_idxs = max_per_batch_df[seed_col] == seed
            # Get all rows with given seed and run-name (values for a single optimization run)
            idxs = run_idxs & seed_idxs
            # Aggregate by calculating the cumulative maximum (examples have been sorted by batch num. before)
            max_per_batch_df[idxs] = max_per_batch_df[idxs].cummax()
    return max_per_batch_df


def plot_multirun_convergence(
    df: pd.DataFrame,
    objective_col: str,
    batch_num_col: str,
    run_col: str,
    seed_col: str,
    ax: Optional[Axes] = None,
    yscale: PLOT_SCALE = "symlog",
    running_best_percentiles: Tuple[float, float] = (0.05, 0.95),
    add_boxplot: bool = True,
    add_scatter: bool = True,
    style_cols: Optional[List[str]] = None,
    plot_style: Union[ConvergencePlotStyles, str] = ConvergencePlotStyles.BOXPLOT,
) -> Tuple[Optional[Figure], Axes]:  # pragma: no cover
    """Makes a plot illustrating convergence over multiple batches for multiple runs/configurations, and aggregates over
    multiple seeded (sub)-runs.
    Plots the best value of the objective observed so far (running best) as a function of the batch number, showing
    the mean of the cumulative best for each run/configuration with a solid line, and the range of values
    (from 10th to 90th percentile) with a shaded region.

    Also overlays the distribution of points in each batch alongside (as a boxplot and a scatter)

    Args:
        df: A DataFrame with at least those columns specified by objective_col, batch_num_col and run_col containing
            the points observed during an optimization run.
        objective_col: The column in the dataframe corresponding to observed objective
        batch_num_col: Column in dataframe corresponding to batch number (iteration) for each observation
        run_col: Column specifying which optimization run each point corresponds to
        seed_col: Column specifying the id/seed of the sub-run of the row corresponds to. The "cumulative best"
            measurements will be aggregated for runs with different sub-run seeds/ids.
        ax (optional): Axis to plot on. Create a new one if None.
        yscale (optional): Scale for the y-axis (objective). Defaults to "symlog".
        running_best_percentiles: The range of cumulative best values to display with the shaded region.
        add_boxplot: whether to plot the boxplot showing distribution of batch points.
        add_scatter: whether to scatterplot the distribution of batch points.
        style_cols: The name of the columns on which to base the style. If not specified (or None), the lines won't
            be styled differently
        plot_style: whether to draw convergence with overlapping lines, or as a boxplot visually separating the
            outcomes at each batch.

    Returns:
        Figure and axis objects with the plot(Return None instead of figure if an axis supplied as argument)
    """
    style_cols = style_cols or []
    # Create figure and axis if one not given as argument to this function
    fig, ax = plt.subplots(figsize=(14, 7)) if ax is None else (None, ax)

    # Only keep relevant columns
    cols_to_keep = list(set([run_col, batch_num_col, objective_col] + style_cols))
    df = df[cols_to_keep].sort_values(by=[run_col, seed_col, batch_num_col])  # type: ignore # auto

    run_names, palette = get_run_names_palette(df, style_cols, run_col)
    num_runs = len(run_names)
    with sns.axes_style("whitegrid"):
        # Plot the points

        if add_boxplot:
            sns.boxplot(
                x=batch_num_col,
                y=objective_col,
                hue=run_col,
                hue_order=run_names,
                whis=[0, 100],
                data=df,
                dodge=0.4,
                zorder=5,
                color=".8",
                width=0.8,
                linewidth=0.5,
                ax=ax,
            )
        if add_scatter:
            sns.stripplot(
                x=batch_num_col,
                y=objective_col,
                hue=run_col,
                hue_order=run_names,
                data=df,
                dodge=0.4,
                zorder=10,
                alpha=0.4,
                palette=sns.color_palette(palette, desat=0.6),
                ax=ax,
            )

        max_per_batch_df = get_max_per_batch(df, run_names, run_col, seed_col, batch_num_col)
        plot_style = ConvergencePlotStyles(plot_style)
        if plot_style == ConvergencePlotStyles.BOXPLOT:
            prev_artists = list(ax.get_children())
            ax.set_yscale(yscale)
            sns.boxplot(
                x=batch_num_col,
                y=objective_col,
                data=max_per_batch_df,
                hue=run_col,
                hue_order=run_names,
                dodge=0.4,
                whis=np.array(running_best_percentiles) * 100,  # Seaborn takes values in percentages
                showfliers=False,
                palette=palette,
                width=0.8,
                linewidth=2.3,
                zorder=50,
                ax=ax,
            )
            # Properly order this boxplot in front of other plots
            convergence_boxplot_artists = list(set(ax.get_children()) - set(prev_artists))
            for artist in convergence_boxplot_artists:
                artist.set_zorder(artist.get_zorder() + 20)
        else:
            sns.lineplot(
                x=batch_num_col,
                y=objective_col,
                data=max_per_batch_df,
                hue=run_col,
                style=style_cols[0],  # take the first specified styled_subset as most important
                hue_order=run_names,
                linewidth=2.6,
                drawstyle="steps-post",
                err_kws={"step": "post"},
                ci=None,
                err_style="bars",
                palette=palette,
                alpha=0.8,
                zorder=20,
                ax=ax,
            )
            ax.set_yscale(yscale)  # Set y-scale has to appear after sns.lineplot. not sure why, but it breaks otherwise
            # Add confidence interval
            for i, run in enumerate(run_names):
                # Get objectives for this run grouped by batch number
                objective_by_batch = max_per_batch_df[max_per_batch_df[run_col] == run].groupby(batch_num_col)[
                    objective_col
                ]
                # Get the lower and upper percentiles for the confidence bounds on the objective
                bounds = objective_by_batch.quantile(running_best_percentiles).unstack()
                ax.fill_between(
                    x=bounds.index, y1=bounds.iloc[:, 0], y2=bounds.iloc[:, 1], alpha=0.1, color=palette[i], step="post"
                )
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        sns.despine(left=True, bottom=True)

        # Make the legend:
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=run_names[i], markerfacecolor=palette[i], markersize=15)
            for i in range(len(run_names))
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=num_runs,
            frameon=False,
        )
    return fig, ax  # type: ignore # auto


def plot_multirun_convergence_per_sample(
    df: pd.DataFrame,
    objective_col: str,
    batch_num_col: str,
    run_col: str,
    seed_col: str,
    ax: Optional[Axes] = None,
    yscale: PLOT_SCALE = "symlog",
    cumulative_best_quantiles: Tuple[float, float] = (0.1, 0.9),
    default_objective: float = 0.0,
) -> Tuple[Optional[Figure], Axes]:  # pragma: no cover
    """Makes a plot illustrating convergence w.r.t to number of samples obserbed for multiple runs/configurations.
    Aggregates over multiple seeded (sub)-runs.
    Plots the best value of the objective observed so far (cumulative best) vs. iteration , showing
    the mean of the cumulative best for each run/configuration with a solid line, and the range of values
    (from 10th to 90th percentile) with a shaded region.

    Args:
        df: A DataFrame with at least those columns specified by objective_col, batch_num_col and run_col containing
            the points observed during an optimization run.
        objective_col: The column in the dataframe corresponding to observed objective
        batch_num_col: Column in dataframe corresponding to batch number (iteration) for each observation
        run_col: Column specifying which optimization run each point corresponds to
        seed_col: Column specifying the id/seed of the sub-run of the row corresponds to. The "cumulative best"
            measurements will be aggregated for runs with different sub-run seeds/ids.
        ax (optional): Axis to plot on. Create a new one if None.
        yscale (optional): Scale for the y-axis (objective). Defaults to "symlog".
        cumulative_best_quantiles: The range of cumulative best values to display with the shaded region.
        default_objective: The value of the objective to plot before any samples have been observed.

    Returns:
        Figure and axis objects with the plot(Return None instead of figure if an axis supplied as argument)
    """
    # Create figure and axis if one not given as argument to this function
    fig, ax = plt.subplots(figsize=(14, 7)) if ax is None else (None, ax)
    # Only keep relevant columns
    df = df[[run_col, seed_col, batch_num_col, objective_col]]  # type: ignore # auto
    df = df.sort_values(by=[run_col, seed_col, batch_num_col])  # type: ignore # auto

    unique_run_names = list(df[run_col].unique())  # Get run names to fix the hue ordering
    palette = get_color_palette(unique_run_names)
    iter_count_col = "iteration_count"
    with sns.axes_style("whitegrid"):
        # - Plot maximum found so far vs. iter number
        # Find the maximum for each batch
        grouped_df = df.groupby([run_col, seed_col, batch_num_col])
        max_per_batch_df = grouped_df.agg(
            objective_max=pd.NamedAgg(column=objective_col, aggfunc="max"),  # type: ignore # auto
            batch_size=pd.NamedAgg(column=objective_col, aggfunc="count"),  # type: ignore # auto
        )
        max_per_batch_df = max_per_batch_df.reset_index()
        max_per_batch_df[iter_count_col] = np.nan
        for run in unique_run_names:
            run_idxs = max_per_batch_df[run_col] == run
            for seed in max_per_batch_df[run_idxs][seed_col].unique():  # type: ignore # auto
                seed_idxs = max_per_batch_df[seed_col] == seed
                # Get all rows with given seed and run-name (values for a single optimization run)
                idxs = run_idxs & seed_idxs
                # Aggregate by calculating the cumulative maximum (examples have been sorted by batch num. before)
                cummax = max_per_batch_df["objective_max"][idxs].cummax()  # type: ignore # auto
                max_per_batch_df["objective_max"][idxs] = cummax  # type: ignore # auto
                # Calculate the number of examples seen so var
                cumsum = max_per_batch_df["batch_size"][idxs].cumsum()  # type: ignore # auto
                max_per_batch_df[iter_count_col][idxs] = cumsum  # type: ignore # auto
                # Add a 0 objective value at the start
                max_per_batch_df = max_per_batch_df.append(
                    {run_col: run, seed_col: seed, iter_count_col: 0, "objective_max": default_objective},
                    ignore_index=True,
                )

        sns.lineplot(
            x=iter_count_col,
            y="objective_max",
            data=max_per_batch_df,
            hue=run_col,
            hue_order=unique_run_names,
            linewidth=2.6,
            drawstyle="steps-post",
            err_kws={"step": "post"},
            ci=None,
            palette=palette,
            alpha=0.8,
            zorder=20,
        )
        # Add confidence interval
        for i, run in enumerate(unique_run_names):
            bounds = (
                max_per_batch_df[max_per_batch_df[run_col] == run]
                .groupby(iter_count_col)["objective_max"]  # type: ignore # auto
                .quantile(cumulative_best_quantiles)  # type: ignore # auto
                .unstack()
            )
            ax.fill_between(
                x=bounds.index, y1=bounds.iloc[:, 0], y2=bounds.iloc[:, 1], alpha=0.1, color=palette[i], step="post"
            )
        ax.set_yscale(yscale)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlim(0, None)
        sns.despine(left=True, bottom=True)

        # Make the legend:
        handles, labels = ax.get_legend_handles_labels()
        num_runs = len(unique_run_names)
        line_handles, line_labels = handles[:num_runs], labels[:num_runs]
        plt.legend(line_handles, line_labels, loc="lower right")
    return fig, ax  # type: ignore # auto


def get_color_groups(df: pd.DataFrame, style_cols: List[str], run_col: str) -> List[List[str]]:
    """
    Group experiments according to the style columns provided.
    Args:
        df: A dataframe containing at least run_col and all columns specified in style_cols
        style_cols: A list of column names to group by
        run_col: Specifies the column containing the names to be grouped

    Returns: A list of groups (lists) of strings which represent the items in 'run_col', grouped by shared
    style_cols

    """
    assert len(style_cols) > 0
    grouped_df_run_col = df.groupby(style_cols)
    names_grouped = grouped_df_run_col[run_col]
    assert isinstance(names_grouped, SeriesGroupBy)
    color_groups_agg = names_grouped.agg(set)
    color_groups = color_groups_agg.values.tolist()  # type: ignore
    return [list(s) for s in color_groups]


def get_color_palette(color_groups: Union[List[str], List[List[str]]]) -> List[Tuple[float, float, float]]:
    """
    Given a list of color groups, return a Seaborn color palette of 1 color per point to be plotted.
    Args:
        color_groups:a list of groups of names (i.e. a list of lists of strings) or a list of
        unique names (a list of strings),

    Returns: A Seaborn color palette with 1 color (Tuple[float, float, float]) per name - not 1 per group!

    """
    num_colors = len(color_groups)
    return sns.color_palette("Set1", num_colors)  # type: ignore


def get_run_names_palette(
    df: pd.DataFrame, style_cols: List[str], run_col: str
) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    """
    Groups the given dataframe by the columns specified in style_cols and for each entry in run_col, determines
    a color value depending on whether it is grouped with any other entries. Entries in the same group will have
    different saturations of the same color, so that they can easily be identified as sharing some property(/ies)
    in the plots.
    Args:
        df: A pandas DataFrame containing at least the columns specified in style_cols and run_col
        style_cols: A list of column names to group by
        run_col: Specifies the column containing the names to be grouped

    Returns:

    """
    if len(style_cols) > 0:
        color_groups = get_color_groups(df, style_cols, run_col)
        color_palette = get_color_palette(color_groups)
        palette, run_names = get_saturation_grouped_palette(color_palette, color_groups, saturation_range=(0.4, 1.0))
    else:
        # Get run names to fix the hue ordering
        run_names = list(df[run_col].unique())
        palette = get_color_palette(run_names)  # type: ignore
    return run_names, palette  # type: ignore


def plot_objective_distribution_convergence(
    df: pd.DataFrame,
    objective_col: str,
    batch_num_col: str,
    run_col: str,
    ax: Optional[Axes] = None,
    yscale: PLOT_SCALE = "symlog",
    whis_percentiles: Tuple[float, float] = (0.1, 0.9),
    style_cols: Optional[List[str]] = None,
) -> Tuple[Optional[Figure], Axes]:  # pragma: no cover
    """
    Plots the distribution of the values in the "Objective" column in dataframe for each run specified in column
    run_col against the iteration (batch numer).

    The plot visualises this distribution with a boxplot.

    Args:
        df: A DataFrame with at least those columns specified by objective_col, batch_num_col and run_col containing
            the distribution of points observed during an optimization run.
        objective_col: The column in the dataframe corresponding to observed objective
        batch_num_col: Column in dataframe corresponding to batch number (iteration) for each observation
        run_col: Column specifying which optimization run each point corresponds to
        ax (optional): Axis to plot on. Create a new one if None.
        yscale (optional): Scale for the y-axis (objective). Defaults to "symlog".
        whis_percentiles: The percentiles to display with whiskers on boxplots
        style_col: The name of the column on which to base the style. If not specified (or None), the lines won't
            be styled differently

    Returns:
        Figure and axis objects with the plot visualising the distribution of "Objective" in dataframe df
        for each batch number.
    """
    style_cols = style_cols or []
    # Create figure and axis if one not given as argument to this function
    fig, ax = plt.subplots(figsize=(14, 7)) if ax is None else (None, ax)

    # Only keep relevant columns
    cols_to_keep = list(set([run_col, batch_num_col, objective_col] + style_cols))

    df = df[cols_to_keep].sort_values(by=[run_col, batch_num_col])  # type: ignore # auto

    run_names, palette = get_run_names_palette(df, style_cols, run_col)

    with sns.axes_style("whitegrid"):
        ax.set_yscale(yscale)
        # Plot the distribution of the points
        sns.boxplot(
            x=batch_num_col,
            y=objective_col,
            data=df,
            hue=run_col,
            hue_order=run_names,
            dodge=0.4,
            whis=np.array(whis_percentiles) * 100,  # Seaborn takes values in percentages
            showfliers=False,
            palette=palette,
            width=0.8,
            linewidth=2.3,
            zorder=50,
            ax=ax,
        )
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        sns.despine(left=True, bottom=True)

        # Make the legend:
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=run_names[i], markerfacecolor=palette[i], markersize=15)
            for i in range(len(run_names))
        ]
        ax.legend(handles=legend_elements, loc="best")
    return fig, ax  # type: ignore # auto


ColorType = Union[Tuple[float, float, float], Tuple[float, float, float, float]]


def sequential_saturation_palette(
    color: ColorType, saturation_range: Tuple[float, float], num_colors: int
) -> List[ColorType]:  # pragma: no cover
    """Make a sequential color palette from a given color, where the colors are differentiated through saturation.

    If color is a list of colors, saturate them sequentially from least to most saturated.

    """
    saturations = np.linspace(*saturation_range, num_colors)
    palette = [sns.desaturate(sns.saturate(color), prop=sat) for sat in saturations]
    return palette  # type: ignore # auto


def sequentially_darken_colors(
    colors: List[ColorType],
    lightness_range: Tuple[float, float],
) -> List[ColorType]:  # pragma: no cover
    """Take a list of colors and modify their lightness sequentially by a factor in the range given
    by lightness_range.
    """
    assert 0 <= lightness_range[0] < lightness_range[1] <= 1.0, "Ligthness range must be an interval within [0, 1]"
    for i, lightness_factor in enumerate(np.linspace(*lightness_range, len(colors))):
        # Darken color at index i by a given lightness_factor
        colors[i]: ColorType = tuple(map(lambda x: lightness_factor * x, colors[i]))  # type: ignore
    return colors


def get_saturation_grouped_palette(
    base_palette, color_groups: List[List], saturation_range: Tuple[float, float]
) -> Tuple[List[ColorType], List]:  # pragma: no cover
    """Make a hierarchically grouped palette where each group has a different color (taken from base_palette),
    and within each group entries are differentiated by saturation.

    Args:
        base_palette: The base color palette specifying colors for each group.
        color_groups (List[List]): A list of lists of entries to assign a palette to. Each inner list will get a hue
            from the base color palette, and each entry in the inner list will be differentiated by saturation.
        saturation_range (Tuple[float, float]): The saturation range for the sequential saturation pallette of
            sub-groups.

    Returns:
        Tuple[List[Tuple[float, float, float]], List]: Tuple with first element being the color palette, and the second
            element being the flattened version of color_groups, with each entry corresponding to one color in
            the color palette.
    """
    run_names = [run_name for grouped_runs in color_groups for run_name in grouped_runs]
    palette = []
    for i, group in enumerate(color_groups):
        if len(group) > 1:
            group_palette = sequential_saturation_palette(
                base_palette[i], saturation_range=saturation_range, num_colors=len(group)
            )
            group_palette = sequentially_darken_colors(group_palette, lightness_range=(0.66, 1.0))
            palette.extend(group_palette)
        else:
            palette.append(base_palette[i])
    return palette, run_names
