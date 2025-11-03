"""Visualization utilities for test data analysis and statistical comparisons."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
import textwrap

# bar chart colors
CUSTOM_PALETTE = [
    "#3F8CB7",
    "#FCEF88",
    "#5D5797",
    "#62AC6B",
    "#B74779",
    "#2C4E98",
    "#CCCCE7",
    "#08625B",
    "#D15056",
    "#F2A541",
    "#FFC0CB",
    "#A9A9A9",
    "#8FBC8F",
    "#E6E6FA",
    "#FFD700",
    "#40E0D0",
    "#FF6347",
    "#90EE90",
    "#D3D3D3",
    "#FFA500",
    "#00CED1",
    "#FF4500",
    "#32CD32",
    "#B0C4DE",
    "#FF8C00",
    "#20B2AA",
    "#FF69B4",
    "#87CEEB",
    "#FF1493",
    "#4682B4",
    "#DB7093",
]
# color for chick data
CHICK_RED = "#AF264A"


def _stats(group, column="percent_correct", mu=0.5):
    data = group[column]
    avgs = data.mean()
    sd = data.std()
    count = data.count()
    se = sd / np.sqrt(count) if count > 0 else np.nan
    tval, pval = stats.ttest_1samp(data.dropna(), mu) if count > 0 else (np.nan, np.nan)
    df = count - 1
    cohensd = (avgs - mu) / sd if sd != 0 else np.nan
    return pd.Series(
        {
            "avgs": avgs,
            "sd": sd,
            "count": count,
            "tval": tval,
            "df": df,
            "pval": pval,
            "se": se,
            "cohensd": cohensd,
        }
    )


def compute_stats(data: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """
    Compute statistics for test data grouped by condition and agent.

    Args:
        data: DataFrame containing test results.
        results_dir: Directory to save statistics CSV file.

    Returns:
        DataFrame with computed statistics including averages, standard deviations,
        t-test values, and effect sizes.
    """
    grouped = data.groupby(["imprint.cond", "agent", "test.cond"])
    by_test_cond = grouped.apply(_stats).reset_index()  # keep group columns as columns
    by_test_cond["imp_agent"] = (
        by_test_cond["imprint.cond"].astype(str)
        + "_"
        + by_test_cond["agent"].astype(str)
    )
    by_test_cond.to_csv(results_dir / "stats_by_agent.csv", index=False)
    return by_test_cond


def make_bar_charts(
    data,
    dots,
    y_col,
    error_min_col,
    error_max_col,
    img_name,
    color_bars,
    chick_data=None,
):
    """
    Create bar charts with error bars for test performance data.

    Args:
        data: DataFrame containing the bar chart data.
        dots: DataFrame containing individual data points to overlay.
        y_col: Column name for y-axis values.
        error_min_col: Column name for minimum error values.
        error_max_col: Column name for maximum error values.
        img_name: Output filename for the chart.
        color_bars: Whether to color bars by test condition.
        chick_data: Optional DataFrame containing chick behavioral data for comparison.
    """
    plt.figure(figsize=(6, 6))
    sns.set_style("white")
    ax = plt.gca()
    # Hide the right and top spines
    ax.spines[["right", "top"]].set_visible(False)

    y = data[y_col]
    yerr = [data[y_col] - data[error_min_col], data[error_max_col] - data[y_col]]

    x_categories = data["test.cond"].unique()

    data["test.cond"] = pd.Categorical(
        data["test.cond"], categories=x_categories, ordered=True
    )

    if color_bars:
        color_mapping = {}
        color_iterator = iter(CUSTOM_PALETTE)  # Use an iterator for the color list

        for x in x_categories:
            # Check for the special case, ignoring case.
            if str(x).lower() == "rest":
                color_mapping[x] = "darkgrey"
            else:
                # Try to get the next color from the iterator.
                try:
                    color_mapping[x] = next(color_iterator)
                except StopIteration:
                    # This block runs if we run out of colors in the y list.
                    # You can decide what to do here. For now, we'll assign None.
                    print(f"Warning: Ran out of colors. Assigning None to '{x}'.")
                    color_mapping[x] = None

        colors = data["test.cond"].map(color_mapping)
    else:
        colors = "grey"

    x_pos = np.arange(len(x_categories))
    ax.bar(x_pos, y, yerr=yerr, color=colors, capsize=14, width=0.7, linewidth=0)

    ax.set_xticks(x_pos)
    labels = ["\n".join(textwrap.wrap(str(l), 10)) for l in x_categories]
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7.5, fontweight="bold")

    # Add significance asterisks
    pvals = data["pval"]
    used_symbols = set()
    for i, pval in enumerate(pvals):
        symbol = ""
        if pval <= 0.001:
            symbol = "***"
            used_symbols.add("***")
        elif pval <= 0.01:
            symbol = "**"
            used_symbols.add("**")
        elif pval <= 0.05:
            symbol = "*"
            used_symbols.add("*")

        if symbol:
            # Adjust y position to be below the x-axis line
            ax.text(
                x_pos[i],
                # -0.15,  # Adjust vertical position based on number of unique symbols
                -0.035
                * (
                    labels[i].count("\n") + 1
                ),  # Adjust vertical position based on number of unique symbols
                symbol,
                ha="center",
                va="top",
                fontsize=14,
                fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )

    if used_symbols:
        significance_texts = []
        if "***" in used_symbols:
            significance_texts.append("*** p \u2264 0.001")
        if "**" in used_symbols:
            significance_texts.append("** p \u2264 0.01")
        if "*" in used_symbols:
            significance_texts.append("* p \u2264 0.05")

        if significance_texts:
            plt.figtext(
                0.5,
                0.01,
                ", ".join(significance_texts),
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    if chick_data is not None:
        chick_data_filtered = chick_data[chick_data["test.cond"].isin(x_categories)]
        if "experiment" in data.columns:
            # Get the value from the first row of 'experiment'
            experiment = data["experiment"].iloc[0]  # Remove the "Exp" from beginning
            # Filter the DataFrame
            chick_data_filtered = chick_data_filtered[
                chick_data_filtered["experiment"].apply(lambda x: experiment == x)
            ]

        if "exp.cond" in data.columns:
            # Get the value from the first row of 'exp.cond'
            imprint_cond = data["exp.cond"].iloc[0]
            # Filter the DataFrame
            chick_data_filtered = chick_data_filtered[
                chick_data_filtered["imprint.cond"].apply(lambda x: imprint_cond == x)
            ]

        elif "imprint.cond" in chick_data_filtered.columns:
            imprints = data["imprint.cond"].unique()
            chick_data_filtered = chick_data_filtered[
                chick_data_filtered["imprint.cond"].apply(lambda x: x in imprints)
            ]

        if not chick_data_filtered.empty:
            chick_data_filtered["test.cond"] = pd.Categorical(
                chick_data_filtered["test.cond"], categories=x_categories, ordered=True
            )
            chick_x_pos = chick_data_filtered["test.cond"].cat.codes - 1

            shift = 0
            for xi, yi, yerr, i in zip(
                chick_x_pos,
                chick_data_filtered["avg"],
                chick_data_filtered["avg_dev"],
                range(len(chick_x_pos)),
            ):
                if x_categories[i].lower() == "rest":
                    shift = 1
                rect = plt.Rectangle(
                    (xi + shift - 0.35, yi - yerr),
                    0.7,
                    2 * yerr,
                    color=CHICK_RED,
                    alpha=0.2,
                )
                ax.add_patch(rect)
                # Calculate the y-coordinate for the horizontal line
                y_center = rect.get_y() + rect.get_height() / 2
                # Add the horizontal line
                line = Line2D(
                    [rect.get_x(), rect.get_x() + rect.get_width()],
                    [y_center, y_center],
                    c=CHICK_RED,
                )
                ax.add_line(line)

    if dots is not None and "test.cond" in dots.columns:
        dots["test.cond"] = pd.Categorical(
            dots["test.cond"], categories=x_categories, ordered=True
        )
        dot_x_pos = dots["test.cond"].cat.codes
        # Add dots to bar chart

        not_nan_indices = pd.notna(dots["avgs"])
        x_vals = dot_x_pos[not_nan_indices]
        y_vals = dots["avgs"][not_nan_indices]
        if len(y_vals) != 0:
            sns.stripplot(x=x_vals, y=y_vals, ax=ax, color="black", jitter=0.3, size=7)
        else:
            print("All values were none. No dots to add.")

    ax.axhline(0.5, linestyle="--", color="grey")
    ax.set_xlabel("Test Condition", fontweight="bold", fontsize=14, labelpad=15)
    ax.set_ylabel("Percent Correct", fontweight="bold", fontsize=14)
    ax.set_ylim(0, 1.04)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels(
        ["{:.0%}".format(t) for t in np.linspace(0, 1, 11)],
        fontsize=7.5,
        fontweight="bold",
    )
    plt.tight_layout(pad=2.0)
    plt.savefig(img_name)
    plt.close()


def agent_bar_charts(
    data: pd.DataFrame,
    results_dir: Path,
    color_bars: bool,
    chick_data: pd.DataFrame = None,
):
    """
    Generate individual bar charts for each agent's test performance.

    Args:
        data: DataFrame containing statistics by agent.
        results_dir: Directory to save the generated charts.
        color_bars: Whether to color bars by test condition.
        chick_data: Optional DataFrame containing chick behavioral data for comparison.
    """
    for imp_agent in data["imp_agent"].unique():
        bar_data = data[data["imp_agent"] == imp_agent]
        bar_data["error_min"] = bar_data["avgs"] - bar_data["se"]
        bar_data["error_max"] = bar_data["avgs"] + bar_data["se"]
        make_bar_charts(
            data=bar_data,
            dots=None,
            y_col="avgs",
            error_min_col="error_min",
            error_max_col="error_max",
            img_name=results_dir / f"{imp_agent}_test.png",
            color_bars=color_bars,
            chick_data=chick_data,
        )


def stats_by_imprint_cond(data: pd.DataFrame, results_dir: Path):
    """
    Compute statistics grouped by imprint condition.

    Args:
        data: DataFrame containing test results.
        results_dir: Directory to save statistics CSV file.

    Returns:
        DataFrame with statistics aggregated by imprint condition.
    """
    grouped_imp = data.groupby(["imprint.cond", "test.cond"])
    by_imp_cond = grouped_imp.apply(lambda g: _stats(g, column="avgs")).reset_index()
    by_imp_cond.to_csv(results_dir / "stats_by_imp_cond.csv", index=False)
    return by_imp_cond


def imprint_cond_bar_charts(
    by_imp_cond, by_test_cond, results_dir, color_bars, chick_data=None
):
    """
    Generate bar charts grouped by imprint condition.

    Args:
        by_imp_cond: DataFrame with statistics by imprint condition.
        by_test_cond: DataFrame with statistics by test condition.
        results_dir: Directory to save the generated charts.
        color_bars: Whether to color bars by test condition.
        chick_data: Optional DataFrame containing chick behavioral data for comparison.
    """
    for imp_cond in by_imp_cond["imprint.cond"].unique():
        bar_data = by_imp_cond[
            (
                by_imp_cond["imprint.cond"] == imp_cond
            )  # & (by_imp_cond['test.cond'] != "Rest")
        ]
        dot_data = by_test_cond[
            (
                by_test_cond["imprint.cond"] == imp_cond
            )  # & (by_test_cond['test.cond'] != "Rest")
        ]
        bar_data["error_min"] = bar_data["avgs"] - bar_data["se"]
        bar_data["error_max"] = bar_data["avgs"] + bar_data["se"]
        img_name = results_dir / f"{imp_cond}_test.png"
        make_bar_charts(
            data=bar_data,
            dots=dot_data,
            y_col="avgs",
            error_min_col="error_min",
            error_max_col="error_max",
            img_name=img_name,
            color_bars=color_bars,
            chick_data=chick_data,
        )


def stats_overall(
    data: pd.DataFrame, results_dir: Path, chick_data: pd.DataFrame = None
):
    """
    Compute overall statistics across all agents and conditions.

    Args:
        data: DataFrame containing test results.
        results_dir: Directory to save statistics CSV file.
        chick_data: Optional DataFrame containing chick behavioral data for matching conditions.

    Returns:
        DataFrame with overall statistics across conditions.
    """
    if chick_data is not None:
        if "experiment" in chick_data.columns:
            valid_experiments = set(str(x) for x in chick_data["experiment"].unique())

            def get_experiment(cond):
                try:
                    # First attempt
                    exp_val = cond.split("_")[0][3:]
                    if exp_val in valid_experiments:
                        return exp_val
                    # Second attempt
                    exp_val = cond.split("_")[0]
                    if exp_val in valid_experiments:
                        return exp_val
                except IndexError:
                    pass
                return None

            data["experiment"] = data["imprint.cond"].apply(get_experiment)
            data.dropna(subset=["experiment"], inplace=True)

        if (
            "imprint.cond" in chick_data.columns
            and len(data["imprint.cond"].unique()) > 1
        ):
            chick_imprint_conds = set(chick_data["imprint.cond"].unique())

            def match_imprint_cond(agent_cond):
                for chick_cond in chick_imprint_conds:
                    if agent_cond.endswith(chick_cond):
                        return chick_cond
                return None

            data["exp.cond"] = data["imprint.cond"].apply(match_imprint_cond)
            data.dropna(subset=["exp.cond"], inplace=True)
    group_by_cols = ["test.cond"]
    if "exp.cond" in data.columns:
        group_by_cols.append("exp.cond")
    # if "experiment" in data.columns:
    #     group_by_cols.append("experiment")

    across_imp_cond = (
        data[~data["test.cond"].str.lower().eq("rest")]
        .groupby(group_by_cols, as_index=False)
        .apply(lambda g: _stats(g, column="avgs"))
    )
    across_imp_cond.to_csv(results_dir / "stats_across_all_agents.csv", index=False)
    return across_imp_cond


def exp_cond_bar_charts(
    by_test_cond, across_imp_cond, results_dir, color_bars, chick_data=None
):
    """
    Generate bar charts grouped by experimental condition.

    Args:
        by_test_cond: DataFrame with statistics by test condition.
        across_imp_cond: DataFrame with overall statistics across conditions.
        results_dir: Directory to save the generated charts.
        color_bars: Whether to color bars by test condition.
        chick_data: Optional DataFrame containing chick behavioral data for comparison.
    """
    if "experiment" in across_imp_cond.columns:
        if "exp.cond" in across_imp_cond.columns:
            combinations = zip(
                across_imp_cond["experiment"].unique(),
                across_imp_cond["exp.cond"].unique(),
            )
        else:
            combinations = [
                (exp, None) for exp in across_imp_cond["experiment"].unique()
            ]
    elif "exp.cond" in across_imp_cond.columns:
    # if "exp.cond" in across_imp_cond.columns:
        combinations = [
            (None, exp_cond) for exp_cond in across_imp_cond["exp.cond"].unique()
        ]
    else:
        combinations = [(None, None)]

    for exp, expCond in set(combinations):
        if exp is None and expCond is None:
            # Skip if both are None
            bar_data = across_imp_cond
            dot_data = by_test_cond
        elif exp is None:
            bar_data = across_imp_cond[across_imp_cond["exp.cond"] == expCond]
            dot_data = by_test_cond[by_test_cond["imprint.cond"].str.endswith(expCond)]
        elif expCond is None:
            bar_data = across_imp_cond[across_imp_cond["experiment"] == exp]
            dot_data = by_test_cond[
                by_test_cond["imprint.cond"].str.startswith(f"Exp{exp}")
            ]
        else:
            bar_data = across_imp_cond[
                across_imp_cond["experiment"] == exp
                and across_imp_cond["exp.cond"] == expCond
            ]
            dot_data = by_test_cond[
                by_test_cond["imprint.cond"].str.startswith(f"Exp{exp}")
                and by_test_cond["imprint.cond"].str.endswith(expCond)
            ]

        dot_data = dot_data[~dot_data["test.cond"].str.lower().eq("rest")]
        bar_data = bar_data[~bar_data["test.cond"].str.lower().eq("rest")]

        dot_data = dot_data.reset_index(drop=True)
        bar_data = bar_data.reset_index(drop=True)

        bar_data["error_min"] = bar_data["avgs"] - bar_data["se"]
        bar_data["error_max"] = bar_data["avgs"] + bar_data["se"]

        exp_name = f"Exp{exp}_{expCond}"
        img_name = results_dir / f"{exp_name}_test.png"
        make_bar_charts(
            data=bar_data,
            dots=dot_data,
            y_col="avgs",
            error_min_col="error_min",
            error_max_col="error_max",
            img_name=img_name,
            color_bars=color_bars,
            chick_data=chick_data,
        )


def all_cond_bar_chart(
    by_test_cond, across_imp_cond, results_dir, color_bars, chick_data=None
):
    """
    Generate a single bar chart showing all conditions combined.

    Args:
        by_test_cond: DataFrame with statistics by test condition.
        across_imp_cond: DataFrame with overall statistics across conditions.
        results_dir: Directory to save the generated chart.
        color_bars: Whether to color bars by test condition.
        chick_data: Optional DataFrame containing chick behavioral data for comparison.
    """
    across_imp_cond["error_min"] = across_imp_cond["avgs"] - across_imp_cond["se"]
    across_imp_cond["error_max"] = across_imp_cond["avgs"] + across_imp_cond["se"]
    dot_data = by_test_cond[~by_test_cond["test.cond"].str.lower().eq("rest")]
    bar_data = across_imp_cond[~across_imp_cond["test.cond"].str.lower().eq("rest")]
    img_name = results_dir / "all_imprinting_conds_test.png"
    make_bar_charts(
        data=bar_data,
        dots=dot_data,
        y_col="avgs",
        error_min_col="error_min",
        error_max_col="error_max",
        img_name=img_name,
        color_bars=color_bars,
        chick_data=chick_data,
    )


def test_viz(
    results_dir: Path, chick_file: Path = None, bar_order="default", color_bars=False
):
    """
    Main function to generate all test visualizations and statistical analyses.

    Args:
        results_dir: Directory containing test results and where outputs will be saved.
        chick_file: Optional path to CSV file containing chick behavioral data.
        bar_order: Order for displaying bars in charts (default: "default").
        color_bars: Whether to color bars by test condition.
    """
    # Do not warn about chained assignments
    pd.options.mode.chained_assignment = None
    # Load data
    chick_data = pd.read_csv(chick_file) if chick_file else None
    test_data = pd.read_csv(results_dir / "test_results.csv")

    # calculate correct steps, incorrect steps, and percent correct
    print("Collating data for test trials...")
    test_data["correct_steps"] = np.where(
        test_data["correct.monitor"] == "left",
        test_data["left_steps"],
        test_data["right_steps"],
    )
    test_data["incorrect_steps"] = np.where(
        test_data["correct.monitor"] == "right",
        test_data["left_steps"],
        test_data["right_steps"],
    )
    test_data["percent_correct"] = np.where(
        test_data["correct_steps"] + test_data["incorrect_steps"] != 0,
        test_data["correct_steps"]
        / (test_data["correct_steps"] + test_data["incorrect_steps"]),
        0,
    )

    print("Adjusting bar order...")
    if bar_order == "desc":
        order = (
            test_data.groupby("test.cond")["percent_correct"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
    elif bar_order == "asc":
        order = (
            test_data.groupby("test.cond")["percent_correct"]
            .mean()
            .sort_values()
            .index.tolist()
        )
    elif bar_order != "default":
        order = [x.strip() for x in bar_order.split(",")]
    elif chick_data is not None:
        conditions = test_data["test.cond"].unique().tolist()
        order = (
            chick_data[chick_data["test.cond"].isin(conditions)]["test.cond"]
            .unique()
            .tolist()
        )
    else:
        order = test_data["test.cond"].unique().tolist()
    test_data["test.cond"] = pd.Categorical(
        test_data["test.cond"], categories=order, ordered=True
    )

    print("Computing statistics by agent...")
    by_test_cond = compute_stats(test_data, results_dir)

    print("Creating bar charts by agent...")
    agent_bar_charts(by_test_cond, results_dir, color_bars, chick_data)

    print("Computing statistics by imprinting condition...")
    by_imp_cond = stats_by_imprint_cond(by_test_cond, results_dir)

    print("Creating bar charts by imprinting condition...")
    imprint_cond_bar_charts(
        by_imp_cond, by_test_cond, results_dir, color_bars, chick_data
    )

    print("Computing statistics across all imprinting conditions...")

    across_imp_cond = stats_overall(
        data=by_test_cond, results_dir=results_dir, chick_data=chick_data
    )

    print("Creating bar chart for all imprinting conditions...")
    if "experiment" in across_imp_cond.columns or "exp.cond" in across_imp_cond.columns:
        exp_cond_bar_charts(
            by_test_cond, across_imp_cond, results_dir, color_bars, chick_data
        )
    else:
        all_cond_bar_chart(
            by_test_cond, across_imp_cond, results_dir, color_bars, chick_data
        )
