from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from matplotlib.lines import Line2D

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
]
# color for chick data
CHICK_RED = "#AF264A"


def sort_cond(data, bar_order: str) -> list[str]:
    match bar_order:
        case "desc":
            order = (
                data.groupby("test.cond")["percent_correct"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
        case "asc":
            order = (
                data.groupby("test.cond")["percent_correct"]
                .mean()
                .sort_values()
                .index.tolist()
            )
        case "default":
            order = data["test.cond"].unique().tolist()
        case _:
            order = [x.strip() for x in bar_order.split(",")]

    return pd.Categorical(data["test.cond"], categories=order, ordered=True)


def _stats(grouped_data, groups, column="percent_correct", mu=0.5):
    data = grouped_data[column]
    stats = (
        {groups[0]: str(grouped_data.name)}
        if len(groups) == 1
        else {k: str(v) for k, v in zip(groups, grouped_data.name)}
    )
    stats["avgs"] = avgs = data.mean()
    stats["sd"] = sd = data.std()
    stats["count"] = count = data.count()

    stats["tval"], pval = (
        ttest_1samp(data.dropna(), mu) if count > 0 else (np.nan, np.nan)
    )
    stats["df"] = count - 1
    stats["pval"] = pval
    stats["se"] = sd / np.sqrt(count) if count > 0 else np.nan
    stats["cohensd"] = (avgs - mu) / sd if sd != 0 else np.nan

    return pd.Series(stats)


def _stats_by_group(data, groups, column="percent_correct", mu=0.5):
    return (
        data.groupby(groups, observed=True)  # group data by groups
        .apply(
            lambda g: _stats(g, groups, column, mu)
        )  # apply _stats to each group, resulting in a single row per group
        .reset_index(drop=True)  # reset the index to clean up the DataFrame
    )


def _compute_agent_stats(data: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    by_test_cond = _stats_by_group(data, ["imprint.cond", "agent", "test.cond"])
    by_test_cond.to_csv(results_dir / "stats_by_agent.csv", index=False)
    return by_test_cond


def make_bar_charts(data, dots, y_col, img_name, chick_data, color_bars):
    plt.figure(figsize=(6, 6))
    sns.set_style("white")
    ax = plt.gca()
    # Hide the right and top spines
    ax.spines[["right", "top"]].set_visible(False)

    # Reorder rows so "Rest" comes first
    data = data.sort_values(
        by="test.cond",
        key=lambda col: (col != "Rest").astype(
            int
        ),  # Assign 0 to "Rest" and 1 to others
    )

    y = data[y_col]
    yerr = [data[y_col] - data["error_min"], data["error_max"] - data[y_col]]

    x_categories = data["test.cond"].unique()

    data["test.cond"] = pd.Categorical(
        data["test.cond"], categories=x_categories, ordered=True
    )

    if color_bars:
        color_mapping = dict(zip(x_categories, CUSTOM_PALETTE))
        colors = data["test.cond"].map(color_mapping)
    else:
        colors = "#737373"

    x_pos = np.arange(len(x_categories))
    ax.bar(x_pos, y, yerr=yerr, color=colors, capsize=14, width=0.7, linewidth=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        x_categories, rotation=0, ha="center", fontsize=7.5, fontweight="bold"
    )

    chick_data_filtered = chick_data[chick_data["test.cond"].isin(x_categories)]
    if not chick_data_filtered.empty:
        chick_data_filtered["test.cond"] = pd.Categorical(
            chick_data_filtered["test.cond"], categories=x_categories, ordered=True
        )
        chick_x_pos = chick_data_filtered["test.cond"].cat.codes  # - 1

        # shift the chick data to the right of Rest
        # shift = int(np.isin("Rest", x_categories))
        shift = 0
        for xi, yi, yerr in zip(
            chick_x_pos,
            chick_data_filtered["avg"],
            chick_data_filtered["avg_dev"],
        ):
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
        sns.stripplot(
            x=dot_x_pos, y=dots["avgs"], ax=ax, color="black", jitter=0.3, size=7
        )

    ax.axhline(0.5, linestyle="--", color="grey")
    ax.set_xlabel("Test Condition", fontweight="bold", fontsize=14)
    ax.set_ylabel("Percent Correct", fontweight="bold", fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels(
        ["{:.0%}".format(t) for t in np.linspace(0, 1, 11)],
        fontsize=7.5,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()


def agent_bar_charts(
    data: pd.DataFrame, results_dir: Path, chick_data: pd.DataFrame, color_bars: bool
):
    # Get the imprinting condition x agent combination for each row
    data["imp_agent"] = data["imprint.cond"] + "_" + data["agent"].astype(str)
    # create a bar chart for each imprinting condition x agent combination
    for imp_agent in data["imp_agent"].unique():
        # get all of the data for the current imprinting condition x agent combination
        bar_data = data[data["imp_agent"] == imp_agent]
        # calculate the error bars
        bar_data["error_min"] = bar_data["avgs"] - bar_data["se"]
        bar_data["error_max"] = bar_data["avgs"] + bar_data["se"]
        # create the bar chart
        make_bar_charts(
            data=bar_data,
            dots=None,
            y_col="avgs",
            img_name=results_dir / f"{imp_agent}_test.png",
            chick_data=chick_data,
            color_bars=color_bars,
        )


def stats_by_imprint_cond(data: pd.DataFrame, results_dir: Path):
    by_imp_cond = _stats_by_group(data, ["imprint.cond", "test.cond"], column="avgs")
    by_imp_cond.to_csv(results_dir / "stats_by_imp_cond.csv", index=False)
    return by_imp_cond


def imprint_cond_bar_charts(
    by_imp_cond, by_test_cond, results_dir, chick_data, color_bars
):
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
            img_name=img_name,
            chick_data=chick_data,
            color_bars=color_bars,
        )


def stats_overall(data: pd.DataFrame, results_dir: Path):
    data_no_rest = data[data["test.cond"] != "Rest"]
    across_imp_cond = _stats_by_group(data_no_rest, ["test.cond"], column="avgs")
    across_imp_cond.to_csv(results_dir / "stats_across_all_agents.csv", index=False)
    return across_imp_cond


def all_cond_bar_chart(
    by_test_cond, across_imp_cond, results_dir, chick_data, color_bars
):
    across_imp_cond["error_min"] = across_imp_cond["avgs"] - across_imp_cond["se"]
    across_imp_cond["error_max"] = across_imp_cond["avgs"] + across_imp_cond["se"]
    dot_data = by_test_cond[by_test_cond["test.cond"] != "Rest"]
    img_name = results_dir / "all_imprinting_conds_test.png"
    make_bar_charts(
        data=across_imp_cond,
        dots=dot_data,
        y_col="avgs",
        img_name=img_name,
        chick_data=chick_data,
        color_bars=color_bars,
    )


def test_viz(results_dir: Path, chick_file: Path, bar_order="default", color_bars=True):
    # Do not warn about chained assignments
    pd.options.mode.chained_assignment = None
    # Load data
    chick_data = pd.read_csv(chick_file)
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
    test_data["percent_correct"] = test_data["correct_steps"] / (
        test_data["correct_steps"] + test_data["incorrect_steps"]
    )

    print("Adjusting bar order...")
    test_data["test.cond"] = sort_cond(test_data, bar_order)

    print("Computing statistics by agent...")
    by_test_cond = _compute_agent_stats(test_data, results_dir)

    print("Creating bar charts by agent...")
    agent_bar_charts(by_test_cond, results_dir, chick_data, color_bars)

    print("Computing statistics by imprinting condition...")
    by_imp_cond = stats_by_imprint_cond(by_test_cond, results_dir)

    print("Creating bar charts by imprinting condition...")
    imprint_cond_bar_charts(
        by_imp_cond, by_test_cond, results_dir, chick_data, color_bars
    )

    print("Computing statistics across all imprinting conditions...")
    across_imp_cond = stats_overall(data=by_test_cond, results_dir=results_dir)

    print("Creating bar chart for all imprinting conditions...")
    all_cond_bar_chart(
        by_test_cond, across_imp_cond, results_dir, chick_data, color_bars
    )
