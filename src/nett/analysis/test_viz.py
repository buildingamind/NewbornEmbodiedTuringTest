from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D

CUSTOM_PALETTE = [
    "#3F8CB7", "#FCEF88", "#5D5797", "#62AC6B", "#B74779",
    "#2C4E98", "#CCCCE7", "#08625B", "#D15056", "#F2A541", "#FFC0CB"
]
CHICK_RED = "#AF264A"

def compute_stats(group, column='percent_correct', mu=0.5):
    data = group[column]
    avgs = data.mean()
    sd = data.std()
    count = data.count()
    se = sd / np.sqrt(count) if count > 0 else np.nan
    tval, pval = stats.ttest_1samp(data.dropna(), mu) if count > 0 else (np.nan, np.nan)
    df = count - 1
    cohensd = (avgs - mu) / sd if sd != 0 else np.nan
    return pd.Series({
        'avgs': avgs, 'sd': sd, 'count': count, 'tval': tval,
        'df': df, 'pval': pval, 'se': se, 'cohensd': cohensd
    })

def make_bar_charts(data, dots, y_col, error_min_col, error_max_col,
                    img_name, chick_data, color_bars):
    plt.figure(figsize=(6, 6))
    sns.set_style("white")
    # sns.set_style("whitegrid")
    ax = plt.gca()
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)

    y = data[y_col]
    yerr = [data[y_col] - data[error_min_col],
            data[error_max_col] - data[y_col]]

    x_categories = data['test.cond'].unique()

    data['test.cond'] = pd.Categorical(data['test.cond'], categories=x_categories, ordered=True)

    if color_bars:
        color_mapping = dict(zip(x_categories, CUSTOM_PALETTE))
        colors = data['test.cond'].map(color_mapping)
    else:
        colors = 'gray45'

    x_pos = np.arange(len(x_categories))
    ax.bar(x_pos, y, yerr=yerr, color=colors, capsize=14,
           width=0.7, linewidth=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_categories, rotation=0, ha='center',
                       fontsize=7.5, fontweight='bold')

    chick_data_filtered = chick_data[chick_data['test.cond'].isin(x_categories)]
    if not chick_data_filtered.empty:
        chick_data_filtered['test.cond'] = pd.Categorical(
            chick_data_filtered['test.cond'], categories=x_categories, ordered=True)
        chick_x_pos = chick_data_filtered['test.cond'].cat.codes - 1
        print(f'Chick x_pos: {chick_x_pos}')
        print(f'Chick avg: {chick_data_filtered["avg"]}')
        print(f'Chick avg_dev: {chick_data_filtered["avg_dev"]}')
        shift = 0
        for xi, yi, yerr, i in zip(chick_x_pos, chick_data_filtered['avg'],
                                chick_data_filtered['avg_dev'], range(len(chick_x_pos))):
            if x_categories[i] == 'Rest':
                shift = 1
            rect = plt.Rectangle((xi + shift - 0.35, yi - yerr), 0.7, 2*yerr,
                                       color=CHICK_RED, alpha=0.2)
            ax.add_patch(rect)
            # Calculate the y-coordinate for the horizontal line
            y_center = rect.get_y() + rect.get_height() / 2
            # Add the horizontal line
            line = Line2D([rect.get_x(), rect.get_x() + rect.get_width()], [y_center, y_center], c=CHICK_RED)
            ax.add_line(line)

    if dots is not None and 'test.cond' in dots.columns:
        dots['test.cond'] = pd.Categorical(
            dots['test.cond'], categories=x_categories, ordered=True)
        dot_x_pos = dots['test.cond'].cat.codes
        sns.stripplot(x=dot_x_pos, y=dots['avgs'], ax=ax, color='black',
                      jitter=0.3, size=3)

    ax.axhline(0.5, linestyle='--', color='grey')
    ax.set_xlabel("Test Condition", fontweight='bold', fontsize=14)
    ax.set_ylabel("Percent Correct", fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels(
        ['{:.0%}'.format(t) for t in np.linspace(0, 1, 11)],
        fontsize=7.5, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()

def test_viz(data_loc, chick_file, results_wd,
             bar_order="default", color_bars=False):
    data_loc = Path(data_loc)
    chick_file = Path(chick_file)
    results_wd = Path(results_wd)

    if not data_loc.exists():
        print(f"Data file not found at: {data_loc}")
        return
    if not chick_file.exists():
        print(f"Chick data file not found at: {chick_file}")
        return
    results_wd.mkdir(parents=True, exist_ok=True)

    chick_data = pd.read_csv(chick_file)
    test_data = pd.read_csv(data_loc / "test_results.csv")

    print("Collating data for test trials...")
    test_data['correct_steps'] = np.where(
        test_data['correct.monitor'] == 'left',
        test_data['left_steps'], test_data['right_steps'])
    test_data['incorrect_steps'] = np.where(
        test_data['correct.monitor'] == 'right',
        test_data['left_steps'], test_data['right_steps'])
    test_data['percent_correct'] = test_data['correct_steps'] / (
        test_data['correct_steps'] + test_data['incorrect_steps'])

    print("Adjusting bar order...")
    if bar_order == "desc":
        order = test_data.groupby('test.cond')['percent_correct']\
            .mean().sort_values(ascending=False).index.tolist()
    elif bar_order == "asc":
        order = test_data.groupby('test.cond')['percent_correct']\
            .mean().sort_values().index.tolist()
    elif bar_order != "default":
        order = [x.strip() for x in bar_order.split(',')]
    else:
        order = test_data['test.cond'].unique().tolist()
    test_data['test.cond'] = pd.Categorical(
        test_data['test.cond'], categories=order, ordered=True)

    print("Computing statistics by agent...")
    grouped = test_data.groupby(['imprint.cond', 'agent', 'test.cond'])
    by_test_cond = grouped.apply(compute_stats).reset_index()
    by_test_cond['imp_agent'] = by_test_cond['imprint.cond'] + '_' + by_test_cond['agent'].astype(str)
    by_test_cond.to_csv(results_wd / "stats_by_agent.csv", index=False)

    pd.options.mode.chained_assignment = None

    print("Creating bar charts by agent...")
    for imp_agent in by_test_cond['imp_agent'].unique():
        bar_data = by_test_cond[by_test_cond['imp_agent'] == imp_agent]
        bar_data['error_min'] = bar_data['avgs'] - bar_data['se']
        bar_data['error_max'] = bar_data['avgs'] + bar_data['se']
        img_name = results_wd / f"{imp_agent}_test.png"
        make_bar_charts(
            data=bar_data, dots=None, y_col='avgs',
            error_min_col='error_min', error_max_col='error_max',
            img_name=img_name, chick_data=chick_data, color_bars=color_bars
        )

    print("Computing statistics by imprinting condition...")
    grouped_imp = by_test_cond.groupby(['imprint.cond', 'test.cond'])
    by_imp_cond = grouped_imp.apply(
        lambda g: compute_stats(g, column='avgs')
    ).reset_index()
    by_imp_cond.to_csv(results_wd / "stats_by_imp_cond.csv", index=False)

    print("Creating bar charts by imprinting condition...")
    for imp_cond in by_imp_cond['imprint.cond'].unique():
        bar_data = by_imp_cond[
            (by_imp_cond['imprint.cond'] == imp_cond) &
            (by_imp_cond['test.cond'] != "Rest")
        ]
        dot_data = by_test_cond[
            (by_test_cond['imprint.cond'] == imp_cond) &
            (by_test_cond['test.cond'] != "Rest")
        ]
        bar_data['error_min'] = bar_data['avgs'] - bar_data['se']
        bar_data['error_max'] = bar_data['avgs'] + bar_data['se']
        img_name = results_wd / f"{imp_cond}_test.png"
        make_bar_charts(
            data=bar_data, dots=dot_data, y_col='avgs',
            error_min_col='error_min', error_max_col='error_max',
            img_name=img_name, chick_data=chick_data, color_bars=color_bars
        )

    print("Computing statistics across all imprinting conditions...")
    across_imp_cond = by_test_cond[
        by_test_cond['test.cond'] != "Rest"
    ].groupby('test.cond').apply(
        lambda g: compute_stats(g, column='avgs')
    ).reset_index()
    across_imp_cond.to_csv(
        results_wd / "stats_across_all_agents.csv", index=False
    )

    print("Creating bar chart for all imprinting conditions...")
    across_imp_cond['error_min'] = across_imp_cond['avgs'] - across_imp_cond['se']
    across_imp_cond['error_max'] = across_imp_cond['avgs'] + across_imp_cond['se']
    dot_data = by_test_cond[by_test_cond['test.cond'] != "Rest"]
    img_name = results_wd / 'all_imprinting_conds_test.png'
    make_bar_charts(
        data=across_imp_cond, dots=dot_data, y_col='avgs',
        error_min_col='error_min', error_max_col='error_max',
        img_name=img_name, chick_data=chick_data, color_bars=color_bars
    )