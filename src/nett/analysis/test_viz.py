# NETT_test_viz.py

# Before running this script, you need to run merge_csvs to merge all of the agents'
# output into a single, standardized format dataframe for training and test data

# Variables --------------------------------------------------------------------

# Read in the user-specified variables:
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Bar Chart Function -----------------------------------------------------------
def make_bar_charts(data, dots, aes_y, error_min, error_max, img_name):
  global chick_data
  global color_bars
  custom_palette = ["#3F8CB7", "#FCEF88", "#5D5797", "#62AC6B", "#B74779", "#2C4E98", "#CCCCE7", "#08625B", "#D15056", "#F2A541", "#FFC0CB"]
  chickred = "#AF264A"
  
  plt.figure(figsize=(6,6))
  sns.set_style("whitegrid")
  ax = plt.gca()
  
  # Prepare data for plotting
  y = data[aes_y]
  yerr_lower = y - data[error_min]
  yerr_upper = data[error_max] - y
  yerr = [yerr_lower.values, yerr_upper.values]
  
  # Determine order of categories
  if hasattr(data['test.cond'], 'cat'):
      x_categories = data['test.cond'].cat.categories
  else:
      x_categories = data['test.cond'].unique()
      data['test.cond'] = pd.Categorical(data['test.cond'], categories=x_categories, ordered=True)
  
  # Map 'test.cond' to colors
  if color_bars:
      color_mapping = dict(zip(x_categories, custom_palette))
      colors = data['test.cond'].map(color_mapping)
  else:
      colors = 'gray45'
  
  # Plot bars with error bars
  positions = np.arange(len(x_categories))
  x_pos = data['test.cond'].cat.codes
  ax.bar(x_pos, y, yerr=yerr, color=colors, capsize=3, edgecolor='black', width=0.7)
  
  # Set x-axis labels
  ax.set_xticks(positions)
  ax.set_xticklabels(x_categories, rotation=45, ha='right', fontsize=7.5, fontweight='bold')
  
  # Plot chick_data error bars
  # Ensure 'chick_data' is filtered to the same 'test.cond' as 'data'
  chick_data_filtered = chick_data[chick_data['test.cond'].isin(x_categories)]
  if not chick_data_filtered.empty:
      chick_data_filtered['test.cond'] = pd.Categorical(chick_data_filtered['test.cond'], categories=x_categories, ordered=True)
      chick_x_pos = chick_data_filtered['test.cond'].cat.codes
      chick_y = chick_data_filtered['avg']
      chick_yerr = chick_data_filtered['avg_dev']
      ax.errorbar(chick_x_pos, chick_y, yerr=chick_yerr, fmt='o', color=chickred, capsize=3, label='Chick Performance')
      # Add crossbars (filled rectangles)
      for xi, yi, yerr in zip(chick_x_pos, chick_y, chick_yerr):
          ax.add_patch(plt.Rectangle((xi - 0.35, yi - yerr), 0.7, 2*yerr, color=chickred, alpha=0.2))
  
  # Plot dots if provided
  if dots is not None:
      if 'test.cond' in dots.columns:
          dots['test.cond'] = pd.Categorical(dots['test.cond'], categories=x_categories, ordered=True)
          dot_x_pos = dots['test.cond'].cat.codes
          sns.stripplot(x=dot_x_pos, y=dots['avgs'], ax=ax, color='black', jitter=0.3, size=3)
  
  # Add horizontal line at y=0.5
  ax.axhline(0.5, linestyle='--', color='grey')
  
  # Set labels and limits
  ax.set_xlabel("Test Condition", fontweight='bold')
  ax.set_ylabel("Percent Correct", fontweight='bold')
  ax.set_ylim(0,1)
  ax.set_yticks(np.linspace(0,1,11))
  ax.set_yticklabels(['{:.0%}'.format(t) for t in np.linspace(0,1,11)], fontsize=7.5, fontweight='bold')
  
  ax.legend()
  
  plt.tight_layout()
  plt.savefig(img_name)
  plt.close()

## Group data by test conditions
def compute_stats(group):
  percent_correct = group['percent_correct']
  avgs = percent_correct.mean()
  sd = percent_correct.std()
  count = percent_correct.count()
  se = sd / np.sqrt(count) if count > 0 else np.nan
  # t-test against mu=0.5
  try:
      tval, pval = stats.ttest_1samp(percent_correct, 0.5)
      df = count - 1
  except:
      tval = np.nan
      pval = np.nan
      df = np.nan
  cohensd = (avgs - 0.5) / sd if sd != 0 else np.nan
  return pd.Series({'avgs': avgs, 'sd': sd, 'count': count, 'tval': tval, 'df': df, 'pval': pval, 'se': se, 'cohensd': cohensd})

def compute_stats_by_imp(group):
  avgs_by_imp = group['avgs'].mean()
  sd = group['avgs'].std()
  count = group['avgs'].count()
  se = sd / np.sqrt(count) if count > 0 else np.nan
  # t-test against mu=0.5
  try:
      tval, pval = stats.ttest_1samp(group['avgs'], 0.5)
      df = count - 1
  except:
      tval = np.nan
      pval = np.nan
      df = np.nan
  cohensd = (avgs_by_imp - 0.5) / sd if sd != 0 else np.nan
  return pd.Series({'avgs_by_imp': avgs_by_imp, 'sd': sd, 'count': count, 'tval': tval, 'df': df, 'pval': pval, 'se': se, 'cohensd': cohensd})

def test_viz(data_loc: Path | str,
             chick_file: Path | str,
             results_wd: Path | str,
             bar_order: str = "default",
             color_bars: bool = False) -> None:

  # Set Up -----------------------------------------------------------------------

  if not isinstance(data_loc, Path):
      data_loc = Path(data_loc)
  if not isinstance(chick_file, Path):
      chick_file = Path(chick_file)
  if not isinstance(results_wd, Path):
      results_wd = Path(results_wd)
  if not data_loc.exists():
      print(f"Data file not found at: {data_loc}")
  if not chick_file.exists():
      print(f"Chick data file not found at: {chick_file}")
  if not results_wd.exists():
      results_wd.mkdir(parents=True, exist_ok=True)

  # Load the chick data
  chick_data = pd.read_csv(chick_file, skipinitialspace=True)

  # Load test data
  test_data = pd.read_csv(data_loc / "test_results.csv", skipinitialspace=True)

  print("Collating data for test trials...")
  # Code each episode correct/incorrect

  test_data['correct_steps'] = test_data['left_steps'].where(
    test_data['correct.monitor'] == 'left',
    test_data['right_steps']
  )

  test_data['incorrect_steps'] = test_data['left_steps'].where(
    test_data['correct.monitor'] == 'right',
    test_data['right_steps']
  )

  test_data['percent_correct'] = test_data['correct_steps'] / (test_data['correct_steps'] + test_data['incorrect_steps'])

  # Adjust bar order according to user input -------------------------------------
  print("Adjusting bar order...")
  # Create a variable to store the final order
  order = None
  if bar_order in ["default", "asc", "desc"]:
      order = bar_order
  else:
      order = [int(x.strip()) for x in bar_order.split(',')]

  # Conditionally reorder the dataframe based on user input
  if order is not None:
      if order == "desc":
          test_data = test_data.sort_values(by='percent_correct', ascending=False)
          unique_levels = test_data['test.cond'].unique()
          test_data['test.cond'] = pd.Categorical(test_data['test.cond'], categories=unique_levels, ordered=True)
      elif order == "asc":
          test_data = test_data.sort_values(by='percent_correct', ascending=True)
          unique_levels = test_data['test.cond'].unique()
          test_data['test.cond'] = pd.Categorical(test_data['test.cond'], categories=unique_levels, ordered=True)
      elif order != "default":
          current_order = test_data['test.cond'].astype('category').cat.categories.tolist()
          adjusted_order = [i - 1 for i in order]  # Adjust for zero-based indexing
          new_order = [current_order[i] for i in adjusted_order]
          test_data['test.cond'] = pd.Categorical(test_data['test.cond'], categories=new_order, ordered=True)
      # If order is "default", no need to change anything

  # Plot aesthetic settings ------------------------------------------------------
  custom_palette = ["#3F8CB7", "#FCEF88", "#5D5797", "#62AC6B", "#B74779", "#2C4E98", "#CCCCE7", "#08625B", "#D15056", "#F2A541", "#FFC0CB"]
  chickred = "#AF264A"

  print("Creating bar charts...")

  grouped = test_data.groupby(['imprint.cond', 'agent', 'test.cond'])
  by_test_cond = grouped.apply(compute_stats).reset_index()
  by_test_cond['imp_agent'] = by_test_cond['imprint.cond'] + '_' + by_test_cond['agent']

  by_test_cond.to_csv("stats_by_agent.csv", index=False)

  for i in by_test_cond['imp_agent'].unique():
      bar_data = by_test_cond[by_test_cond['imp_agent'] == i]
      bar_data['error_min'] = bar_data['avgs'] - bar_data['se']
      bar_data['error_max'] = bar_data['avgs'] + bar_data['se']
      img_name = f"{i}_test.png"
      make_bar_charts(data=bar_data,
                      dots=None,
                      aes_y='avgs',
                      error_min='error_min',
                      error_max='error_max',
                      img_name=img_name)

  # Plot by imprinting condition -------------------------------------------------
  ## Remove rest data once we start to group agents (for ease of presentation)
  print("Creating imprinting condition-level bar charts...")

  grouped_imp = by_test_cond.groupby(['imprint.cond', 'test.cond'])
  by_imp_cond = grouped_imp.apply(compute_stats_by_imp).reset_index()

  by_imp_cond.to_csv("stats_by_imp_cond.csv", index=False)

  for i in by_imp_cond['imprint.cond'].unique():
      bar_data = by_imp_cond[(by_imp_cond['imprint.cond'] == i) & (by_imp_cond['test.cond'] != "Rest")]
      dot_data = by_test_cond[(by_test_cond['imprint.cond'] == i) & (by_test_cond['test.cond'] != "Rest")]
      bar_data['error_min'] = bar_data['avgs_by_imp'] - bar_data['se']
      bar_data['error_max'] = bar_data['avgs_by_imp'] + bar_data['se']
      img_name = f"{i}_test.png"
      make_bar_charts(data=bar_data,
                      dots=dot_data,
                      aes_y='avgs_by_imp',
                      error_min='error_min',
                      error_max='error_max',
                      img_name=img_name)

  # Plot across all imprinting conditions ----------------------------------------
  print("Creating bar chart for all imprinting conditions...")
  across_imp_cond = by_test_cond[by_test_cond['test.cond'] != "Rest"].groupby('test.cond').apply(compute_stats_by_imp).reset_index()
  across_imp_cond.rename(columns={'avgs_by_imp': 'all_avgs'}, inplace=True)
  across_imp_cond['error_min'] = across_imp_cond['all_avgs'] - across_imp_cond['se']
  across_imp_cond['error_max'] = across_imp_cond['all_avgs'] + across_imp_cond['se']

  across_imp_cond.to_csv("stats_across_all_agents.csv", index=False)

  dot_data = by_test_cond[by_test_cond['test.cond'] != "Rest"]

  make_bar_charts(data=across_imp_cond,
                  dots=dot_data,
                  aes_y='all_avgs',
                  error_min='error_min',
                  error_max='error_max',
                  img_name='all_imprinting_conds_test.png')