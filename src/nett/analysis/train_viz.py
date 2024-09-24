import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_viz(data_loc, results_wd, ep_bucket_size, num_episodes):
    """
    A function to analyze and visualize training data from the Newborn Embodied Turing Tests.

    Parameters:
    - data_loc (str): Full filename (including working directory) of the merged R data.
    - results_wd (str): Directory to save the resulting visualizations.
    - ep_bucket_size (int): How many episodes to group the x-axis by.
    - num_episodes (int): How many episodes should be included.
    """

    # Load data
    result = pd.read_csv(data_loc, skipinitialspace=True)
    train_data = result['train_data']

    # Remove test_data if present
    if 'test_data' in result:
        del result['test_data']

    print(f"Collating data for {num_episodes} training episodes...")
    train_data_fixed = train_data.copy()
    train_data_fixed = train_data_fixed[train_data_fixed['Episode'] < num_episodes]

    # Create variables for correct/incorrect calculations
    train_data_fixed['correct_steps'] = np.where(
        train_data_fixed['correct.monitor'] == " left",
        train_data_fixed['left_steps'],
        train_data_fixed['right_steps']
    )
    train_data_fixed['incorrect_steps'] = np.where(
        train_data_fixed['correct.monitor'] == " left",
        train_data_fixed['right_steps'],
        train_data_fixed['left_steps']
    )
    train_data_fixed['percent_correct'] = train_data_fixed['correct_steps'] / (
        train_data_fixed['correct_steps'] + train_data_fixed['incorrect_steps']
    )

    # Summarize data by condition, agent, and episode bucket for graphing
    train_data_fixed['episode_block'] = (train_data_fixed['Episode'] // ep_bucket_size) + 1
    grouped = train_data_fixed.groupby(['imprint.cond', 'agent', 'episode_block'])

    summary = grouped['percent_correct'].agg(['mean', 'std', 'size']).reset_index()
    summary.rename(columns={'mean': 'avgs', 'std': 'sd', 'size': 'count'}, inplace=True)
    summary['se'] = summary['sd'] / np.sqrt(summary['count'])

    # Convert numerical variables into correct type
    summary['episode_block'] = summary['episode_block'].astype(float)
    summary['agent'] = summary['agent'].astype(float)

    # Plot line graphs by imprinting condition
    print("Plotting training data...")
    for cond in summary['imprint.cond'].unique():
        data = summary[summary['imprint.cond'] == cond]

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x='episode_block', y='avgs', hue='agent', legend=False)
        plt.axhline(y=0.5, linestyle='--', color='gray')
        plt.xlabel(f"Groups of {ep_bucket_size} Episodes")
        plt.ylabel("Average Time with Imprinted Object")
        plt.ylim(0, 1)
        plt.yticks(
            np.arange(0, 1.1, 0.1),
            ['{:.0%}'.format(x) for x in np.arange(0, 1.1, 0.1)]
        )
        plt.xlim(0, num_episodes / ep_bucket_size)
        plt.xticks(np.arange(0, (num_episodes / ep_bucket_size) + 1, 1))
        plt.title(cond)
        plt.tight_layout()

        img_name = os.path.join(results_wd, f"{cond}_train.png")
        plt.savefig(img_name)
        plt.close()
