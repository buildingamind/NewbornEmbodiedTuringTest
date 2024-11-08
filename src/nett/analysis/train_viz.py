import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_viz(data_loc, results_wd, ep_bucket_size, num_episodes):
    """
    Analyze and visualize training data from the Newborn Embodied Turing Tests.

    Parameters:
        data_loc (str): Path to the merged R data CSV file.
        results_wd (str): Directory to save the resulting visualizations.
        ep_bucket_size (int): Number of episodes per x-axis group.
        num_episodes (int): Total number of episodes to include.
    """

    # Load data and filter episodes
    train_data = pd.read_csv(data_loc / "train_results.csv", skipinitialspace=True)
    print(f"Collating data for {num_episodes} training episodes...")
    train_data = train_data[train_data['Episode'] < num_episodes]

    # Clean 'correct.monitor' and compute correct/incorrect steps
    train_data['correct.monitor'] = train_data['correct.monitor'].str.strip()
    condition = train_data['correct.monitor'] == 'left'
    train_data['correct_steps'] = np.where(condition, train_data['left_steps'], train_data['right_steps'])
    train_data['incorrect_steps'] = np.where(condition, train_data['right_steps'], train_data['left_steps'])
    train_data['percent_correct'] = train_data['correct_steps'] / (train_data['correct_steps'] + train_data['incorrect_steps'])

    # Create episode blocks
    train_data['episode_block'] = (train_data['Episode'] // ep_bucket_size) + 1

    # Group data and calculate summary statistics
    summary = train_data.groupby(['imprint.cond', 'agent', 'episode_block'])['percent_correct'].agg(
        avgs='mean', sd='std', count='size').reset_index()
    summary['se'] = summary['sd'] / np.sqrt(summary['count'])

    # Plotting
    print("Plotting training data...")
    for cond in summary['imprint.cond'].unique():
        data = summary[summary['imprint.cond'] == cond]
        plt.figure(figsize=(6, 6))
        sns.lineplot(data=data, x='episode_block', y='avgs', hue='agent', palette='bright', legend=False)
        sns.despine(top=True, right=True)
        plt.axhline(y=0.5, linestyle='--', color='gray')
        plt.xlabel(f"Groups of {ep_bucket_size} Episodes", fontsize=14)
        plt.ylabel("Average Time with Imprinted Object", fontsize=14)
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.1), ['{:.0%}'.format(x) for x in np.arange(0, 1.1, 0.1)])
        max_block = data['episode_block'].max()
        plt.xlim(1, max_block)
        plt.xticks(np.arange(1, max_block + 1, 1))
        plt.tight_layout()
        img_name = os.path.join(results_wd, f"{cond}_train.png")
        plt.savefig(img_name)
        plt.close()