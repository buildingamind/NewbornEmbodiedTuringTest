import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _plot_and_save_trajectory(episode_df, episode_id, output_path, arrow_freq):
    """
    Helper function to plot and save the trajectory for a single episode.

    Args:
        episode_df (pd.DataFrame): DataFrame containing data for a single episode.
        episode_id (int): The ID of the episode being plotted.
        output_path (str): The full path to save the output PNG file.
        arrow_freq (int): Frequency of steps at which to draw an arrow.
    """
    if episode_df.empty:
        print(f"  - Skipping Episode {episode_id}: No data.")
        return

    # Ensure data is sorted by step for correct trajectory plotting
    episode_df = episode_df.sort_values('Step').reset_index()

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- Plot Trajectory Line ---
    ax.plot(
        episode_df['agent.x'],
        episode_df['agent.z'],
        color='lightgray',
        linestyle='-',
        linewidth=1.5,
        zorder=1,
        label='Agent Path'
    )

    # --- Prepare for Arrows ---
    # Filter for the steps where we'll draw an arrow
    arrow_df = episode_df[episode_df['Step'] % arrow_freq == 0]

    if not arrow_df.empty:
        # Setup colormap: progressing from red to purple through a rainbow spectrum
        # The color is based on the step number within the episode.
        min_step = episode_df['Step'].min()
        max_step = episode_df['Step'].max()
        # Handle case where there's only one step
        if min_step == max_step:
            norm = mcolors.Normalize(vmin=min_step, vmax=min_step + 1)
        else:
            norm = mcolors.Normalize(vmin=min_step, vmax=max_step)
        cmap = plt.get_cmap('rainbow')

        # Determine a dynamic arrow size based on the plot's data range
        x_range = episode_df['agent.x'].max() - episode_df['agent.x'].min()
        z_range = episode_df['agent.z'].max() - episode_df['agent.z'].min()
        # Avoid division by zero if range is 0
        max_range = max(x_range, z_range, 1) 
        base_arrow_length = max_range * 0.04  # 4% of the max range
        head_width = base_arrow_length * 0.75
        head_length = base_arrow_length

        # --- Draw Arrows ---
        for _, row in arrow_df.iterrows():
            x, z = row['agent.x'], row['agent.z']
            agent_angle = row['agent.angle']
            head_angle = row['head.angle']
            step = row['Step']

            # --- Calculate Squish Factor and Arrow Dimensions ---
            # Based on user feedback. The squish factor determines the arrow's length.
            # A factor of 1.0 means no squish (at head_angle=0).
            # The length decreases as abs(head_angle) increases.
            # Using a divisor of 40.0 as requested.
            squish_factor = max(0, 1.0 - (abs(head_angle) / 40.0))

            # The total length of the arrow (body + head) is scaled by the squish factor.
            current_arrow_length = base_arrow_length * squish_factor
            
            # To create a "squished" look, the head's length is also scaled, but its
            # width is kept constant. This makes the arrow shorter and proportionally fatter.
            current_head_length = head_length * squish_factor
            
            # Ensure the head length is not larger than the arrow's total length.
            # This can happen if squish_factor is very small.
            current_head_length = min(current_head_length, current_arrow_length)

            # Convert agent angle to matplotlib angle
            # Data: 0 degrees is positive Y. Matplotlib: 0 degrees is positive X.
            # Transformation: plot_angle = 90 - agent_angle
            plot_angle_rad = np.deg2rad(90 - agent_angle)

            # Calculate arrow vector components (the vector from tail to tip)
            dx = current_arrow_length * np.cos(plot_angle_rad)
            dy = current_arrow_length * np.sin(plot_angle_rad)
            
            # Get color for the current step
            color = cmap(norm(step))

            # Draw the arrow with the new, scaled dimensions
            ax.arrow(
                x, z, dx, dy,
                head_width=head_width,          # Original width for "fat" look
                head_length=current_head_length, # Scaled length
                fc=color,
                ec=color,
                length_includes_head=True,
                zorder=2
            )

    # --- Final Plot Styling ---
    ax.set_title(f'Agent Trajectory - Episode {int(episode_id)}', fontsize=16)
    ax.set_xlabel('Agent X Position', fontsize=12)
    ax.set_ylabel('Agent Z Position', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    # ax.set_aspect('equal', adjustable='box') # Crucial for correct angle representation

    # Set fixed axis limits as requested
    ax.set_xlim(-33.15, 33.15)
    ax.set_ylim(-21, 21)

    ax.set_aspect('equal', adjustable='box') # Crucial for correct angle representation
    fig.tight_layout()

    # --- Save Figure ---
    try:
        plt.savefig(output_path, dpi=150)
        print(f"  - Saved trajectory for Episode {int(episode_id)} to {output_path}")
    except Exception as e:
        print(f"  - FAILED to save plot for Episode {int(episode_id)}. Error: {e}")
    finally:
        plt.close(fig) # Close the figure to free up memory


def map_trajectories(input_dir, output_dir, arrow_freq=100):
    """
    Processes experiment logs to generate and save agent trajectory plots.

    This function walks through a directory structure of:
    <IMPRINT CONDITION>/brain_<BRAIN_NUM>/logs/

    It reads 'train' and 'test' CSVs, and for each episode, it plots the
    agent's trajectory (x vs z), adding colored and shaped arrows to indicate
    the agent's direction and head angle at regular step intervals.

    Args:
        input_dir (str): The root directory containing the experiment data.
        output_dir (str): The directory where generated plots will be saved.
                          It will be created if it doesn't exist.
        arrow_freq (int, optional): The frequency of steps at which to draw
                                    a directional arrow. Defaults to 100.
    """
    print(f"Starting trajectory mapping from '{input_dir}' to '{output_dir}'...")
    
    # --- Validate Input and Create Output Directory ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{output_dir}'. Reason: {e}")
        return

    # Define the columns required for plotting
    required_cols = [
        'Episode', 'Step', 'agent.x', 'agent.z', 'agent.angle', 'head.angle'
    ]

    # --- Walk Through Input Directory Structure ---
    for imprint_cond in os.listdir(input_dir):
        imprint_path = os.path.join(input_dir, imprint_cond)
        if not os.path.isdir(imprint_path):
            continue

        print(f"\nProcessing Condition: {imprint_cond}")
        for brain_dir in os.listdir(imprint_path):
            if not brain_dir.startswith('brain_') or not os.path.isdir(os.path.join(imprint_path, brain_dir)):
                continue
            
            print(f"- Processing Brain: {brain_dir}")
            logs_path = os.path.join(imprint_path, brain_dir, 'logs')

            if not os.path.isdir(logs_path):
                print(f"  - No 'logs' directory found for {brain_dir}. Skipping.")
                continue

            # --- Process CSV files in the logs directory ---
            for filename in os.listdir(logs_path):
                if not filename.endswith('.csv'):
                    continue

                subfolder_type = None
                if filename.lower().startswith('train'):
                    subfolder_type = 'train'
                elif filename.lower().startswith('test'):
                    subfolder_type = 'test'
                else:
                    continue # Skip files that aren't train or test logs

                print(f"  - Found {subfolder_type} file: {filename}")
                
                # --- Create Final Output Directory ---
                final_output_dir = os.path.join(output_dir, imprint_cond, brain_dir, subfolder_type)
                os.makedirs(final_output_dir, exist_ok=True)
                
                # --- Read and Clean CSV Data ---
                csv_path = os.path.join(logs_path, filename)
                try:
                    df = pd.read_csv(csv_path)
                    # Drop rows that are missing any of the essential values
                    df.dropna(subset=required_cols, inplace=True)
                    # Ensure numeric types, coercing errors to NaN, then dropping again
                    for col in required_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.dropna(subset=required_cols, inplace=True)
                except Exception as e:
                    print(f"    - ERROR: Could not read or process {filename}. Error: {e}")
                    continue
                
                if df.empty:
                    print(f"    - No valid data found in {filename} after cleaning. Skipping.")
                    continue
                    
                # --- Generate Plot for Each Episode ---
                unique_episodes = df['Episode'].unique()
                print(f"    - Found {len(unique_episodes)} unique episodes. Generating plots...")
                for episode_id in unique_episodes:
                    episode_df = df[df['Episode'] == episode_id]
                    plot_filename = f"{int(episode_id)}.png"
                    output_file_path = os.path.join(final_output_dir, plot_filename)
                    
                    _plot_and_save_trajectory(
                        episode_df=episode_df,
                        episode_id=episode_id,
                        output_path=output_file_path,
                        arrow_freq=arrow_freq
                    )

    print("\nTrajectory mapping complete.")
