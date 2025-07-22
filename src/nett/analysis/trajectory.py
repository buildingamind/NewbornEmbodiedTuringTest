import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _plot_and_save_trajectory(
    episode_df, episode_id, output_path, arrow_freq, path_arrow_freq, asterisk_side=None
):
    """
    Helper function to plot and save the trajectory for a single episode.

    Args:
        episode_df (pd.DataFrame): DataFrame containing data for a single episode.
        episode_id (int): The ID of the episode being plotted.
        output_path (str): The full path to save the output PNG file.
        arrow_freq (int): Frequency of steps at which to draw an arrow.
        path_arrow_freq (int): Frequency of steps at which to draw path direction arrows.
        asterisk_side (str, optional): Side to place the asterisk ('left' or 'right'). Defaults to None.
    """
    if episode_df.empty:
        print(f"  - Skipping Episode {episode_id}: No data.")
        return

    # Ensure data is sorted by step for correct trajectory plotting
    episode_df = episode_df.sort_values("Step").reset_index()

    # --- Get Episode Step Range ---
    min_step = episode_df["Step"].min()
    max_step = episode_df["Step"].max()

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- Plot Trajectory Line ---
    ax.plot(
        episode_df["agent.x"],
        episode_df["agent.z"],
        color="lightgray",
        linestyle="-",
        linewidth=1.5,
        zorder=1,
        label="Agent Path",
    )

    # --- Add small arrows indicating path direction ---
    path_arrow_steps = np.arange(
        path_arrow_freq + min_step, len(episode_df), path_arrow_freq
    )
    for step_index in path_arrow_steps:
        # Ensure we have a previous step to calculate direction from
        if step_index >= 1:
            # Use .iloc for integer-location based indexing
            prev_step_df = episode_df.iloc[int(step_index) - 1]
            curr_step_df = episode_df.iloc[int(step_index)]

            x_start = prev_step_df["agent.x"]
            z_start = prev_step_df["agent.z"]

            dx = curr_step_df["agent.x"] - x_start
            dz = curr_step_df["agent.z"] - z_start

            # Don't draw an arrow if there was no movement
            if dx == 0 and dz == 0:
                continue

            ax.arrow(
                x_start,
                z_start,
                dx,
                dz,
                head_width=0.4,  # "very small"
                head_length=0.4,  # "very small"
                fc="lightgrey",
                ec="lightgrey",
                length_includes_head=True,
                zorder=1,  # Same z-order as the path line
            )

    # --- Prepare for Arrows ---
    # Filter for the steps where we'll draw an arrow
    arrow_df = episode_df[
        ((episode_df["Step"] - min_step) % arrow_freq == 0)
        | (episode_df["Step"] == max_step)
    ]

    # --- Adjust subplot for whitespace ---
    fig.subplots_adjust(left=0.1, right=0.9)

    # --- Add asterisk based on side (in the new whitespace) ---
    if asterisk_side:
        x_pos = 0.05 if asterisk_side == "left" else 0.95
        ha = asterisk_side
        fig.text(  # Use fig.text for figure-relative coordinates
            x_pos,
            0.5,
            "*",
            fontsize=40,
            color="red",
            ha=ha,
            va="center",
        )

    # --- Final Plot Styling ---
    ax.set_title(f"Agent Trajectory - Episode {int(episode_id)}", fontsize=16)
    ax.set_xlabel("Agent X Position", fontsize=12)
    ax.set_ylabel("Agent Z Position", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.set_aspect('equal', adjustable='box') # Crucial for correct angle representation

    # Set fixed axis limits as requested
    ax.set_xlim(-33.15, 33.15)
    ax.set_ylim(-21, 21)

    ax.set_aspect("equal", adjustable="box")  # Crucial for correct angle representation
    fig.tight_layout()

    # --- Draw Arrows (after setting axis limits for correct scaling) ---
    if not arrow_df.empty:
        # Setup colormap: progressing from red to purple through a rainbow spectrum
        # The color is based on the step number within the episode.
        min_step = episode_df["Step"].min()
        max_step = episode_df["Step"].max()
        # Handle case where there's only one step
        if min_step == max_step:
            norm = mcolors.Normalize(vmin=min_step, vmax=min_step + 1)
        else:
            norm = mcolors.Normalize(vmin=min_step, vmax=max_step)
        cmap = plt.get_cmap("gist_rainbow")

        # Determine a dynamic arrow size based on the plot's axis range
        x_axis_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        z_axis_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        max_range = max(x_axis_range, z_axis_range)
        base_arrow_length = max_range * 0.04  # 4% of the max range
        head_width = base_arrow_length * 0.75
        head_length = base_arrow_length

        # --- Draw Arrows ---
        for _, row in arrow_df.iterrows():
            x, z = row["agent.x"], row["agent.z"]
            agent_angle = row["agent.angle"]
            head_angle = (
                (row["head.angle"] + 20) % 360
            ) - 20  # Normalize to [-20, 20]

            step = row["Step"]

            # --- Calculate Squish Factor and Arrow Dimensions ---
            squish_factor = 1.0 - (abs(head_angle) / 40.0)

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
                x,
                z,
                dx,
                dy,
                head_width=head_width,  # Original width for "fat" look
                head_length=current_head_length,  # Scaled length
                fc=color,
                ec=color,
                length_includes_head=True,
                zorder=2,
            )

    # --- Save Figure ---
    try:
        plt.savefig(output_path, dpi=150)
        print(f"  - Saved trajectory for Episode {int(episode_id)} to {output_path}")
    except Exception as e:
        print(f"  - FAILED to save plot for Episode {int(episode_id)}. Error: {e}")
    finally:
        plt.close(fig)  # Close the figure to free up memory


def map_trajectories(
    input_dir,
    output_dir,
    arrow_freq=100,
    path_arrow_freq=10,
    episode_range={"train": None, "test": None},
):
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
        path_arrow_freq (int, optional): The frequency of steps at which to draw
                                        path direction arrows. Defaults to 10.
        episode_range (dict, optional): Range of episodes to plot for train and test.
                                       Format: {"train": "start:stop:step", "test": "start:stop:step"}
                              If None, all episodes are plotted. Defaults to None.
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
        "Episode",
        "Step",
        "agent.x",
        "agent.z",
        "agent.angle",
        "head.angle",
        "left.monitor",
        "correct.monitor",
    ]

    numeric_cols = [
        "Episode",
        "Step",
        "agent.x",
        "agent.z",
        "agent.angle",
        "head.angle",
    ]

    # --- Walk Through Input Directory Structure ---
    for imprint_cond in os.listdir(input_dir):
        imprint_path = os.path.join(input_dir, imprint_cond)
        if not os.path.isdir(imprint_path):
            continue

        print(f"\nProcessing Condition: {imprint_cond}")
        for brain_dir in os.listdir(imprint_path):
            if not brain_dir.startswith("brain_") or not os.path.isdir(
                os.path.join(imprint_path, brain_dir)
            ):
                continue

            print(f"- Processing Brain: {brain_dir}")
            logs_path = os.path.join(imprint_path, brain_dir, "logs")

            if not os.path.isdir(logs_path):
                print(f"  - No 'logs' directory found for {brain_dir}. Skipping.")
                continue

            # --- Process CSV files in the logs directory ---
            for filename in os.listdir(logs_path):
                if not filename.endswith(".csv"):
                    continue

                subfolder_type = None
                if filename.lower().startswith("train"):
                    subfolder_type = "train"
                elif filename.lower().startswith("test"):
                    subfolder_type = "test"
                else:
                    continue  # Skip files that aren't train or test logs

                print(f"  - Found {subfolder_type} file: {filename}")

                # --- Create Final Output Directory ---
                final_output_dir = os.path.join(
                    output_dir, imprint_cond, brain_dir, subfolder_type
                )
                os.makedirs(final_output_dir, exist_ok=True)

                # --- Read and Clean CSV Data ---
                csv_path = os.path.join(logs_path, filename)
                try:
                    df = pd.read_csv(csv_path, skipinitialspace=True)
                    # Trim whitespace from headers and all string columns
                    # df.columns = df.columns.str.strip()
                    # for col in df.select_dtypes(["object"]).columns:
                    #     df[col] = df[col].str.strip()
                    # Drop rows that are missing any of the essential values
                    df.dropna(subset=required_cols, inplace=True)
                    # Ensure numeric types, coercing errors to NaN, then dropping again
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df.dropna(subset=numeric_cols, inplace=True)
                except Exception as e:
                    print(
                        f"    - ERROR: Could not read or process {filename}. Error: {e}"
                    )
                    continue

                if df.empty:
                    print(
                        f"    - No valid data found in {filename} after cleaning. Skipping."
                    )
                    continue

                # --- Generate Plot for Each Episode ---
                unique_episodes = df["Episode"].unique()

                # Filter episodes based on episode_range parameter
                if episode_range[subfolder_type] is not None:
                    try:
                        parts = episode_range[subfolder_type].split(":")
                        if len(parts) == 3:
                            start, stop, step = map(int, parts)
                            unique_episodes = [
                                ep
                                for ep in range(start, stop, step)
                                if ep in unique_episodes
                            ]
                        else:
                            print(
                                f"    - WARNING: Invalid episode_range format '{episode_range[subfolder_type]}'. Expected 'start:stop:step'. Using all episodes."
                            )
                    except ValueError:
                        print(
                            f"    - WARNING: Invalid episode_range format '{episode_range[subfolder_type]}'. Expected numeric values. Using all episodes."
                        )

                print(
                    f"    - Found {len(unique_episodes)} episodes to plot{' (filtered by episode_range)' if episode_range[subfolder_type] else ''}..."
                )
                for episode_id in unique_episodes:
                    episode_df = df[df["Episode"] == episode_id]

                    # Determine asterisk side
                    asterisk_side = None  # Default
                    if not episode_df.empty:
                        # The first row is sufficient to determine the side.
                        first_row = episode_df.iloc[0]
                        asterisk_side = first_row["correct.monitor"]

                    plot_filename = f"{int(episode_id)}.png"
                    output_file_path = os.path.join(final_output_dir, plot_filename)

                    _plot_and_save_trajectory(
                        episode_df=episode_df,
                        episode_id=episode_id,
                        output_path=output_file_path,
                        arrow_freq=arrow_freq,
                        path_arrow_freq=path_arrow_freq,
                        asterisk_side=asterisk_side,
                    )

    print("\nTrajectory mapping complete.")
