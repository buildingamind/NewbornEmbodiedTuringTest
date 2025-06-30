from pathlib import Path
import pandas as pd

# Define x_limits and calculate BOUNDS of chamber
X_LIMITS: tuple[int, int] = (-10, 10)
# X_LIMITS: tuple[int, int] = (-30.15, 30.15)
ONE_THIRD: float = (X_LIMITS[1] - X_LIMITS[0]) / 3
BOUNDS: list[float] = [X_LIMITS[0] + ONE_THIRD, X_LIMITS[1] - ONE_THIRD]

# Define the grouping columns
GROUP_COLUMNS: list[str] = [
    "Episode",
    "left.monitor",
    "right.monitor",
    "correct.monitor",
    "experiment.phase",
    "imprint.cond",
    "test.cond",
]


def _read_data(filename: Path) -> pd.DataFrame:
    print(f"Reading {filename}")
    data: pd.DataFrame = pd.read_csv(
        filename, skipinitialspace=True, on_bad_lines="skip"
    ).fillna("NA")

    ###################################
    # Convert columns ["Episode", "Step", "agent.x", "agent.y", "agent.z"] to numeric types, coercing errors.
    # This handles cases where these columns might contain 'NA' strings (due to prior fillna)
    # or other non-numeric values.
    for column in ["Episode", "Step", "agent.x", "agent.y", "agent.z"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # Proceed with the logic only if 'Episode' and 'Step' columns exist
    # and have at least some non-NaN values after conversion.
    if (
        "Episode" in data.columns
        and data["Episode"].notna().any()
        and "Step" in data.columns
        and data["Step"].notna().any()
    ):

        # Filter for rows where Episode is 0
        # .loc is used for clarity and to avoid potential SettingWithCopyWarning on chained indexing
        episode_0_data = data.loc[data["Episode"] == 0]

        # Check if there's any data for Episode 0 and if it has any valid (non-NaN) Step values
        if not episode_0_data.empty and episode_0_data["Step"].notna().any():
            max_step_episode_0 = episode_0_data["Step"].max()

            # Ensure max_step_episode_0 is a valid number (not NaN).
            # This check is important if all "Step" values in episode_0_data were NaN.
            if pd.notna(max_step_episode_0):
                # Find the maximum episode number in the entire dataset.
                # .max() on a pandas Series with NaNs will ignore NaNs by default.
                max_episode = data["Episode"].max()

                # Proceed only if max_episode is a valid number and is greater than 0
                # (i.e., there are episodes beyond Episode 0).
                if pd.notna(max_episode) and max_episode > 0:
                    # Filter for rows belonging to the last episode
                    last_episode_data = data.loc[data["Episode"] == max_episode]

                    if not last_episode_data.empty:
                        # Check if the max_step_episode_0 exists in any 'Step' of the last episode.
                        # (Series == value).any() correctly handles potential NaNs in last_episode_data["Step"],
                        # as NaN compared to any value (including NaN) results in False.
                        found_max_step_in_last_episode = (
                            last_episode_data["Step"] == max_step_episode_0
                        ).any()

                        if not found_max_step_in_last_episode:
                            # If not found, remove all rows belonging to the last episode.
                            # A .copy() is used to ensure 'data' becomes a new DataFrame,
                            # avoiding potential views and SettingWithCopyWarning.
                            data = data.loc[data["Episode"] != max_episode].copy()
    # If the conditions for this logic block were not met (e.g., missing columns,
    # no Episode 0 data, no valid Steps in Episode 0, no max_episode > 0, etc.),
    # 'data' remains as it was (potentially with 'Episode'/'Step' columns
    # converted to numeric types by the initial part of this block).

    ######################

    # Add 'left', 'right', 'middle' columns based on 'agent.x' and 'BOUNDS'
    data["left"] = (data["agent.x"] < BOUNDS[0]).astype(int)
    data["right"] = (data["agent.x"] > BOUNDS[1]).astype(int)
    data["middle"] = (
        (data["agent.x"] >= BOUNDS[0]) & (data["agent.x"] <= BOUNDS[1])
    ).astype(int)

    # Sum the steps for each condition and rename columns
    data = (
        data.groupby(GROUP_COLUMNS)
        .agg(
            left_steps=("left", "sum"),
            right_steps=("right", "sum"),
            middle_steps=("middle", "sum"),
        )
        .reset_index()
    )

    # Convert 'Episode' column to numeric
    data["Episode"] = pd.to_numeric(data["Episode"], errors="coerce")

    # Remove spaces in 'left.monitor' and 'right.monitor' columns
    data["left.monitor"] = data["left.monitor"].str.replace(" ", "")
    data["right.monitor"] = data["right.monitor"].str.replace(" ", "")

    # Add 'filename' and 'agent' columns
    data["filename"] = filename.name
    data["agent"] = data["filename"].str.extract("(?<=_)(\d+)", expand=False)

    return data


def _find_files(logs_dir: Path, mode: str) -> list[Path]:
    # Find all CSV files
    files = list(logs_dir.glob(f"**/{mode}*.csv"))

    if not files:
        raise FileNotFoundError(f"No {mode}ing data found at {logs_dir}")

    return files


def _combine_data(files: list[Path], mode: str) -> pd.DataFrame:
    # Combine all data, ignoring empty files
    data_frames = [_read_data(file) for file in files if file.stat().st_size > 0]
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        raise ValueError(
            f"No valid {mode}ing data found. Log csv files are likely empty."
        )


def merge(logs_dir: str, results_dir: str) -> None:
    # Convert directories to Path objects
    logs_dir = Path(logs_dir)
    results_dir = Path(results_dir)

    for mode in ("train", "test"):
        print(f"Searching for {mode} data files...\n")
        files: list[Path] = _find_files(logs_dir, mode)

        print(f"Combining {mode}ing data...\n")
        data: pd.DataFrame = _combine_data(files, mode)

        # Save the combined data
        print("Saving data...\n")
        data.to_csv(results_dir / f"{mode}_results.csv", index=False)
