from pathlib import Path
import pandas as pd

# Define x_limits and calculate BOUNDS of chamber
X_LIMITS: tuple[float, float] = (-33.15, 33.15)
AGENT_RADIUS: float = 3.0  # Radius of the agent in meters
AGENT_LIMITS: tuple[float, float] = (
    X_LIMITS[0] + AGENT_RADIUS,
    X_LIMITS[1] - AGENT_RADIUS,
)
ONE_THIRD: float = (AGENT_LIMITS[1] - AGENT_LIMITS[0]) / 3
BOUNDS: list[float] = [AGENT_LIMITS[0] + ONE_THIRD, AGENT_LIMITS[1] - ONE_THIRD]

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
    # Convert columns ["Episode", "Step", "agent.x", "agent.z"] to numeric types, coercing errors.
    # This handles cases where these columns might contain 'NA' strings (due to prior fillna)
    # or other non-numeric values.
    for column in ["Episode", "Step", "agent.x", "agent.z", "agent.angle", "head.angle"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

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
