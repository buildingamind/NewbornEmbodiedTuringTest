from pathlib import Path
import pandas as pd

def read_data(filename: Path, bounds: list[float]) -> pd.DataFrame:
    print(f"Reading {filename}")
    data = pd.read_csv(filename, skipinitialspace=True).fillna("NA")

    # Add 'left', 'right', 'middle' columns based on 'agent.x' and 'bounds'
    data['left'] = (data['agent.x'] < bounds[0]).astype(int)
    data['right'] = (data['agent.x'] > bounds[1]).astype(int)
    data['middle'] = ((data['agent.x'] >= bounds[0]) & (data['agent.x'] <= bounds[1])).astype(int)

    # Define the grouping columns
    group_cols = [
        'Episode', 'left.monitor', 'right.monitor', 'correct.monitor',
        'experiment.phase', 'imprint.cond', 'test.cond'
    ]

    # Sum the steps for each condition and rename columns
    data = data.groupby(group_cols).agg(
        left_steps=('left', 'sum'),
        right_steps=('right', 'sum'),
        middle_steps=('middle', 'sum')
    ).reset_index()

    # Convert 'Episode' column to numeric
    data['Episode'] = pd.to_numeric(data['Episode'], errors='coerce')

    # Remove spaces in 'left.monitor' and 'right.monitor' columns
    data['left.monitor'] = data['left.monitor'].str.replace(' ', '')
    data['right.monitor'] = data['right.monitor'].str.replace(' ', '')

    # Add 'filename' and 'agent' columns
    data['filename'] = filename.name
    data['agent'] = data['filename'].str.extract('(\d+)', expand=False)

    return data

def merge(
    logs_dir: Path | str,
    results_dir: Path | str,
    csv_train_name: str = "train_results.csv",
    csv_test_name: str = "test_results.csv"
) -> None:
    # Define x_limits and calculate bounds
    x_limits = (-10, 10)
    one_third = (x_limits[1] - x_limits[0]) / 3
    bounds = [x_limits[0] + one_third, x_limits[1] - one_third]

    # Convert directories to Path objects
    logs_dir = Path(logs_dir)
    results_dir = Path(results_dir)

    if not logs_dir.exists():
        print(f"Logs directory {logs_dir} does not exist")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    # Find all train and test CSV files
    train_files = list(logs_dir.glob('**/*train.csv'))
    test_files = list(logs_dir.glob('**/*test.csv'))

    if not train_files:
        print("No training data found")
        return
    if not test_files:
        print("No testing data found")
        return

    # Combine all the training data
    print("Combining training data...\n")
    train_data = pd.concat([read_data(file, bounds) for file in train_files], ignore_index=True)

    # Combine all the testing data
    print("Combining testing data...\n")
    test_data = pd.concat([read_data(file, bounds) for file in test_files], ignore_index=True)

    # Save the combined data
    print("Saving data...\n")
    train_data.to_csv(results_dir / csv_train_name, index=False)
    test_data.to_csv(results_dir / csv_test_name, index=False)