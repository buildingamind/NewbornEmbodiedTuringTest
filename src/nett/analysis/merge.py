from pathlib import Path

import pandas as pd
import numpy as np

import os

def read_data(filename: Path, bounds: list[int]):
  data = pd.read_csv(filename, skipinitialspace=True)

  data['left'] = (data['agent.x'] < bounds[0]).astype(int)
  data['right'] = (data['agent.x'] > bounds[1]).astype(int)
  data['middle'] = 1 - data['left'] - data['right']



  # Define the grouping columns
  group_cols = [
      'Episode', 'left.monitor', 'right.monitor', 'correct.monitor',
      'experiment.phase', 'imprint.cond', 'test.cond'
  ]

  # sum the steps for each condition
  data = (
      data.groupby(group_cols)
          .agg({
              'left': 'sum',
              'right': 'sum',
              'middle': 'sum'
          })
          .rename(columns={
              'left': 'left_steps',
              'right': 'right_steps',
              'middle': 'middle_steps'
          })
          .reset_index()
  )

  # Convert 'Episode' column to numeric
  data['Episode'] = pd.to_numeric(data['Episode'], errors='coerce')

  # Remove spaces in 'left.monitor' and 'right.monitor' columns
  data['left.monitor'] = data['left.monitor'].str.replace(' ', '', regex=False)
  data['right.monitor'] = data['right.monitor'].str.replace(' ', '', regex=False)

  # Add columns for original filename and agent ID number
  data['filename'] = os.path.basename(filename)
  data['agent'] = data['filename'].str.replace(r'\D', '', regex=True)  # Only keep the numbers

  return data

def merge(
    logs_dir: Path | str, 
    results_dir: Path | str,
    csv_train_name: str = "train_results.csv",
    csv_test_name: str = "train_results.csv") -> None:
  x_limits = [-10, 10]
  one_third = (x_limits[1] - x_limits[0]) / 3
  bounds = [
    x_limits[0] + one_third, 
    x_limits[1] - one_third
  ]

  if type(logs_dir) == str:
    logs_dir = Path(logs_dir)
  if type(results_dir) == str:
    results_dir = Path(results_dir)

  if not logs_dir.exists():
    print("Logs directory does not exist")
    return
  if not results_dir.exists():
    results_dir.mkdir(parents=True, exist_ok=True)

  # Load the data
  train_files = list(logs_dir.glob('**/*train.csv'))
  test_files = list(logs_dir.glob('**/*test.csv'))

  if len(train_files) == 0:
    print("No training data found")
    return
  if len(test_files) == 0:
    print("No testing data found")
    return
  
  # Combine all the training
  print("Combining training data...\n")
  train_data = [read_data(file, bounds) for file in train_files]
  train_data = pd.concat(train_data, ignore_index=True)

  # Combine all the testing
  print("Combining testing data...\n")
  test_data = [read_data(file, bounds) for file in test_files]
  test_data = pd.concat(test_data, ignore_index=True)

  # Save it
  print("Saving data...\n")


  # Write train_data to a CSV file if csv_train_name is not None
  train_data.to_csv(results_dir / csv_train_name, index=False)

  # Write test_data to a CSV file if csv_test_name is not None
  test_data.to_csv(results_dir / csv_test_name, index=False)
