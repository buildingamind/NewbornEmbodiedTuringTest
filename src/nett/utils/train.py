"""
Train performance

This module contains the functions to compute the train performance of the agent.

Functions:
    compute_train_performance: Compute Train performance
    average_in_episode_three_region: Train performance
    moving_average: Smooth values by doing a moving average
"""

import os
import glob
import numpy as np
import pandas as pd

def compute_train_performance(path) -> tuple[list, np.ndarray | list]:
    """
    Compute Train performance

    Args:
        path (Path or String?): path to training files # TODO check the type

    Returns:
        x (list): list of the episode numbers
        y (numpy.array) : the moving averages of the success rate 
    """
    x,y = [], []
    try:
        training_files = glob.glob(os.path.join(path, "*.csv"))

        if len(training_files) == 0:
            raise Exception(f"Training file: {training_files} was not found in the {path}")


        for file_name in training_files:

            log_df = pd.read_csv(file_name, skipinitialspace=True)

            _, _, values = average_in_episode_three_region(log_df,"agent.x") # percents,df,
            y = moving_average(values, window=100)
            x = list(range(len(y)))

            break


        return x, y
    except Exception as ex:
        print(str(ex))

    return x,y

def average_in_episode_three_region(log: pd.DataFrame, column: str = 'agent.x', transient: int = 90) -> tuple[dict, pd.DataFrame, list]:
    """
    Train performance

    Args:
        log (_type_): _description_
        column (str, optional): _description_. Defaults to 'agent.x'.
        transient (int, optional): _description_. Defaults to 90.

    Returns:
        _type_: _description_
    """
    try:
        log.loc[log.Episode % 2 == 1, column] *= -1
        #Translate coordinates
        log[column] += 10
        #Bin into 3 sections
        log[column] = pd.cut(log[column], [-0.1,20/3,40/3,20.1],labels=["Distractor","Null","Imprint"])
        episodes = log.Episode.unique()
        percents = {}
        for ep in episodes:
            #Get success percentage
            l = log[log["Episode"]==ep]
            l = l[l["Step"]>transient]
            total = l[l[column]=="Distractor"].count() + l[l[column]=="Imprint"].count()
            success = l[l[column]=="Imprint"].count()/total
            percents[ep] = success[column]

            if np.isnan(percents[ep]):
                percents[ep] = 0.5

        rv = list(percents.values())

        return (percents,log,rv)
    except Exception as ex:
        print(str(ex))
        return (None, None, None)

def moving_average(values: list, window: int) -> np.ndarray:
    """
    Smooth values by doing a moving average.

    Args:
        values (numpy.array): The input array of values.
        window (int): The size of the moving window.

    Returns:
        numpy.array: The smoothed array of values.
    """
    weights: np.ndarray = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')
