import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_move_dir(x1, y1, x2, y2):
    x,y = x2 - x1, y2 - y1
    return np.arctan2(y,x)

def get_dir_diff(log):
    x = log["ChicAgent.x"]
    y = log["ChicAgent.y"]
    
    x1 = x[:-1].to_numpy()
    x2 = x[1:].to_numpy()
    y1 = y[:-1].to_numpy()
    y2 = y[1:].to_numpy()

    dirs = get_move_dir(x1, y1, x2, y2)
    rotations = log["ChicAgent.rotations"][:-1].to_numpy()

    return (dirs - rotations).mean()

def analyze_agent(agent):
    data = pd.read_csv(agent)