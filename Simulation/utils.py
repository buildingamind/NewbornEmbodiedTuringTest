#!/usr/bin/env python3
import glob
import logging
import sys
import os
import json
import socket
import pandas as pd
import numpy as np


def create_logger(name, loglevel=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(fmt=f'%(asctime)s - {name} - %(message)s', \
                datefmt='%d/%m/%Y %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def port_in_use(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
    except socket.error:
        return True
    return False


def save_configuration(args, log_dir,filename="configuration.json"):
    dict = {}
    dict.update(vars(args))
    
    log_file = os.path.join(log_dir, filename)
    with open(log_file, "w") as f:
        json.dump(dict, f)
    print("saved configuration to logfile:{}".format(log_file))
    
    
def compute_train_performance(path):
    x,y = [], []
    try:
        training_files = glob.glob(os.path.join(path, "*.csv"))
        
        if len(training_files) == 0:
            raise Exception(f"Training file: {training_files} was not found in the {path}")
        
        
        for file_name in training_files:
            
            log_df = pd.read_csv(file_name, skipinitialspace=True)
            
            percents,df,values = average_in_episode_three_region(log_df,"agent.x")
            y = moving_average(values, window=100)
            x = list([i for i in range(0,len(y))])
            
            break
            
        
        return x, y
    except Exception as ex:
        print(str(ex))
        
    return x,y

def get_train_performance_plot_data(path):
    x,y = [], []
    try:
        training_files = glob.glob(os.path.join(path, "*.csv"))
        
        if len(training_files) == 0:
            raise Exception(f"Training file: {training_files} was not found in the {path}")
        
        file_name = training_files.pop()
        log_df = pd.read_csv(file_name, skipinitialspace=True)            
        percents,df,values = average_in_episode_three_region(log_df,"agent.x")
        
        val = []
        for key in percents:
            val.append(percents[key])
            
            
        kernel_size = 100
        kernel = np.ones(kernel_size)/kernel_size
        convolved_val = np.convolve(val,kernel,mode='valid')
        
        return convolved_val

    
    except Exception as ex:
        print(str(ex))

def average_in_episode_three_region(log,column='agent.x',transient=90):
    """
    Train performance

    Args:
        log (_type_): _description_
        column (str, optional): _description_. Defaults to 'ChickAgent.x'.
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

        rv = []
        for key in percents:
            rv.append(percents[key])
        
        return (percents,log,rv)
    except Exception as ex:
        print(str(ex))
        return (None, None)
    
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')
    
    