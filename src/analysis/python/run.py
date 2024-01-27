import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Generate Analysis of agents. Will return nothing unless desired plot is specified.')
parser.add_argument('-r','--train', help='whether to plot the training of the agents',action='store_true')
parser.add_argument('-e','--test', help='whether to plot the test trials of the agents',action='store_true')
parser.add_argument('--run_id', help='Name of the agent to analyze',required=True)

args = parser.parse_args()

def read_log(name,version="train"):
    if version == "test": version = "test/1"
    path = f"../Data/Runs/{name}/logs/{version}/ChickAgent.txt"
    log = pd.read_csv(path, skipinitialspace=True)
    return log

def average_three_region(log,column='agent.x',transient=90):
    log.loc[log.Episode % 2 == 1, column] *= -1
    #Translate coordinates
    log[column] += 10
    #Bin into 3 sections
    log[column] = pd.cut(log[column], [-0.1,20/3,40/3,20.1],labels=["Distractor","Null","Imprint"])
    episodes = log.Episode.unique()
    percents = {}
    summation = 0
    for ep in episodes:
        #Get success percentage
        l = log[log["Episode"]==ep]
        l = l[l["Step"]>transient]
        total = l[l[column]=="Distractor"].count() + l[l[column]=="Imprint"].count()
        success = l[l[column]=="Imprint"].count()/total

        if np.isnan(success[column]):
            summation += 0.5
        else:
            summation += success[column]

    return summation/len(episodes)


def average_in_episode_three_region(log,column='agent.x',transient=90):
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
    return rv


if args.test:
    log = read_log(args.run_id, version="test")
    preformance = average_three_region(log)
    plt.ylim([0,1])
    plt.bar(x=[1],height=preformance)
    plt.savefig("./test.png")
    plt.clf()


if args.train:
    log = read_log(args.run_id, version="train")
    val = average_in_episode_three_region(log)

    kernel_size = 100
    kernel = np.ones(kernel_size)/kernel_size
    val = np.convolve(val,kernel,mode='valid')
    plt.ylim([0,1])
    plt.plot(val,alpha=0.3)
    plt.savefig("train.png")
    plt.clf()
