import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

rv.append(average_in_episode_three_region(df))

for val in rv:
    kernel_size = 100
    kernel = np.ones(kernel_size)/kernel_size
    val = np.convolve(val,kernel,mode='valid')
    plt.plot(val,alpha=0.3)
plt.show()
