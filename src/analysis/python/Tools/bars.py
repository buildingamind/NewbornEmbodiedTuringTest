import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import os
import seaborn as sns
import matplotlib.pyplot as plt


#sns.set(style="darkgrid",)
plt.style.use("dark_background")



models = ["rnd","contrastive","pathak","chicks"]#,"supervised"]
objs = ["ship","fork"]
angles = ["front","side"]
trials = ["Training","Fixed"]#,"Matched"]

for trial in trials:
    vals = {}
    for model in models:
        vals[model] = []
        for obj in objs:
            for angle in angles:
                for agent in range(27):
                    path = f"../organized/viewpoint_{model}_new/{obj}_{angle}_Agent{agent}_{trial}_Performance.csv"
                    is_exist = os.path.exists(path)
                    if is_exist:
                        perf = pd.read_csv(path)["Performance"]
                        perf = perf.to_numpy()
                        if trial == "Training":
                            trial_name = "Rest"
                            perf = perf[-40:]
                        else:
                            trial_name = trial
                            perf = perf[:-1]
                        vals[model].append(perf)
    
            
    fig, ax = plt.subplots()
    ax.set_ylim(0,1)
    scores = {}
    for model in models:
        scores[model] = (np.mean(vals[model]), np.std(vals[model]) * np.sqrt(1/39))
    print(scores)
    df = pd.DataFrame(scores)
    df.index = ["Mean", "std"]
    df = df.transpose()
    print(df.transpose())
    ax.set_title(f"{trial}-Condition")
    sns.barplot(data=df, x=df.index, y="Mean",errorbar=("sd",2),# hue=df.index.to_list(),
            palette=sns.color_palette("hls", len(models)), ax=ax,)

    plt.savefig(f"{trial}_bars.png")
    plt.clf()
