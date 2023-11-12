import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import os
import seaborn as sns
import matplotlib.pyplot as plt


#sns.set(style="darkgrid",)
plt.style.use("dark_background")



models = ["rnd","contrastive","pathak","chicks"]
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
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    x = []
    y = []
    for i, model in enumerate(models):
        x.extend(vals[model])
        y += [model] * len(vals[model])
            
    z = tsne.fit_transform(x)
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    name = f"{trial_name} T-SNE projection"

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", len(models)),
                    data=df,linewidth=0).set(title=name)
    plt.savefig(f"{name}.png")
    plt.clf()