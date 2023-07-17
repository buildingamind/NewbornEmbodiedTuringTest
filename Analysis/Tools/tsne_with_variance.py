import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import os
import seaborn as sns
import matplotlib.pyplot as plt


#sns.set(style="darkgrid",)
#plt.style.use("dark_background")



models = ["Newborn Chicks","Contrastive Curiosity","Intinsic Curiosity","RND"]
models2 = ["chicks","contrastive","pathak","rnd"]
mp = {}
for i, model in enumerate(models):
    mp[model] = models2[i]

objs = ["ship","fork"]
angles = ["front","side"]
trials = ["Training","Fixed"]#,"Matched"]
colors = sns.color_palette(["#FF0000","#27BFC7","#0077BC","#1F3765"])

for trial in trials:
    vals = {}
    for model in models:
        vals[model] = []
        for obj in objs:
            for angle in angles:
                for agent in range(27):
                    path = f"../organized/viewpoint_{mp[model]}_new/{obj}_{angle}_Agent{agent}_{trial}_Performance.csv"
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
    x = []
    y = []
    for i, model in enumerate(models):
        x.extend(vals[model])
        y += [model] * len(vals[model])
            
    x = np.array(x)

    for i in range(1):
        z = TSNE(n_components=2, verbose=1, random_state=i,perplexity=30).fit_transform(x)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    #for t in range(len(perf)):
    #    df[f"trial-{t}"] 
    name = f"{trial_name} T-SNE projection"
    fig, ax = plt.subplots(figsize=(16,9))


    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=colors,
                    data=df, s=140,ax=ax,).set(title=name)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0,fontsize=16)
    plt.savefig(f"{name}.png")
    plt.clf()
