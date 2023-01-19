import pandas as pd
import os

# Specify path
path = '../organized/viewpoint_chicks_new/'

# Check whether the specified
# path exists or not
isExist = os.path.exists(path)
if not isExist:
    os.mkdir(path)

df = pd.read_csv("Wood_2013_PNAS_results_summary.csv")

objs = ["ship","fork"]
views = ["front","side"]
subjs = range(1,7)
exps = [1,2,3]

for obj in objs:
    for view in views:
        for subj in subjs:
            l = df.loc[(df["imp_obj"]==obj) & (df["imp_view"]==view) & (df["subj"]==subj) & (df["Experiment"]!=3)].copy()
            if not l.empty:
                print(l)
                if obj == "fork": o = "O1"
                else: o = "O2"
                #l.loc[(l["trialtype"]==val)]
                l["Condition"] = l["trialtype"].apply(lambda x: f"V{x}{o}")
                _ = l.rename(columns={"perc_time":"Performance"},inplace=True)   
                _ = l.to_csv(f"{path}{obj}_{view}_Agent{subj}_Fixed_Performance.csv",columns=["Condition", "Performance"],index=False)
                rest_l = pd.DataFrame(columns=["Episode", "Performance","Distractor","Correct"])
                rest_l[:,"Episode"] = list(range(1960,2000))
                rest_l[:, "Performance"] = l.loc[l["trialtype"] == "rest"]["Performance"]
                rest_l[:, "Distractor"] = "White"
                rest_l[:, "Correct"] = f"V{x}{o}"
                print(rest_l)