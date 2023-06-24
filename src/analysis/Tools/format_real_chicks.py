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
                if obj == "fork": o = "O1"
                else: o = "O2"
                #l.loc[(l["trialtype"]==val)]
                l["Condition"] = l["trialtype"].apply(lambda x: f"V{x}{o}")
                l.rename(columns={"perc_time":"Performance"},inplace=True)   
                l.to_csv(f"{path}{obj}_{view}_Agent{subj}_Fixed_Performance.csv",columns=["Condition", "Performance"],index=False)
                rest_l = pd.DataFrame(columns=["Episode", "Performance","Distractor","Correct"])
                print(list(range(1950,2001)))
                rest_l["Episode"] = list(range(1960,2001))
                perf = l.loc[l["trialtype"] == "rest"]["Performance"]
                rest_l.loc[:, "Performance"] = perf.values[0]
                rest_l.loc[:, "Distractor"] = "White"
                if view == "side": x = "V10"
                else: x = "V1"
                rest_l.loc[:, "Correct"] = f"V{x}{o}"
                rest_l.to_csv(f"{path}{obj}_{view}_Agent{subj}_Training_Performance.csv")
            
            #l = df.loc[(df["imp_obj"]==obj) & (df["imp_view"]==view) & (df["subj"]==subj) & (df["Experiment"]==3)].copy()