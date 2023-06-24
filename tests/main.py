#!/usr/bin/env python3

import os
from pathlib import  Path
import glob
import shutil

## transfer
def copy(srcpaths, destpath, object, type, background):
    for index, p in enumerate(srcpaths):
        outpath = os.path.join(destpath,object,type,background)
        os.makedirs(outpath,exist_ok=True)
        shutil.copyfile(p, os.path.join(outpath, os.path.basename(p)))

# ship small
s = "/data/mchivuku/embodiedai/benchmark_experiments"    
m =  "/data/mchivuku/samantha_output"
backgroundA_paths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/ship*backgroundA_exp*/Env_Logs/*.csv")
backgroundB_paths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/ship*backgroundB_exp*/Env_Logs/*.csv")
backgroundC_paths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/ship*backgroundC_exp*/Env_Logs/*.csv")

copy(backgroundA_paths, m, "ship","small", "backgroundA")   
copy(backgroundB_paths, m, "ship","small", "backgroundB")   
copy(backgroundC_paths, m, "ship","small", "backgroundC")   

backgroundA_mpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/ship*backgroundA_med_exp*/Env_Logs/*.csv")
backgroundB_mpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/ship*backgroundB_med_exp*/Env_Logs/*.csv")
backgroundC_mpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/ship*backgroundC_med_exp*/Env_Logs/*.csv")


copy(backgroundA_mpaths, m, "ship","medium", "backgroundA")   
copy(backgroundB_mpaths, m, "ship","medium", "backgroundB")   
copy(backgroundC_mpaths, m, "ship","medium", "backgroundC") 


fbackgroundA_paths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/fork*backgroundA_exp*/Env_Logs/*.csv")
fbackgroundB_paths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/fork*backgroundB_exp*/Env_Logs/*.csv")
fbackgroundC_paths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/fork*backgroundC_exp*/Env_Logs/*.csv")

copy(fbackgroundA_paths, m, "fork","small", "backgroundA")   
copy(fbackgroundB_paths, m, "fork","small", "backgroundB")   
copy(fbackgroundC_paths, m, "fork","small", "backgroundC")   

fbackgroundA_mpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/fork*backgroundA_med_exp*/Env_Logs/*.csv")
fbackgroundB_mpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/fork*backgroundB_med_exp*/Env_Logs/*.csv")
fbackgroundC_mpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/fork*backgroundC_med_exp*/Env_Logs/*.csv")


copy(fbackgroundA_mpaths, m, "fork","medium", "backgroundA")   
copy(fbackgroundB_mpaths, m, "fork","medium", "backgroundB")   
copy(fbackgroundC_mpaths, m, "fork","medium", "backgroundC") 
         

### large

sbackgroundA_lpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/ship*backgroundA_large_exp*/Env_Logs/*.csv")
sbackgroundB_lpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/ship*backgroundB_large_exp*/Env_Logs/*.csv")
sbackgroundC_lpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/ship*backgroundC_large_exp*/Env_Logs/*.csv")


copy(sbackgroundA_lpaths, m, "ship","large", "backgroundA")   
copy(sbackgroundB_lpaths, m, "ship","large", "backgroundB")   
copy(sbackgroundC_lpaths, m, "ship","large", "backgroundC") 
         

fbackgroundA_lpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/fork*backgroundA_large_exp*/Env_Logs/*.csv")
fbackgroundB_lpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/fork*backgroundB_large_exp*/Env_Logs/*.csv")
fbackgroundC_lpaths = glob.glob("/data/mchivuku/embodiedai/benchmark_experiments/fork*backgroundC_large_exp*/Env_Logs/*.csv")


copy(fbackgroundA_lpaths, m, "fork","large", "backgroundA")   
copy(fbackgroundB_lpaths, m, "fork","large", "backgroundB")   
copy(fbackgroundC_lpaths, m, "fork","large", "backgroundC") 