#!/usr/bin/env python3

import csv
import pandas
import argparse
import os
from subprocess import Popen, PIPE

parser = argparse.ArgumentParser(description='run analysis script')
parser.add_argument("--log-dir", type=str, help="Working directory of the agents' log files",\
    required=True)
parser.add_argument("--results-dir", type=str, help="Working directory to store the merged output",\
    required=True)
parser.add_argument("--results-name", type=str, help="File name for the R file storing the results",\
    required=False, default="segmentation_data")


parser.add_argument("--ep-bucket", type=int, help="How many episodes to group the x-axis by",\
    required=False, default=100)
parser.add_argument("--key-csv", type=str, dest="key_csv",
                    help="Full address of the key file that designates condition and correct monitor for each trial",
                    required=False, default="segmentation_key_new.csv")


def build_r_script_for_merge(args):
    """
    Build R script cmd for MERGE
    --logs-dir /data/mchivuku/embodiedai/benchmark_experiments/ship/backgroundA/small/ 
    --results-dir /data/mchivuku/embodiedai/benchmark_experiments/ship/backgroundA/small/ 
    --results-name 'data'

    Args:
        args (_type_): _description_
    """
    log_dir = args.log_dir
    results_dir = args.results_dir
    results_name = args.results_name
    
    cmd = ["Rscript","NETT_merge_csvs.R"]
    cmd.extend(["--logs-dir", log_dir])
    cmd.extend(["--results-dir", results_dir])
    cmd.extend(["--results-name", results_name])
    cmd.extend(["--csv-train", "csv_train_results.csv"])
    cmd.extend(["--csv-test", "csv_test_results.csv"])
    
    return cmd, "NETT_merge_csvs.R"

def build_r_script_for_train(args):
    """Rscript NETT_train_viz.R --data-loc /data/mchivuku/embodiedai/benchmark_experiments/ship/backgroundA/small/data 
    --results-wd /data/mchivuku/embodiedai/benchmark_experiments/ship/backgroundA/small/ --ep-bucket "100"

    Args:
        args (_type_): _description_
    """
    log_dir = args.log_dir
    results_dir = args.results_dir
    results_name = args.results_name
    ep_bucket = args.ep_bucket
    
    script_name = "NETT_train_viz.R"
    cmd = ["Rscript","NETT_train_viz.R"]
    cmd.extend(["--data-loc", os.path.join(results_dir, results_name)])
    cmd.extend(["--results-wd", results_dir])
    cmd.extend(["--ep-bucket", f"{ep_bucket}"])
    
    print(cmd)
    return cmd, script_name

def build_r_script_for_test(args):
    """Rscript NETT_test_viz.R --data-loc /data/mchivuku/embodiedai/benchmark_experiments/ship/backgroundA/small/data 
    --key-csv segmentation_key_new.csv 
    --results-wd /data/mchivuku/embodiedai/benchmark_experiments/ship/backgroundA/small/ 
    --color-dots T 

    Args:
        args (_type_): _description_
    """
    log_dir = args.log_dir
    results_dir = args.results_dir
    results_name = args.results_name
    ep_bucket = args.ep_bucket
    key_csv =args.key_csv
    
    script_name = "NETT_test_viz.R"
    cmd = ["Rscript",script_name]
    data_loc = os.path.join(results_dir, results_name)
    cmd.extend(["--data-loc", data_loc])
    cmd.extend(["--key-csv", key_csv])
    cmd.extend(["--results-wd" , results_dir])
    cmd.extend(["--color-dots", "T"])
    
    return cmd,    f"./{script_name}"
    
    
def run_R(script_name,args):
    ## cmd with arguments
    if "merge" in script_name:
        cmd, name = build_r_script_for_merge(args)
    elif "train" in script_name:
        print(script_name)
        cmd, name = build_r_script_for_train(args)
    elif "test" in script_name:
        cmd, name = build_r_script_for_test(args)
    
    else:
        raise Exception("script name not found!")
    
    p = Popen(cmd,stdin=PIPE, stdout=PIPE, stderr=PIPE)  
    output, error = p.communicate()
    
    # PRINT R CONSOLE OUTPUT (ERROR OR NOT)
    if p.returncode == 0:            
        print('R OUTPUT:\n {0}'.format(output)) 
        p.kill()
        return True           
                    
    print('R ERROR:\n {0}'.format(error))
    raise Exception(f"Error occurred in the Rscript:" + script_name)
    
  

if __name__=="__main__":
    args = parser.parse_args()
    
    for script in ["merge","train","test"]:
        try:
            run_R(script,args)
        except Exception as ex:
            print(f"Exception occurred while running the script:{str(ex)},{script}")
            raise Exception(str(ex))
    
    