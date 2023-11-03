#!/usr/bin/env python3

from tqdm.auto import tqdm
import argparse
import os
from subprocess import Popen, PIPE

parser = argparse.ArgumentParser(description='run analysis script')
parser.add_argument("--log-dir", 
                    type=str, 
                    help="Working directory of the agents' log files",
                    required=True)
parser.add_argument("--results-dir", 
                    type=str, 
                    help="Working directory to store the merged output",
                    required=True)
parser.add_argument("--results-name", 
                    type=str, 
                    help="File name for the R file storing the results", 
                    required=False, 
                    default="output_file.R")
parser.add_argument("--ep-bucket", 
                    type=int, 
                    help="How many episodes to group the x-axis by", 
                    required=False, 
                    default=100)
parser.add_argument("--num-episodes", 
                    type=str, 
                    help="How many episodes should be included (in case agent runs a few episodes too long)",
                    required=False, 
                    default=1000)
parser.add_argument("--key-csv", 
                    type=str, 
                    help="Full address of the key file that designates condition and correct monitor for each trial",
                    required=False, 
                    default="Keys/segmentation_key_new.csv")
parser.add_argument("--chick-file", 
                    type=str, 
                    help="Full filename (inc working directory) of the chick data CSV file",
                    required=False, 
                    default="ChickData/ChickData_Parsing.csv")



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
    cmd.extend(["--csv-train", "train_output.csv"])
    cmd.extend(["--csv-test", "test_output.csv"])
    
    return cmd, "NETT_merge_csvs.R"

def build_r_script_for_train(args):
    """Rscript NETT_train_viz.R --data-loc /data/mchivuku/embodiedai/benchmark_experiments/ship/backgroundA/small/data 
    --results-wd /data/mchivuku/embodiedai/benchmark_experiments/ship/backgroundA/small/ --ep-bucket "100"

    Args:
        args (_type_): _description_
    """
    results_dir = args.results_dir
    results_name = args.results_name
    ep_bucket = args.ep_bucket
    num_episdoes = args.num_episodes
    
    script_name = "NETT_train_viz.R"
    cmd = ["Rscript","NETT_train_viz.R"]
    cmd.extend(["--data-loc", os.path.join(results_dir, results_name)])
    cmd.extend(["--results-wd", results_dir])
    cmd.extend(["--ep-bucket", f"{ep_bucket}"])
    cmd.extend(["--num-episodes", str(num_episdoes)])
    
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
    results_dir = args.results_dir
    results_name = args.results_name
    key_csv = args.key_csv
    chick_file = args.chick_file
    
    script_name = "NETT_test_viz.R"
    cmd = ["Rscript",script_name]
    cmd.extend(["--data-loc", os.path.join(results_dir, results_name)])
    cmd.extend(["--key-csv", key_csv])
    cmd.extend(["--results-wd" , results_dir])
    cmd.extend(["--color-bars", "true"])
    cmd.extend(["--chick-file", chick_file])
    
    return cmd, f"./{script_name}"
    
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
    for script in (pbar := tqdm(["merge", "train", "test"])):
        pbar.set_description(f"Running {script}")
        try:
            run_R(script,args)
        except Exception as ex:
            print(f"Exception occurred while running the script:{str(ex)},{script}")
            raise Exception(str(ex))
    
    