#!/usr/bin/env python3

import pandas as pd
import argparse
import os
import glob
import json
parser = argparse.ArgumentParser(description='validate logs script')
parser.add_argument("--log-dir", type=str, help="Working directory of the agents' log files",\
    required=True)
parser.add_argument("--key",type=str, help="training key' log files",\
    required=True)

def validate_train(path, imprint):
    train_file = glob.glob(os.path.join(path, f"*train.csv"))
    train_file_status = {}
    if len(train_file)==0:
        print("train file not found")
        return 
    df = pd.read_csv(train_file[0])
    distract = "White"
    filtered_df = df.loc[((df[df.columns[5]].str.contains(distract)\
                & df[df.columns[6]].str.contains(imprint)) | 
                                  
                (df[df.columns[5]].str.contains(imprint)\
                & df[df.columns[6]].str.contains(distract)
                
                )),:]
    
    print("len df:{}, len filtered_df: {}".format(len(df),len(filtered_df)))
    assert len(df) == len(filtered_df)
    
    
    train_file_status["num_recs"] = len(filtered_df)
    train_file_status["num_episodes"] = len(df.groupby("Episode"))
    train_file_status["valid"] = "true"
    train_file_status["train_filename"] = train_file[0]
    return train_file_status
    

def validate_test(path,key):
    object = "ship" if key.startswith("1") else "fork"
    test_file = glob.glob(os.path.join(path, f"*exp.csv"))
    if len(test_file)==0:
        print("test file not found")
        return 
    df = pd.read_csv(test_file[0])
    ship_keys = [("1A_00","2A_00"),("1A_00","2B_00"),("1A_00","2C_00"),
                 ("1B_00","2A_00"),("1B_00","2B_00"),("1B_00","2C_00"),
                 ("1C_00","2A_00"),("1C_00","2B_00"),("1C_00","2C_00"),
                 
                 ("1A_30","2A_30"),("1A_30","2B_30"),("1A_30","2C_30"),
                 ("1B_30","2A_30"),("1B_30","2B_30"),("1B_30","2C_30"),
                 ("1C_30","2A_30"),("1C_30","2B_30"),("1C_30","2C_30"),
                 
                 ("1A_60","2A_60"),("1A_60","2B_60"),("1A_60","2C_60"),
                 ("1B_60","2A_60"),("1B_60","2B_60"),("1B_60","2C_60"),
                 ("1C_60","2A_60"),("1C_60","2B_60"),("1C_60","2C_60")]
    
    fork_keys = [("2A_00","1A_00"),("2A_00","1B_00"),("2A_00","1C_00"),
                 ("2B_00","1A_00"),("2B_00","1B_00"),("2B_00","1C_00"),
                 ("2C_00","1A_00"),("2C_00","1B_00"),("2C_00","1C_00"),
                
                 ("2A_30","1A_30"),("2A_30","1B_30"),("2A_30","1C_30"),
                 ("2B_30","1A_30"),("2B_30","1B_30"),("2B_30","1C_30"),
                 ("2C_30","1A_30"),("2C_30","1B_30"),("2C_30","1C_30"),
                
                 ("2A_60","1A_60"),("2A_60","1B_60"),("2A_60","1C_60"),
                 ("2B_60","1A_60"),("2B_60","1B_60"),("2B_60","1C_60"),
                 ("2C_60","1A_60"),("2C_60","1B_60"),("2C_60","1C_60")]
                 
                 
    if object=="ship":
        keys = ship_keys
    else:
        keys = fork_keys
    
    test_status = {}
    test_status["test_filename"] = test_file[0]
    
    if key == "1A_00":
        keys.append(("1A_00","White"))
    elif key == "1B_00":
        keys.append(("1B_00","White"))
    elif key == "1C_00":
        keys.append(("1C_00","White"))
    elif key == "2A_00":
        keys.append(("2A_00","White"))
    elif key == "2B_00":
        keys.append(("2B_00","White"))
    elif key == "2C_00":
        keys.append(("2C_00","White"))
    
    print(len(keys))
    for (imprint, distract) in keys:
        
        for i in range(2):
            if i == 0:
                
                filtered_df = df.loc[((df[df.columns[5]].str.contains(distract)\
                    & df[df.columns[6]].str.contains(imprint))),:]
                if len(filtered_df.groupby("Episode")) == 20:
                    status = "valid"
                else:
                    status = "invalid"
                
                group_row = filtered_df.loc[(filtered_df[' correct.monitor'].str.contains("right"))]
                
                
                k = "_".join([distract, imprint]) 
                cnt = len(filtered_df.groupby("Episode"))
                print(f"{k}:{cnt}")
                print({"epsiodes":len(filtered_df.groupby("Episode")), 
                                                              "status":status, 'correct_monitor':len(group_row), 
                                                              'imprint':imprint, 'right.monitor':len(filtered_df.loc[(filtered_df[' right.monitor'].str.contains(imprint))])})
                test_status["_".join([distract, imprint])] = {"epsiodes":len(filtered_df.groupby("Episode")), 
                                                              "status":status, 'correct_monitor':len(group_row), 
                                                              'imprint':imprint, 'right.monitor':len(filtered_df.loc[(filtered_df[' right.monitor'].str.contains(imprint))])}
            else:
                
                filtered_df = df.loc[(
                    (df[df.columns[5]].str.contains(imprint)\
                    & df[df.columns[6]].str.contains(distract)
                    
                    )),:]
                if len(filtered_df.groupby("Episode")) == 20:
                    status = "valid"
                else:
                    status = "invalid"
                   
                k = "_".join([imprint, distract]) 
                cnt = len(filtered_df.groupby("Episode"))
                print(f"{k}:{cnt}")
                group_row = filtered_df.loc[(filtered_df[' correct.monitor'].str.contains("left"))]
                print({"epsiodes":len(filtered_df.groupby("Episode")), 
                                                              "status":status, 'correct_monitor':len(group_row), 
                                                              'imprint':imprint, 'left.monitor':len(filtered_df.loc[(filtered_df[' left.monitor'].str.contains(imprint))])})
                test_status["_".join([imprint, distract])] = {"epsiodes":len(filtered_df.groupby("Episode")), 
                                                              "status":status, 'correct_monitor':len(group_row), 
                                                              'imprint':imprint,
                                                              'left.monitor':len(filtered_df.loc[(filtered_df[' left.monitor'].str.contains(imprint))])}
    
    test_status["num_conditions"] = len(keys)
    return test_status

    

if __name__=="__main__":
    args = parser.parse_args()
    log_dir = args.log_dir
    key = args.key
    
    ## validate train
    train_status = validate_train(log_dir, key)
    print(train_status)
    
    if train_status:
        test_status = validate_test(log_dir, key)
    
    train_status.update(test_status)
    
    with open(os.path.join(log_dir,"status.json"),'w') as f:
        json.dump(train_status,f, indent = 4)
        
    
        
        