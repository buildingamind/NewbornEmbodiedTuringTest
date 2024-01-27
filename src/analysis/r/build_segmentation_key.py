#!/usr/bin/env python3
import csv
import pandas

def get_keys(object, background):
    """
    Build segmentation key
    left_right	left	right	imprinting	cond_name
    1A_00-2A_00	1A_00	2A_00	fork_backgroundA	
    Imprinted Object Imprinted Background, Novel Object Imprinted Background
    Args:
        object (_type_): _description_
        background (_type_): _description_
    """
    keynames = ["Imprinted Object Imprinted Background, Novel Object Novel Background",
                "Imprinted Object Novel Background, Novel Object Imprinted Background",
               "Imprinted Object Imprinted Background, Novel Object Imprinted Background",
               "Imprinted Object Novel Background, Novel Object Novel Background" ,
               "Rest"
               ]
    
    keys1 = []
    keys2 = []
    keys3 = []
    keys4 = []
    keys5 = []
    
    print(object, background)
    if object == "ship":
        if background == "A":
            # 1 - imprinted object (imprinted background) - novel object (novel background)
            novel_object = ["2B", "2C"]
            keys1 = ["1A_00_2B_00","1A_30_2B_30","1A_60_2B_60","1A_00_2C_00","1A_30_2C_30","1A_60_2C_60"]
            # 2 - ImprintedObj, Novel Background vs Novel Obj Imprinted Background
            keys2 = ["1B_00_2A_00","1B_30_2A_30","1B_60_2A_60","1C_00_2A_00","1C_30_2A_30","1C_60_2A_60"]
            # 3 - ImprintedObj, Imprinted Background vs Novel Obj Imprinted Background
            keys3 = ["1A_00_2A_00","1A_30_2A_30","1A_60_2A_60"]
            # 3 - ImprintedObj, Novel Background vs Novel Obj Novel Background
            keys4 = ["1B_00_2B_00","1B_30_2B_30","1B_60_2B_60","1B_00_2C_00","1B_30_2C_30","1B_60_2C_60",
                            "1C_00_2B_00","1C_30_2B_30","1C_60_2B_60","1C_00_2C_00","1C_30_2C_30","1C_60_2C_60"
                            ]
            keys5 = ["1A_00_White"]
               
        elif background == "B":
            imprinted_background = "1B"
            novel_object = ["2B", "2C"]
            ## imprinted obj (imprint background) vs novel obj vs novel background
            keys1 = ["1B_00_2A_00","1B_30_2A_30","1B_60_2A_60","1B_00_2C_00","1B_30_2C_30","1B_60_2C_60"]
            # 2 - ImprintedObj, Novel Background vs Novel Obj Imprinted Background
            keys2 = ["1A_00_2B_00","1A_30_2B_30","1A_60_2B_60","1C_00_2B_00","1C_30_2B_30","1C_60_2B_60"]
            # 3 - ImprintedObj, Imprinted Background vs Novel Obj Imprinted Background
            keys3 = ["1B_00_2B_00","1B_30_2B_30","1B_60_2B_60"]
            # 3 - ImprintedObj, Novel Background vs Novel Obj Novel Background
            keys4 = ["1A_00_2A_00","1A_30_2A_30","1A_60_2A_60","1A_00_2C_00","1A_30_2C_30","1A_60_2C_60",
                            "1C_00_2A_00","1C_30_2A_30","1C_60_2A_60","1C_00_2C_00","1C_30_2C_30","1C_60_2C_60"]
            
            keys5 = ["1B_00_White"]
            
            
        elif background == "C":
            imprinted_background = "1C"
            novel_object = ["1A", "1B"]
            ## imprinted obj (imprint background) vs novel obj vs novel background
            keys1 = ["1C_00_2A_00","1C_30_2A_30","1C_60_2A_60","1C_00_2B_00","1C_30_2B_30","1C_60_2B_60"]
            # 2 - ImprintedObj, Novel Background vs Novel Obj Imprinted Background
            keys2 = ["1A_00_2C_00","1A_30_2C_30","1A_60_2C_60","1B_00_2C_00","1B_30_2C_30","1B_60_2C_60"]
            # 3 - ImprintedObj, Imprinted Background vs Novel Obj Imprinted Background
            keys3 = ["1C_00_2C_00","1C_30_2C_30","1C_60_2C_60"]
            # 3 - ImprintedObj, Novel Background vs Novel Obj Novel Background
            keys4 = ["1B_00_2B_00","1B_30_2B_30","1B_60_2B_60","1B_00_2A_00","1B_30_2A_30","1B_60_2A_60","1A_00_2B_00","1A_30_2B_30","1A_60_2B_60","1A_00_2A_00","1A_30_2A_30","1A_60_2A_60"]

            keys5 = ["1C_00_White"]
            
    else:
        if background == "A":
            # 1 - imprinted object (imprinted background) - novel object (novel background)
            imprinted_background = ["A"]
            novel_object = ["1"]
            novel_background = ["B","C"]
            
            keys1 = ["2A_00_1B_00","2A_30_1B_30","2A_60_1B_60","2A_00_1C_00","2A_30_1C_30","2A_60_1C_60"]
            # 2 - ImprintedObj, Novel Background vs Novel Obj Imprinted Background
            keys2 = ["2B_00_1A_00","2B_30_1A_30","2B_60_1A_60","2C_00_1A_00","2C_30_1A_30","2C_60_1A_60"]
            # 3 - ImprintedObj, Imprinted Background vs Novel Obj Imprinted Background
            keys3 = ["2A_00_1A_00","2A_30_1A_30","2A_60_1A_60"]
            # 3 - ImprintedObj, Novel Background vs Novel Obj Novel Background
            keys4 = ["2B_00_1B_00","2B_30_1B_30","2B_60_1B_60","2B_00_1C_00","2B_30_1C_30","2B_60_1C_60",
                            "2C_00_1B_00","2C_30_1B_30","2C_60_1B_60","2C_00_1C_00","2C_30_1C_30","2C_60_1C_60"
                            ]
            
            keys5 = ["2A_00_White"]
            
            
        elif background == "B":
            imprinted_background = "2B"
            novel_object = ["2B", "2C"]
            ## imprinted obj (imprint background) vs novel obj vs novel background
            keys1 = ["2B_00_1A_00","2B_30_1A_30","2B_60_1A_60","2B_00_1C_00","2B_30_1C_30","2B_60_1C_60"]
            # 2 - ImprintedObj, Novel Background vs Novel Obj Imprinted Background
            keys2 = ["2A_00_1B_00","2A_30_1B_30","2A_60_1B_60","2C_00_1B_00","2C_30_1B_30","2C_60_1B_60"]
            # 3 - ImprintedObj, Imprinted Background vs Novel Obj Imprinted Background
            keys3 = ["2B_00_1B_00","2B_30_1B_30","2B_60_1B_60"]
            # 3 - ImprintedObj, Novel Background vs Novel Obj Novel Background
            keys4 = ["2A_00_1A_00","2A_30_1A_30","2A_60_1A_60","2A_00_1C_00","2A_30_1C_30","2A_60_1C_60",
                            "2C_00_1A_00","2C_30_1A_30","2C_60_1A_60","2C_00_1C_00","2C_30_1C_30","2C_60_1C_60"]
            
            keys5 = ["2B_00_White"]
            
        elif background == "C":
            imprinted_background = "2C"
            novel_object = ["1A", "1B"]
            ## imprinted obj (imprint background) vs novel obj vs novel background
            keys1 = ["2C_00_1A_00","2C_30_1A_30","2C_60_2A_60","2C_00_1B_00","2C_30_1B_30","2C_60_1B_60"]
            # 2 - ImprintedObj, Novel Background vs Novel Obj Imprinted Background
            keys2 = ["2A_00_1C_00","2A_30_1C_30","2A_60_2C_60","2B_00_1C_00","2B_30_1C_30","2B_60_1C_60"]
            # 3 - ImprintedObj, Imprinted Backgroud vs Novel Obj Imprinted Background
            keys3 = ["2C_00_1C_00","2C_30_1C_30","2C_60_1C_60"]
            # 3 - ImprintedObj, Novel Background vs Novel Obj Novel Background
            keys4 = ["2B_00_1B_00","2B_30_1B_30","2B_60_1B_60","2B_00_1A_00","2B_30_1A_30","2B_60_1A_60",
                            "2A_00_1B_00","2A_30_1B_30","2A_60_1B_60","2A_00_1A_00","2A_30_1A_30","2A_60_1A_60"]
        
            keys5 = ["2C_00_White"]
        
    ## for keys1,
    get_l_r = lambda k: ["_".join(k.split("_")[:2]), "_".join(k.split("_")[2:])]
    data = []
    cond_name = keynames[0]
    loop_keys(object, background, keys1, get_l_r, data, cond_name)
        
    cond_name = keynames[1]
    loop_keys(object, background, keys2, get_l_r, data, cond_name)
        
        
    cond_name = keynames[2]
    loop_keys(object, background, keys3, get_l_r, data, cond_name)
    
        
    cond_name = keynames[3]
    loop_keys(object, background, keys4, get_l_r, data, cond_name)
    
        
    
    cond_name = keynames[4]
    loop_keys(object, background, keys5, get_l_r, data, cond_name)
    
    
    return data

def loop_keys(object, background, keys, get_l_r, data, cond_name):
    for k in keys:
        d = {}
        #left_right	left	right	imprinting	cond_name
        p = get_l_r(k)
        left_right = p[0] + "-" + p[1]
        imprinting = f"{object}_{background}"
        data.extend(build_k(cond_name, d, p, imprinting))

def build_k( cond_name, d, p, imprinting):
    data = []
    for x in ["left","right"]:
        if x == "left":
            d = {}
            left_right = p[0] + "-" + p[1]
            
            d["left_right"] = left_right
            d["left"] = p[0]
            d["right"] = p[1]
            d["imprinting"] = imprinting
            d["cond_name"] = cond_name
            d["correct_monitor"] = "left"
            
        else:
            right_left = p[1] + "-" + p[0]
            d = {}
            
            d["left_right"] = right_left
            d["left"] = p[1]
            d["right"] = p[0]
            d["imprinting"] = imprinting
            d["cond_name"] = cond_name
            d["correct_monitor"] = "right"
        data.append(d)
        
    
    print((data))
    return data

def main():
    
    keys_ship_A = get_keys("ship","A")
    keys_ship_B = get_keys("ship", "B")

    keys_ship_C = get_keys("ship", "C")
    
    keys_fork_A = get_keys("fork","A")
    keys_fork_B = get_keys("fork", "B")
    keys_fork_C = get_keys("fork", "C")
    
    all_data = keys_ship_A+ keys_ship_B + keys_ship_C+ keys_fork_A+ keys_fork_B+keys_fork_C
    
    csv_file = "segmentation_key_new.csv"
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["left_right","left","right","imprinting","cond_name","correct_monitor"])
            writer.writeheader()
            for data in all_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")
    
    
    

if __name__== "__main__":
    main()
