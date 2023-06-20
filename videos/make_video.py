#!/usr/bin/env python3

import os
import glob
import argparse
import subprocess
import cv2

parser = argparse.ArgumentParser(description='provide details for path')
parser.add_argument('--dir_path', '--dir_path', required=False, type=str, help='folder path', default="")
args = parser.parse_args()
    
dir_path = args.dir_path

dirpaths = []
print('\nNamed with wildcard *:')

## AgentRecording
agent_recording = "../Data/test_ship_recording/Recordings/AgentRecorder"
### ChamberRecording
chamber_recording = "../Data/test_ship_recording/Recordings/ChamberRecorder"
agent_pngs = glob.glob(f'{agent_recording}/*.png')
chamber_pngs = glob.glob(f'{chamber_recording}/*.png')
save_path = "../Data/test_ship_recording/Recordings/train"
for a, c in zip(agent_pngs, chamber_pngs):
    img1 = cv2.imread(c)
    img2 = cv2.imread(a)
    h_img = cv2.hconcat([img1, img2])
    os.makedirs(save_path,exist_ok=True)
    cv2.imwrite(os.path.join(save_path, os.path.basename(c)),h_img)
    


for path in [save_path]:    
    for filename in os.listdir(path):
        prefix, num = filename[:-4].split('_')
        num = num.zfill(8)
        new_filename = prefix + "_" + num + ".png"
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
    print(f"completed:{path}")
