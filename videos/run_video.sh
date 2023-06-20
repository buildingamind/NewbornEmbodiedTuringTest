#!/bin/bash
## soccer
#ffmpeg -framerate 5 -i /home/mchivuku/projects/visual_parsing/pipeline_embodied_sb/logfile/AgentRecorder/output_%08d.png /home/mchivuku/projects/visual_parsing/pipeline_embodied_sb/logfile/AgentRecorder/AgentRecorder_smalldp5_agent_1.mp4
#ffmpeg -framerate 5 -i /home/mchivuku/projects/visual_parsing/pipeline_embodied_sb/logfile/ChamberRecorder/output_%08d.png /home/mchivuku/projects/visual_parsing/pipeline_embodied_sb/logfile/ChamberRecorder/ChamberRecorder_smalldp5_agent_1.mp4


#ffmpeg -framerate 5 -i /data/mchivuku/embodiedai/imprinting/experiments/background_experiments/testdvswrappersmall2001kthres90_6/dvs_obs/obs_%08d.png /data/mchivuku/embodiedai/imprinting/experiments/background_experiments/testdvswrappersmall2001kthres90_6/dvs_obs/dvs_dp5_agent_1.mp4
#ffmpeg -framerate 5 -i /data/mchivuku/embodiedai/imprinting/experiments/background_experiments/testdvswrappersmall2001kthres90_6/grayscale_obs/img_%08d.png /data/mchivuku/embodiedai/imprinting/experiments/background_experiments/testdvswrappersmall2001kthres90_6/grayscale_obs/grayscale_dp5_agent_1.mp4
ffmpeg -framerate 5 -i /home/mchivuku/projects/embodied_pipeline/benchmark_experiments/Data/test_ship_recording/Recordings/train/output_%08d.png /home/mchivuku/projects/embodied_pipeline/benchmark_experiments/Data/test_ship_recording/Recordings/train/train_recording.mp4

