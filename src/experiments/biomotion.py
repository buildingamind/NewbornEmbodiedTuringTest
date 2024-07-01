#!/usr/bin/env python3

from nett import Body, Brain, Environment
from nett import NETT

brain = Brain(
    # how can we ensure that this is set correctly such that 
    # the Unity build's brain matches the parameters of the Brain from NETTS? 
    policy="CnnPolicy",
    algorithm="PPO",
    train_encoder=True,
    encoder="resnet10"
)

body = Body(
    type = "basic",
    wrappers = [],
    dvs = False
)

environment = Environment(
    config = "biomotion",
    executable_path = "/home/wjpeacoc/nett_env/builds/biomotion.x86_64"
)

benchmarks = NETT(
    brain = brain,
    body = body,
    environment = environment
)

job_sheet = benchmarks.run(
    output_dir="/data/wjpeacoc/experiments/results/biomotion",
    num_brains=1,
    train_eps=1,
    test_eps=1
)

benchmarks.analyze(
    config = environment.config,
    run_dir = job_sheet.output_dir,
    output_dir = job_sheet.output_dir + "/analysis"
)
