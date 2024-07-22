#!/usr/bin/env python3

from nett import Body, Brain, Environment
from nett import NETT

# num_brains = 5

brain = Brain(
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
    executable_path = "/data/wjpeacoc/builds/biomotion.x86_64"
)

benchmarks = NETT(
    brain = brain,
    body = body,
    environment = environment
)

job_sheet = benchmarks.run(
    output_dir="/data/wjpeacoc/experiments/results/biomotion",
    mode="full",
    num_brains=1,
    train_eps=1000,
    test_eps=20
)
