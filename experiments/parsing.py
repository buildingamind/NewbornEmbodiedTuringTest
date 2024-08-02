#!/usr/bin/env python3

from nett import Body, Brain, Environment
from nett import NETT


# could these be programmatically generated from the .yaml (or vice versa)?
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
    config = "parsing",
    executable_path = "/data/wjpeacoc/builds/parsing.x86_64",
    record_agent=True,
    record_chamber=True
)

benchmarks = NETT(
    brain = brain,
    body = body,
    environment = environment
)

job_sheet = benchmarks.run(
    output_dir="/data/wjpeacoc/experiments/results/parsing/",
    num_brains=5,
    mode="full",
    train_eps=1000,
    test_eps=20
)

job_sheet.analyze(
    config = "parsing",
    run_dir="/data/wjpeacoc/experiments/results/parsing/",
    output_dir = "/data/wjpeacoc/experiments/results/parsing/analysis/"
)
