#!/usr/bin/env python3

from nett import Body, Brain, Environment
from nett import NETT


### use the same settings for all Brains & Bodies ###
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

### per-build settings for running all tests in "one command" on server ###
# /main branch
env_main = Environment(
    config = "parsing",
    executable_path = "/data/wjpeacoc/builds/parsing_main/parsing.x86_64",
    record_agent=True,
    record_chamber=True
)

benchmarks_main = NETT(
    brain = brain,
    body = body,
    environment = env_main
)

job_sheet_main = benchmarks_main.run(
    output_dir="/data/wjpeacoc/experiments/test_output/parsing_main/",
    num_brains=1,
    mode="full",
    train_eps=1000,
    test_eps=20
)

# /dev branch
env_dev = Environment(
    config = "parsing",
    executable_path = "/data/wjpeacoc/builds/parsing_dev/parsing.x86_64",
    record_agent=True,
    record_chamber=True
)

benchmarks_dev = NETT(
    brain = brain,
    body = body,
    environment = env_dev
)

job_sheet_dev = benchmarks_dev.run(
    output_dir="/data/wjpeacoc/experiments/test_output/parsing_dev/",
    num_brains=1,
    mode="full",
    train_eps=1000,
    test_eps=20
)

### uncomment the methods below for testing a build from a specific feature branch cut from origin/dev ###
# env_<branch_to_test> = Environment(
#     config = "parsing",
#     executable_path = "/data/wjpeacoc/builds/parsing_<branch_to_test>/parsing.x86_64",
#     record_agent=True,
#     record_chamber=True
# )
#
# benchmarks_<branch_to_test> = NETT(
#     brain = brain,
#     body = body,
#     environment = env_<branch_to_test>
# )
#
# job_sheet_<branch_to_test> = benchmarks_<branch_to_test>.run(
#     output_dir="/data/wjpeacoc/experiments/test_output/parsing_<branch_to_test>/",
#     num_brains=1,
#     mode="full",
#     train_eps=1000,
#     test_eps=20
# )
#