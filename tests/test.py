import sys
sys.path.append('src') # add the src directory to the path
from netts import Brain, Body, Environment, NETT


brain = Brain(policy='CnnPolicy', algorithm='PPO', train_encoder=True)
body = Body(type="basic", dvs=False)
environment = Environment(config="parsing", executable_path="./tests/input/parsing/parsing.x86_64") # uses data if in same dir

# construct the NETT
nett = NETT(brain=brain, body=body, environment=environment)
# run the NETT
job_sheet = nett.run(dir="./tests/output", num_brains=1, train_eps=1, test_eps=1) # output directory
