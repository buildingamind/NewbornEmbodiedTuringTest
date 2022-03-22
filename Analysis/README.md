# Analysis of Virtual Controlled Rearing Experiments
This repository generates standard graphs and measures one 
might like to see as a result of their simulations. 

Before using the code in this repo make sure that you have a run a simulation as described in the Simulation folder.

## What to Analyze
The clearest data to be produced from a simulation is a performance graph. A performance graph is a bar plot of agents performance across all test trials for that agent.

Further analysis can be done using the tools package however
this package is still being developed.

## Running an analysis
To run an analysis the user simply puts the run_id of their agent.
```
python run.py get_performance run_id
```
In the command above `get_performance` is a subcommand of run.py. Users may choose to try other subcommands which can be listed using the command
```
python run.py --help
```
Once a graph is generated it will be located in the Analysis folder named as \[run_id\]_\[sub-command\].png