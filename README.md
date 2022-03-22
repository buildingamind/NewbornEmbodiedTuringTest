# Unity environment for ChickAI: virtual controlled-rearing experiments
This is a collection of tools for simulating virtual agents under 
controlled-rearing conditions. The agents
generated and studied through this pipeline can be compared directly to real chicks 
recorded by the Building a Mind
Lab. This pipeline provides all necessary components for simulating and replicating 
embodied models from the lab.

## How to Use this Repository
This directory provides three components for building embodied virtual agents. These are 
a video game which serves as a virtual world, a set of programs to run experiments in the
virtual world, and a set of programs to visualize the data coming from the experiments.
Once users download this repo they will most likely need to open Unity at least once to 
generate an executable of the environment. After an executable is available, the user 
should run the necessary simulations. This will result in data that can be analyzed using
the script in the analysis folder.

## Sub-Directories

* `Unity`: Contains a Unity project which is a virtual replication of the VR Chambers 
used in contolled-rearing studies from the lab. This folder should be opened as a Unity
project in the Unity Editor.
* `Data`: This folder contains executables, experimental results, and analysis results.
A user will rarely interact with this folder directly. Most scripts assume this directory and all its contents exist. 
* `Simulation`: Contains the code for running experiments with simulated agents.
* `Analysis`: Contains the code for visualizing and analyzing results from the
simulation experiments.

## Projects
The table below lists the projects and publications using this pipeline.
| Project/Publication | Date | Authors                                | Branch | Webpage |
|---------------------|------|----------------------------------------|--------|---------|
|Twin Studies           |      | Denizhan Pak, Samantha Wood, Justin Wood |        |         |
|Embodied Imprinting |      | Denizhan Pak, Justin Wood        |        |         |
