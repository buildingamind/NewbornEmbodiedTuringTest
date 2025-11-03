# Developer Notes

## Introduction

This document serves as a guide to the development process of the NETT toolkit. It provides an overview of the toolkit's development workflow.

## Online Documentation Website

The documentation website is hosted on GitHub Pages and is available at [https://buildingamind.github.io/NewbornEmbodiedTuringTest/](https://buildingamind.github.io/NewbornEmbodiedTuringTest/). It grabs the latest documentation from the `docs` folder in the repository. It creates the documentation using Sphinx. The documentation is written in reStructuredText format and is located in the `docs/source` folder. 

All assets used by Sphinx are located in the `docs/source/_static` folder. The `index.rst` file in the `docs/source` folder is the main entry point for the documentation. Each section of the documentation is pulled in from the `index.rst` file in each subdirectory in `docs/source`.

The website is updated automatically when changes are pushed to the `main` branch. To see how this process works, please see the `.github/workflows/docs.yml` file.

The configuration for the website is defined in `conf.py`. The website uses the `ReadTheDocs` theme. It is possible to change the theme by modifying the `html_theme` variable in `conf.py`. The website currently uses three extensions: `sphinx.ext.autodoc`, `sphinx.ext.napoleon`, and `myst-parser`. The `autodoc` extension is used to automatically generate documentation from the docstrings in the source code. The `napoleon` extension is used to parse the Google-style docstrings. The `myst-parser` extension is used to parse markdown files. 

### Installation

#### Linux
Install the required packages using the typical methods listed on the [README](../../README.md).
#### MacOS
You are not able to run the code locally on MacOS. However, you can install the necessary packages for building the documentation locally. First, you need to create a python environment using the following command:
```bash
conda create -y -n nett_docs python=3.10.8
```
Then, activate the environment:
```bash
conda activate nett_docs
```
You can then install the required packages from the file `docs/mac_requirements.txt` by running the following command:
```bash
pip install -r docs/mac_requirements.txt
```
#### Windows
TBD

### Building the Documentation Locally

To build the documentation locally, you can run the following command:

```bash
sphinx-build -M html docs/source/ docs/build/
```

Subsequent builds can be done by running the following commands:
```bash
cd docs
make html
```

The documentation can be viewed by opening the `index.html` file in the `docs/build` folder in a web browser.

## NETT Architecture and Development

### Overview
The Newborn Embodied Turing Test (NETT) is a framework for training and testing AI agents in Unity-based environments. The codebase is structured around three main components: Brain (neural networks), Body (sensory processing), and Environment (Unity simulation).

### Core Architecture

#### NETT Class (nett.py)
The main orchestrator class that manages the entire training and testing pipeline.

**Key Components:**
- **Configuration Management**: Validates configs against JSON schema
- **Task Scheduling**: Creates and manages parallel jobs
- **Memory Management**: Tracks GPU memory usage and allocates resources
- **Executor System**: Manages concurrent training/testing using `ProcessPoolExecutor`

**Important Methods:**
- `run()`: Main entry point for launching experiments
- `single_run()`: Handles individual benchmark configurations
- `task_waiter()`: Monitors and schedules task execution
- `analyze()`: Static method for post-run analysis using R scripts

#### Brain Component (`brain/brain.py`)
Manages the neural network architecture and training algorithms.

**Key Features:**
- **Algorithm Support**: PPO, DQN, A2C, SAC, TD3, etc. via Stable-Baselines3
- **Custom Encoders**: Various CNN architectures (small, medium, large, etc.)
- **Policy Networks**: CnnPolicy, MlpPolicy, MultiInputPolicy
- **Reward Functions**: Closeness, completeness, RE3 intrinsic motivation
- **Checkpointing**: Automatic model saving during training

**Architecture Flow:**
```
Environment Observations → Encoder → Policy Network → Actions
                      ↓
                 Reward Function → Algorithm Update
```

#### Body Component (`body/body.py`)
Handles sensory processing and environment wrapping.

**Key Features:**
- **Wrapper System**: DVS (Dynamic Vision Sensor), binocular vision, frame stacking
- **Recording**: Video recording of agent perspective and environment
- **Validation**: Environment compatibility checking
- **Multi-agent Support**: PettingZoo integration for multi-agent scenarios

**Wrapper Chain:**
```
Unity Environment → Body Wrappers → Gym/PettingZoo Environment → VecEnv
```

#### Environment Component (environment.py)
Interfaces with Unity executables and manages simulation parameters.

**Key Features:**
- **Unity Integration**: Uses ML-Agents toolkit for Unity communication
- **Condition Management**: Handles different experimental conditions
- **Recording**: Chamber-wide video recording
- **Multi-modal Support**: Supports both single and multi-agent environments

### Task Management System

#### Task Structure
- **TaskConfig**: Configuration for individual tasks (brain_id, condition, paths, etc.)
- **Agent**: Combines Brain, Body, and Environment for a specific task
- **Task**: Complete unit of work with config and agent
- **TaskList**: Iterator over all tasks for a given experiment

#### Execution Flow
1. **Configuration Validation**: Validate all input configs against schema
2. **Task Generation**: Create tasks for each brain × condition combination
3. **Memory Estimation**: Calculate required GPU memory per task
4. **Device Allocation**: Assign tasks to available GPU devices
5. **Parallel Execution**: Run tasks using ProcessPoolExecutor
6. **Result Collection**: Gather training logs and test results

### Memory Management

#### MemoryManager Class
- **GPU Memory Tracking**: Monitors VRAM usage across devices
- **Dynamic Allocation**: Assigns tasks based on available memory
- **Memory Estimation**: Calculates required memory for each task type

#### Memory Calculation
```python
# Auto memory calculation based on:
# - Environment complexity
# - Model size (encoder + policy)
# - Batch size and buffer size
# - Number of parallel environments
```

### Parallel Processing

#### SubprocVecEnv Usage
The code extensively uses `SubprocVecEnv` for parallel environment execution:
- **Training**: Multiple environments running simultaneously
- **Testing**: Parallel evaluation across different conditions
- **Memory Efficiency**: Each subprocess manages its own memory space

#### Process Management
- **ProcessPoolExecutor**: Manages worker processes for tasks
- **Future Objects**: Track task completion and results
- **Queue System**: Communication between main process and workers

### Data Flow

#### Training Pipeline
```
Unity Environment → Body Wrappers → VecEnv → Brain Training → Model Checkpoints
                                         ↓
                                    Training Logs → CSV Files
```

#### Testing Pipeline
```
Trained Model → Unity Environment → Body Wrappers → VecEnv → Evaluation → Test Results
```

### Configuration System

#### Schema Validation
- **JSON Schema**: Validates all configuration parameters
- **Type Checking**: Ensures correct data types for all parameters
- **Default Values**: Provides sensible defaults for optional parameters

#### Config Structure
```yaml
name: "Experiment1"
episodes:
  train: 5000
  test: 100
brain:
  algorithm: "PPO"
  encoder: "small"
  policy: "CnnPolicy"
body:
  wrappers: ["dvs"]
environment:
  executable_path: "path/to/unity.x86_64"
  conditions: ["condition1", "condition2"]
```

### Error Handling

#### Common Issues
- **SubprocVecEnv Errors**: Requires `if __name__ == '__main__':` guard
- **Unity Connection**: Port conflicts and environment startup failures
- **Memory Issues**: GPU OOM errors and memory estimation failures
- **File Permissions**: Unity executable and temp directory permissions

#### Recovery Mechanisms
- **Port Management**: Automatic port selection for Unity environments
- **Memory Fallback**: Dynamic memory allocation adjustment
- **Task Retry**: Failed task rescheduling
- **Graceful Shutdown**: Proper environment cleanup

### Analysis System

#### R Integration
The framework includes R scripts for post-training analysis:
- **NETT_merge_csvs.R**: Combines training/testing logs
- **NETT_train_viz.R**: Generates training performance plots
- **NETT_test_viz.R**: Creates test results visualizations

#### Output Structure
```
output_dir/
├── experiment_name/
│   ├── condition1/
│   │   ├── brain_1/
│   │   │   ├── logs/
│   │   │   ├── models/
│   │   │   └── recordings/
│   │   └── brain_2/
│   └── condition2/
└── results/
    ├── analysis_data/
    └── visualizations/
```

### Performance Considerations

#### Optimization Strategies
- **Vectorized Environments**: Multiple parallel environments per process
- **GPU Utilization**: Efficient GPU memory management
- **I/O Optimization**: Minimal disk writes during training
- **Process Pooling**: Reuse of worker processes

#### Scaling
- **Multi-GPU**: Distribute tasks across multiple GPUs
- **Memory Estimation**: Automatic task sizing based on available resources
- **Load Balancing**: Dynamic task allocation based on device capacity

### Integration Points

#### External Dependencies
- **Stable-Baselines3**: Core RL algorithms
- **ML-Agents**: Unity environment interface
- **Gymnasium**: Environment API standard
- **PettingZoo**: Multi-agent environment support
- **PyTorch**: Neural network backend

#### Unity Communication
- **Side Channels**: Logging and parameter passing
- **Command Line Args**: Environment configuration
- **Port Management**: Network communication setup

This framework provides a comprehensive system for conducting large-scale embodied AI experiments with proper resource management, parallel execution, and scientific rigor.

## Random Seeding in NETTs

### Training

When training, the agent number (brain number) is used as the seed for all random number generators. For example, if `brain.num_brains==5`, there would be 5 agents each with the seeds 1, 2, 3, 4, and 5 respectively. This seed is used in generating random numbers that are consistent between runs. As long as the seed is the same, all random aspects of the training will be the same, assuming that the length of training is the same. There are two main sources of randomness during training:

1. Model seed - The subset of steps assigned to each mini-batch from the replay buffer.

    - This is defined during intialization of the model in the Brain component.
    - For example, if the replay buffer has 4 steps and the mini-batch size is 2, the mini-batches could be [1,2] and [3,4], [1,3] and [2,4], [1,4] and [2,3], or any of these with their orders altered. The seed determines which of these combinations is used.

2. Environment seed - The initial placement of agents in the environment at the start of each episode.

    - This is defined during initialization of the Unity Environment in the Environment component.
    - During Training, the position and orientation of the agent is randomized at the start of each episode.  

### Testing

During Testing, nothing is random, so seeding does not impact the results.
