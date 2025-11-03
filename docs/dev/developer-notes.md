# Developer Notes

## Table of Contents

- [Introduction](#introduction)
- [Online Documentation Website](#online-documentation-website)
- [Source Code Organization](#source-code-organization)
- [NETT Architecture and Development](#nett-architecture-and-development)
- [How the Code Runs: Execution Flow](#how-the-code-runs-execution-flow)
- [Random Seeding in NETTs](#random-seeding-in-netts)
- [Extending the Codebase](#extending-the-codebase)
- [Development Best Practices](#development-best-practices)
- [Key Dependencies](#key-dependencies)
- [File Structure Generated During Execution](#file-structure-generated-during-execution)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Contributing Guidelines](#contributing-guidelines)
- [Version Control](#version-control)

## Introduction

This document serves as a comprehensive guide to the development process and architecture of the NETT (Newborn Embodied Turing Test) toolkit. It provides an overview of the toolkit's structure, development workflow, and key implementation details for developers working on or extending the codebase.

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

## Source Code Organization

The NETT toolkit source code is located in `/src/nett/` and follows a modular architecture organized into the following key components:

```
src/nett/
├── __init__.py              # Package initialization and public API
├── nett.py                  # Main NETT orchestrator class
├── schema.json              # JSON schema for configuration validation
├── _version.py              # Version information
├── brain/                   # Neural network components
│   ├── brain.py            # Brain class (training algorithms)
│   ├── encoders/           # Feature extraction networks
│   │   ├── <encoder>.py        # each files defines a specific encoder
│   │   └── disembodied_models/ # Pre-trained model architectures
│   ├── rewards/            # Intrinsic motivation and reward functions
│   │   ├── e3b.py           # E3B reward
│   │   ├── icm.py           # Intrinsic Curiosity Module
│   │   ├── pseudo_counts.py # Pseudo-count exploration
│   │   └── ride.py          # RIDE reward
│   └── utils/               # Brain utilities and validation
│      ├── callbacks.py     # Training callbacks
│      └── validate.py      # Brain configuration validation
├── body/                    # Sensory processing components
│   ├── body.py             # Body class (environment wrappers)
│   ├── wrappers/           # Observation transformation wrappers
│   │   └── <wrapper>.py    # each file defines a specific wrapper
│   └── utils/              # Body utilities and validation
│      └── validate.py      # Wrapper validation
├── environment/            # Unity environment integration
│   ├── environment.py      # Environment class
│   └── utils/              # Environment utilities
│       ├── design.py       # Experiment design parsing
│       ├── logger.py       # Unity logging integration
│       └── ports.py        # Network port management
├── utils/                   # Core system utilities
│   ├── executor.py         # Process pool executor management
│   ├── loading_bar_queue.py # Progress tracking
│   ├── memory.py           # GPU memory management
│   ├── task.py             # Task execution logic
│   ├── tasklist.py         # Task list management
│   └── validate.py         # Configuration validation
└── analysis/               # Post-experiment analysis
    ├── analysis.py         # Main analysis interface
    ├── ChickData/          # Biological comparison data
    └── utils/              # Analysis utilities
        ├── merge.py        # Data merging
        ├── test_viz.py     # Test visualization
        └── train_viz.py    # Training visualization
```

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

## How the Code Runs: Execution Flow

Understanding how NETT executes experiments is crucial for developers. Here's the complete execution flow from configuration to results:

### 1. Entry Point and Initialization

Users create experiments by instantiating the `NETT` class with configuration dictionaries or file paths:

```python
from nett import NETT

# Configuration can be dict, JSON, or YAML
configs = [
    {
        "name": "Experiment1",
        "episodes": {"train": 5000, "test": 100},
        "steps_per_episode": 200,
        "num_brains": 5,
        "brain": {"policy": "CnnPolicy", "algorithm": "PPO", "encoder": "small"},
        "body": {"wrappers": ["dvs"]},
        "environment": {"executable_path": "path/to/unity.x86_64"}
    }
]

nett = NETT(configs)
nett.run(output_path="./results", devices=[0,1], num_threads=32)
```

**Initialization Flow (`__init__`)**:
1. Load the JSON schema from `schema.json`
2. Validate each configuration against the schema using `validate_config()`
3. Store validated configurations in `self.configs`

### 2. Run Method Execution

The `run()` method orchestrates the entire experiment:

**Step 2.1: Setup Phase**
- Resolve output directory path
- Calculate thread allocation based on number of test configurations
- Initialize task sheet (`dict[Future, TaskConfig]`) and waitlist
- Initialize `MemoryManager` for GPU tracking
- Validate and store device list
- Query free memory on each GPU device
- Initialize `Executor` (ProcessPoolExecutor) with loading bar queue

**Step 2.2: Configuration Processing**
For each configuration, call `single_run()` which:
- Validates episode structure (must contain 'train' and/or 'test')
- Creates output directory: `output_path/name/`
- Saves a copy of the configuration as `config.yaml`

**Step 2.3: Component Initialization**
Each configuration creates three base components:
- `Brain(**brain_config)` - Neural network and training algorithm
- `Body(**body_config)` - Sensory processing and wrappers
- `Environment(**environment_config)` - Unity executable interface

**Step 2.4: Brain Calculation**
Call `brain.calc_iterations()` to compute:
- Total training steps across all brains
- Test iterations per condition
- Parallel environment counts

**Step 2.5: Environment Adjustment**
Call `env.adjust_to_agent()` to configure Unity parameters:
- Steps per episode
- Reward function type
- Visual settings (binocular, resolution, etc.)

**Step 2.6: Memory Estimation**
Call `_calculate_task_memory()` to estimate GPU memory required:
- If `task_memory="auto"`, run a validation pass to measure actual memory usage
- Uses `MemoryManager.get_free_memory()` before and after environment creation
- Memory calculation accounts for:
  - Environment complexity
  - Model size (encoder + policy networks)
  - Batch size and buffer size
  - Number of parallel environments

**Step 2.7: Task Generation**
Create a `TaskList` containing all task combinations:
- Cartesian product: `num_brains × conditions`
- Each task contains: Brain, Body, Environment, brain_id, condition, output_dir, modes
- Example: 5 brains × 3 conditions = 15 tasks

**Step 2.8: Task Validation**
For non-multiagent environments:
- Submit `validate_tasklist()` to executor
- Validates environment compatibility for each condition (once per condition)
- Checks if Body wrappers are compatible with Environment observations
- Waits for validation to complete before proceeding

**Step 2.9: Task Assignment**
For each task in the tasklist:
- Call `_assign_task()` which:
  - Finds GPU with most free memory
  - If sufficient memory available:
    - Assign task to that GPU
    - Submit `run_task()` to executor (returns Future)
    - Store Future and TaskConfig in `task_sheet`
    - Deduct memory from free memory pool
  - If insufficient memory:
    - Add task to `waitlist`

**Step 2.10: Task Waiting**
Call `task_waiter()` which:
- Uses `as_completed()` to monitor task futures
- When a task completes:
  - Free up its GPU memory
  - Check waitlist for tasks that can now fit
  - Assign waiting tasks to freed resources
- Continues until all tasks complete

### 3. Task Execution (`run_task`)

Each task submitted to the executor runs in its own process:

**Step 3.1: Setup**
- Extract `config` (TaskConfig) and `agent` (Agent with Brain, Body, Environment)
- Create log directory: `output_path/name/condition/brain_N/logs/`

**Step 3.2: Mode Iteration**
For each mode in `config.modes` (typically ['train', 'test']):
- Set `config.current_mode = mode`

**Step 3.3: Body Embedding**
Call `agent.body.embed(agent.env, config)` (context manager) which:
- Creates vectorized environments using `SubprocVecEnv` or `DummyVecEnv`
- For each parallel environment:
  - Call `agent.env.load(config, validation_mode=False, seed=brain_id+i)`
  - Apply body wrappers (DVS, binocular, etc.)
  - Add Monitor wrapper for logging
  - Add RecordVideo wrapper if recording enabled
- Yields the vectorized environment to the brain

**Step 3.4: Brain Training/Testing**
Call `agent.brain.train(body_interface, config)` or `agent.brain.test(body_interface, config)`:

**Training (`brain.train`)**:
1. Set device: `torch.device(f"cuda:{config.device}")`
2. Create policy with encoder and policy network
3. Initialize algorithm (PPO, DQN, etc.) with:
   - Policy
   - Vectorized environment
   - Learning rate, batch size, buffer size
   - Checkpoint callbacks
   - Custom reward function if specified
4. Set random seed: `model.set_random_seed(config.brain_id)`
5. Optionally freeze encoder if `train_encoder=False`
6. Call `model.learn(total_timesteps)` - Stable-Baselines3 training loop
7. Save model, policy, and feature extractor to `models/` directory
8. Close environments

**Testing (`brain.test`)**:
1. Load trained model from `models/latest_model.zip`
2. Set model to evaluation mode
3. For each test episode:
   - Reset environment
   - Run episode with learned policy (no exploration)
   - Log observations, actions, rewards
   - Update progress queue
4. Save test results to `logs/` directory
5. Close environments

**Step 3.5: Cleanup**
- Context manager exits, closing all Unity environment connections
- Progress bar updated
- Log completion

### 4. Environment Management

**Unity Communication**:
- Uses `mlagents_envs.UnityEnvironment` for communication
- Each environment instance uses a unique network port (auto-assigned)
- Side channels pass logging and parameters to Unity
- Command-line arguments configure recording and display settings

**Port Management**:
- `random_port()` finds available ports to avoid conflicts
- Each parallel environment and each task gets unique port
- Handles `UnityWorkerInUseException` by retrying with new port

**Recording**:
- Chamber-wide recording: Unity captures entire environment view
- Agent perspective recording: Gymnasium RecordVideo wrapper
- Recording controlled by episode ranges: `"start:stop:step"` format
- Videos saved to `recordings/` subdirectory

### 5. Progress Tracking

**Loading Bar System**:
- `LoadingBarQueue` uses multiprocessing queue for cross-process communication
- Each task updates progress after every episode
- Main process runs `updateLoadingBars()` in separate thread
- Displays progress bars for all concurrent experiments
- Total steps calculated: `steps_per_episode × num_brains × (train_episodes × conditions + test_episodes × test_iterations)`

### 6. Post-Experiment Analysis

After all tasks complete, users can run analysis:

```python
from nett import analyze

analyze("experiment_name", run_dir="./results/Experiment1", output_dir="./results/analysis")
```

**Analysis Pipeline**:
1. **Merge** (`merge.py`): Combines CSV logs from all brains and conditions
2. **Train Visualization** (`train_viz.py`): Generates training curves (reward, loss, entropy)
3. **Test Visualization** (`test_viz.py`): Creates test performance plots, compares with chick data
4. Saves visualizations and combined datasets to `results/` directory

### 7. Key Design Patterns

**Singleton Pattern**:
- `MemoryManager` and `Executor` use singleton pattern
- Ensures single GPU monitor and process pool across entire run

**Context Managers**:
- `MemoryManager`, `Executor`, and `Body.embed()` use context managers
- Guarantees proper cleanup of GPU resources and Unity processes
- Prevents resource leaks even on exceptions

**Process Pool Execution**:
- Each task runs in isolated process (subprocess)
- Prevents GPU memory leaks between tasks
- Enables true parallelism across multiple GPUs
- `initializer` mutes stdout in worker processes when not verbose

**Vectorized Environments**:
- Multiple Unity instances per training task
- `SubprocVecEnv` runs each environment in separate subprocess
- Increases sample efficiency during training
- Test mode typically uses single environment for reproducibility

**Seeding Strategy**:
- Brain ID used as master seed
- Seeds both model initialization and environment randomization
- Ensures reproducibility across runs with same brain_id
- Test episodes have no randomization (deterministic placement)

### 8. Configuration Schema

All configurations validated against `schema.json`:
- Required fields: `name`, `environment.executable_path`
- Optional with defaults: `episodes`, `brain`, `body`, `steps_per_episode`, `num_brains`
- Supports enums for algorithms, encoders, policies, rewards, wrappers
- Custom validation: batch_size ≤ buffer_size
- Extensible: custom encoder/policy/reward classes accepted

### 9. Error Handling

**Common Errors and Recovery**:
- **SubprocVecEnv Failure**: Requires `if __name__ == '__main__':` guard in scripts
- **Port Conflicts**: Automatic retry with new random port
- **GPU OOM**: Tasks placed in waitlist, assigned when memory freed
- **Unity Connection**: Subprocess retries, eventually raises UnityWorkerInUseException
- **Validation Failure**: Stops execution before submitting tasks, shows clear error message

**Logging Strategy**:
- Package logger: `logging.getLogger("nett")`
- Task-specific loggers: `f"{experiment_name}-{condition}-{brain_id}"`
- Hierarchical logging allows filtering by experiment, condition, or brain
- Exception logging with full stack traces for debugging

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

## Extending the Codebase

NETT is designed to be extensible. Here's how to add new components:

### Adding a New Encoder

1. Create a new file in `src/nett/brain/encoders/` (e.g., `my_encoder.py`)
2. Implement a class inheriting from `BaseFeaturesExtractor`:

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class MyEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, **kwargs):
        super().__init__(observation_space, features_dim)
        # Define your network architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # ... more layers
        )
        self._features_dim = features_dim
    
    def forward(self, observations):
        return self.cnn(observations)
```

3. Register in `src/nett/brain/encoders/__init__.py`:
```python
from .my_encoder import MyEncoder
```

4. Add to validation list in `src/nett/brain/utils/validate.py`
5. Update schema in `src/nett/schema.json` to include new encoder name

### Adding a New Wrapper

1. Create a new file in `src/nett/body/wrappers/` (e.g., `my_wrapper.py`)
2. Implement a class inheriting from `gym.Wrapper`:

```python
import gymnasium as gym
import numpy as np

class MyWrapper(gym.ObservationWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        # Modify observation space if needed
        self.observation_space = gym.spaces.Box(...)
    
    def observation(self, obs):
        # Transform observation
        return transformed_obs
```

3. Register in `src/nett/body/wrappers/__init__.py`
4. Add to wrapper list in `src/nett/body/utils/validate.py`
5. Update schema to include new wrapper name

### Adding a New Reward Function

1. Create a new file in `src/nett/brain/rewards/` (e.g., `my_reward.py`)
2. Implement a class inheriting from `rllte.common.prototype.BaseReward`:

```python
from rllte.common.prototype import BaseReward
import torch

class MyReward(BaseReward):
    def __init__(self, observation_space, action_space, device, **kwargs):
        super().__init__(observation_space, action_space, device)
        # Initialize reward components
    
    def compute_irs(self, samples, step=0):
        # Compute intrinsic reward
        return intrinsic_rewards
```

3. Register in `src/nett/brain/rewards/__init__.py`
4. Add to rewards list in `src/nett/brain/utils/validate.py`
5. Update schema enum

### Adding a New Algorithm

NETT uses Stable-Baselines3 algorithms. To add a custom algorithm:

1. Implement a class inheriting from `BaseAlgorithm` (or `OnPolicyAlgorithm`/`OffPolicyAlgorithm`)
2. Register in `src/nett/brain/utils/validate.py`
3. Update schema enum
4. Ensure algorithm is compatible with your policy and encoder choices

### Testing Custom Components

Always test custom components in isolation before integrating:

```python
# Test encoder
from nett.brain.encoders import MyEncoder
from gymnasium import spaces

obs_space = spaces.Box(low=0, high=255, shape=(3, 84, 84))
encoder = MyEncoder(obs_space, features_dim=512)

# Test wrapper
from nett.body.wrappers import MyWrapper
env = gym.make('CartPole-v1')
wrapped_env = MyWrapper(env)
obs, _ = wrapped_env.reset()
```

## Development Best Practices

### Code Organization

- **Modularity**: Keep Brain, Body, and Environment concerns separate
- **Type Hints**: Use Python type hints for all public methods
- **Documentation**: Follow Google-style docstrings
- **Validation**: Always validate inputs in public methods

### Testing

- Test custom components with minimal configurations first
- Use `validation_mode=True` for quick environment checks
- Monitor GPU memory usage during development
- Test with small `num_brains` and `episodes` values initially

### Performance Optimization

- **Memory Management**: Profile memory usage with `nvidia-smi`
- **Vectorization**: Use vectorized environments for training
- **Batch Sizes**: Tune batch_size and buffer_size for your GPU
- **Parallel Environments**: Balance between memory and sample efficiency

### Debugging

- Set `verbose=True` in `run()` to see detailed logs
- Check task-specific logs in `output_path/name/condition/brain_N/logs/`
- Use single brain and condition for debugging
- Enable checkpoint saving to recover from interruptions

### Common Pitfalls

1. **Missing `if __name__ == '__main__':`**: Required for `SubprocVecEnv`
2. **Incompatible Observation Spaces**: Ensure wrappers and encoders match
3. **Memory Leaks**: Always use context managers, close environments properly
4. **Port Conflicts**: NETT handles this, but manual Unity launches can conflict
5. **Unity Permissions**: Ensure executable has correct permissions (755)
6. **GPU Memory**: Don't oversubscribe - leave headroom for memory spikes

## Key Dependencies

Understanding the key external dependencies helps when debugging or extending:

### Core Dependencies

- **Stable-Baselines3**: RL algorithms (PPO, DQN, SAC, etc.)
  - Provides: `BaseAlgorithm`, `BasePolicy`, `BaseFeaturesExtractor`
  - Version compatibility important for reproducibility

- **ML-Agents**: Unity environment interface
  - Provides: `UnityEnvironment`, side channels, communication protocol
  - Must match Unity package version in executables

- **Gymnasium**: Environment API standard
  - Provides: `gym.Env`, `gym.Wrapper`, spaces
  - Successor to OpenAI Gym

- **PettingZoo**: Multi-agent environment support
  - Used for multiagent experiments
  - Provides parallel and AEC APIs

- **PyTorch**: Deep learning backend
  - All models are PyTorch `nn.Module`
  - GPU memory management via CUDA

- **SuperSuit**: Environment preprocessing wrappers
  - Used for frame stacking, vector environment utilities
  - Provides `ConcatVecEnv`, `SB3VecEnvWrapper`

### Analysis Dependencies

- **Matplotlib**: Plotting and visualization
- **Pandas**: Data manipulation
- **Scikit-learn**: PCA, tSNE for feature analysis
- **NumPy**: Numerical operations

### System Dependencies

- **pynvml**: NVIDIA GPU monitoring
  - Queries VRAM usage
  - Device discovery

- **subprocess**: Unity executable management
- **multiprocessing**: Task parallelization, progress tracking
- **concurrent.futures**: Process pool execution

## File Structure Generated During Execution

Understanding the output structure helps with result analysis:

```
output_path/
└── experiment_name/              # Experiment name from config
    ├── config.yaml               # Copy of configuration used
    ├── condition1/               # One directory per condition
    │   ├── brain_1/              # One directory per brain
    │   │   ├── logs/             # Training and testing logs
    │   │   │   ├── train.csv     # Training episode data
    │   │   │   ├── test.csv      # Test episode data
    │   │   │   └── progress.csv  # Algorithm progress (loss, entropy)
    │   │   ├── models/           # Saved models
    │   │   │   ├── latest_model.zip         # Full SB3 model
    │   │   │   ├── policy.pkl               # Policy network
    │   │   │   ├── feature_extractor.pth    # Encoder weights
    │   │   │   └── checkpoint_N.zip         # Checkpoints (if enabled)
    │   │   ├── recordings/       # Video recordings
    │   │   │   ├── agent/        # Agent perspective videos
    │   │   │   └── chamber/      # Full chamber videos (if enabled)
    │   │   └── monitor/          # Stable-Baselines3 Monitor logs
    │   │       └── brain_1_0.csv
    │   ├── brain_2/
    │   └── ...
    ├── condition2/
    └── results/                  # Analysis results (if run)
        ├── analysis_data/        # Merged CSV data
        ├── train_plots/          # Training visualizations
        └── test_plots/           # Test performance plots
```

## Troubleshooting Guide

### Installation Issues

**Problem**: `ImportError` for mlagents_envs
- **Solution**: Ensure ML-Agents is installed: `pip install mlagents-envs`

**Problem**: CUDA not available
- **Solution**: Install PyTorch with CUDA support matching your CUDA version

### Runtime Issues

**Problem**: "No jobs could be scheduled. Job size too large for GPUs."
- **Solution**: Reduce `n_parallel_envs` in Brain or set lower `task_memory`

**Problem**: SubprocVecEnv connection reset
- **Solution**: Add `if __name__ == '__main__':` guard to your script

**Problem**: Unity worker in use
- **Solution**: NETT auto-retries with new ports. If persists, manually kill Unity processes

**Problem**: Permission denied on executable
- **Solution**: Run `chmod 755 path/to/executable.x86_64`

### Performance Issues

**Problem**: Training very slow
- **Solution**: Increase `n_parallel_envs`, reduce `steps_per_episode`, or use faster encoder

**Problem**: GPU memory errors during training
- **Solution**: Reduce batch_size, buffer_size, or n_parallel_envs

**Problem**: Port exhaustion
- **Solution**: Reduce number of parallel tasks or restart to free ports

## Contributing Guidelines

When contributing to NETT:

1. **Branch Strategy**: Create feature branches from `main`
2. **Code Style**: Follow PEP 8, use `black` formatter
3. **Documentation**: Update docstrings and this guide for new features
4. **Testing**: Test with multiple configurations and GPUs
5. **Schema Updates**: Update `schema.json` for new configuration options
6. **Examples**: Provide example configs for new features
7. **Backwards Compatibility**: Maintain compatibility with existing configs when possible

## Version Control

- `_version.py`: Contains version string
- Semantic versioning: MAJOR.MINOR.PATCH
- Update for breaking changes (MAJOR), new features (MINOR), bug fixes (PATCH)
