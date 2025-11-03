# NETT Developer Notes

## Overview
The Newborn Embodied Turing Test (NETT) is a framework for training and testing AI agents in Unity-based environments. The codebase is structured around three main components: Brain (neural networks), Body (sensory processing), and Environment (Unity simulation).

## Core Architecture

### NETT Class (nett.py)
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

### Brain Component (`brain/brain.py`)
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

### Body Component (`body/body.py`)
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

### Environment Component (environment.py)
Interfaces with Unity executables and manages simulation parameters.

**Key Features:**
- **Unity Integration**: Uses ML-Agents toolkit for Unity communication
- **Condition Management**: Handles different experimental conditions
- **Recording**: Chamber-wide video recording
- **Multi-modal Support**: Supports both single and multi-agent environments

## Task Management System

### Task Structure
- **TaskConfig**: Configuration for individual tasks (brain_id, condition, paths, etc.)
- **Agent**: Combines Brain, Body, and Environment for a specific task
- **Task**: Complete unit of work with config and agent
- **TaskList**: Iterator over all tasks for a given experiment

### Execution Flow
1. **Configuration Validation**: Validate all input configs against schema
2. **Task Generation**: Create tasks for each brain × condition combination
3. **Memory Estimation**: Calculate required GPU memory per task
4. **Device Allocation**: Assign tasks to available GPU devices
5. **Parallel Execution**: Run tasks using ProcessPoolExecutor
6. **Result Collection**: Gather training logs and test results

## Memory Management

### MemoryManager Class
- **GPU Memory Tracking**: Monitors VRAM usage across devices
- **Dynamic Allocation**: Assigns tasks based on available memory
- **Memory Estimation**: Calculates required memory for each task type

### Memory Calculation
```python
# Auto memory calculation based on:
# - Environment complexity
# - Model size (encoder + policy)
# - Batch size and buffer size
# - Number of parallel environments
```

## Parallel Processing

### SubprocVecEnv Usage
The code extensively uses `SubprocVecEnv` for parallel environment execution:
- **Training**: Multiple environments running simultaneously
- **Testing**: Parallel evaluation across different conditions
- **Memory Efficiency**: Each subprocess manages its own memory space

### Process Management
- **ProcessPoolExecutor**: Manages worker processes for tasks
- **Future Objects**: Track task completion and results
- **Queue System**: Communication between main process and workers

## Data Flow

### Training Pipeline
```
Unity Environment → Body Wrappers → VecEnv → Brain Training → Model Checkpoints
                                         ↓
                                    Training Logs → CSV Files
```

### Testing Pipeline
```
Trained Model → Unity Environment → Body Wrappers → VecEnv → Evaluation → Test Results
```

## Configuration System

### Schema Validation
- **JSON Schema**: Validates all configuration parameters
- **Type Checking**: Ensures correct data types for all parameters
- **Default Values**: Provides sensible defaults for optional parameters

### Config Structure
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

## Error Handling

### Common Issues
- **SubprocVecEnv Errors**: Requires `if __name__ == '__main__':` guard
- **Unity Connection**: Port conflicts and environment startup failures
- **Memory Issues**: GPU OOM errors and memory estimation failures
- **File Permissions**: Unity executable and temp directory permissions

### Recovery Mechanisms
- **Port Management**: Automatic port selection for Unity environments
- **Memory Fallback**: Dynamic memory allocation adjustment
- **Task Retry**: Failed task rescheduling
- **Graceful Shutdown**: Proper environment cleanup

## Analysis System

### R Integration
The framework includes R scripts for post-training analysis:
- **NETT_merge_csvs.R**: Combines training/testing logs
- **NETT_train_viz.R**: Generates training performance plots
- **NETT_test_viz.R**: Creates test results visualizations

### Output Structure
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

## Performance Considerations

### Optimization Strategies
- **Vectorized Environments**: Multiple parallel environments per process
- **GPU Utilization**: Efficient GPU memory management
- **I/O Optimization**: Minimal disk writes during training
- **Process Pooling**: Reuse of worker processes

### Scaling
- **Multi-GPU**: Distribute tasks across multiple GPUs
- **Memory Estimation**: Automatic task sizing based on available resources
- **Load Balancing**: Dynamic task allocation based on device capacity

## Integration Points

### External Dependencies
- **Stable-Baselines3**: Core RL algorithms
- **ML-Agents**: Unity environment interface
- **Gymnasium**: Environment API standard
- **PettingZoo**: Multi-agent environment support
- **PyTorch**: Neural network backend

### Unity Communication
- **Side Channels**: Logging and parameter passing
- **Command Line Args**: Environment configuration
- **Port Management**: Network communication setup

This framework provides a comprehensive system for conducting large-scale embodied AI experiments with proper resource management, parallel execution, and scientific rigor.