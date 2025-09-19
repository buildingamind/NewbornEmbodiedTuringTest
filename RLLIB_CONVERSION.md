# RLlib Conversion Guide

This document explains the conversion from stable-baselines3 (SB3) to Ray RLlib in the NETT framework.

## Overview

The NETT framework has been converted from using stable-baselines3 and sb3-contrib to using Ray RLlib as the underlying reinforcement learning library. This conversion maintains full backward compatibility with the existing API.

## What Changed

### Dependencies
- **Removed**: `stable-baselines3[extra_no_roms]>=2.0.0`, `sb3-contrib>=2.0.0`
- **Added**: `ray[rllib]>=2.30.0`, `gymnasium>=0.26.0`

### Core Components

#### 1. Algorithms
RLlib equivalents are now used for all algorithms:
- **PPO**: Ray RLlib PPO (replaces SB3 PPO)
- **SAC**: Ray RLlib SAC (replaces SB3 SAC)  
- **DQN**: Ray RLlib DQN (replaces SB3 DQN)
- **A2C**: Ray RLlib A2C (replaces SB3 A2C)

#### 2. Feature Extractors/Encoders
All custom encoders now inherit from the RLlib-compatible `BaseFeaturesExtractor`:
- Maintains the same interface as SB3's `BaseFeaturesExtractor`
- Works seamlessly with RLlib's model architecture
- Supports all existing encoders: sam, vit, resnet10, resnet18, etc.

#### 3. Policies
Policy strings remain the same but map to RLlib equivalents:
- `CnnPolicy`, `MlpPolicy`, `MultiInputPolicy`, etc.

#### 4. Training and Testing
The `Brain.train()` and `Brain.test()` methods maintain the same interface:
- Same parameters and return types
- Compatible with existing training loops
- Automatic RLlib algorithm configuration

## API Compatibility

### Brain Instantiation (No Changes Required)
```python
# This code works exactly the same after conversion
brain = Brain(
    policy="CnnPolicy",
    algorithm="PPO", 
    encoder="small",
    batch_size=512,
    buffer_size=2048,
    learning_rate=3e-4
)
```

### List Functions (No Changes Required)
```python
# These still work exactly the same
nett.list_algorithms()  # ['PPO', 'SAC', 'DQN', 'A2C']
nett.list_policies()    # ['CnnPolicy', 'MlpPolicy', ...]
nett.list_encoders()    # ['small', 'medium', 'large', 'sam', ...]
```

### Training (No Changes Required)
```python
# Training interface remains unchanged
brain.train(envs, config)
brain.test(envs, config)
```

## Technical Implementation

### RLlib Compatibility Layer
A comprehensive compatibility layer (`src/nett/brain/utils/rllib_compat.py`) provides:

1. **BaseAlgorithm Wrapper**: Wraps RLlib algorithms to provide SB3-like interface
2. **BaseFeaturesExtractor**: Base class for custom encoders compatible with RLlib
3. **BasePolicy**: Policy base class for compatibility
4. **Utility Functions**: Helper functions like `make_vec_env`, `get_flattened_obs_dim`

### Callback System
Callbacks have been adapted for RLlib:
- Maintains the same interface for loading bars and memory tracking
- Intrinsic rewards integration adapted for RLlib training loop
- Video recording and other utilities unchanged

### Model Architecture
Custom models are automatically wrapped and integrated with RLlib's model architecture:
- Feature extractors work seamlessly with RLlib trainers
- Value and policy heads are automatically created
- Custom encoder arguments are preserved

## Migration Notes

### For End Users
**No changes required!** The existing API is fully preserved. Your existing NETT code will work without modification.

### For Developers
- All SB3 imports have been replaced with RLlib equivalents
- Feature extractors now inherit from the compatibility layer's `BaseFeaturesExtractor`
- Algorithm instantiation goes through the RLlib wrapper classes
- Callback system uses the compatibility layer's base classes

## Performance and Features

### Advantages of RLlib
- **Scalability**: Better support for distributed training
- **Algorithm Diversity**: Access to more state-of-the-art algorithms
- **Integration**: Better integration with Ray ecosystem for hyperparameter tuning
- **Performance**: Optimized for modern hardware and distributed systems

### Maintained Features
- All custom encoders (SAM, ViT, ResNet variants, etc.)
- Intrinsic reward systems
- Custom policy architectures
- Hyperparameter logging and checkpointing
- Video recording and analysis tools

## Testing

Run the compatibility test to verify the conversion:
```bash
python test_rllib_conversion.py
```

This test validates that all core APIs work as expected with the new RLlib backend.

## Troubleshooting

### Import Errors
If you encounter import errors, ensure you have the new dependencies:
```bash
pip install ray[rllib]>=2.30.0 gymnasium>=0.26.0
```

### Performance Differences
Some performance characteristics may differ due to RLlib's different optimization strategies. Monitor your training metrics and adjust hyperparameters if needed.

### Algorithm-Specific Issues
If you encounter issues with specific algorithms, check the RLlib documentation for algorithm-specific configurations that might need adjustment.