# Development Notes

## Test Commands

Run the fast `src_isaac` unit suite:

```bash
VIRTUAL_ENV=/path/to/venv uv run --active --project src_isaac \
  pytest src_isaac/tests
```

Current expected result:

```text
155 passed, 24 deselected
```

Run a smoke training job:

```bash
VIRTUAL_ENV=/path/to/venv uv run --active --project src_isaac \
  python src_isaac/examples/run_smoke.py \
  --config src_isaac/examples/smoke.yaml \
  --output /path/to/output
```

## Design Rules

- Prefer Isaac Lab / Isaac Sim for simulation, cameras, vectorization, and
  recording.
- Prefer skrl trainer hooks over SB3 callback emulation.
- Keep heavy model dependencies optional and lazy.
- Remove Unity-only surfaces instead of carrying confusing compatibility shims.
- Use `register_encoder` and `register_reward` for user-provided extensions.
- Keep output paths vectorization-safe with explicit env/brain metadata.

## Useful Entry Points

- `nett_skrl/nett.py`: top-level config orchestration.
- `nett_skrl/runtime/task.py`: immutable task/run config.
- `nett_skrl/runtime/isaac_mode_runner.py`: Isaac Sim per-mode subprocess runner.
- `nett_skrl/runtime/memory.py`: NVML-backed GPU memory accounting.
- `nett_skrl/environment/environment.py`: config bridge into Isaac Lab.
- `nett_skrl/brain/brain.py`: thin public Brain facade.
- `nett_skrl/brain/agent_factory.py`: skrl agent/model/memory construction.
- `nett_skrl/brain/trainer.py`: NETT wrapper around skrl trainers.
- `nett_skrl/brain/env_adapter.py`: env observation/action bridge.
- `nett_skrl/brain/env_wrappers.py`: skrl rollout wrappers such as intrinsic reward injection.
- `nett_skrl/recording/export.py`: PNG sequence to MP4 export.

## Adding A Native Encoder

1. Implement a `nett_skrl.brain.encoders.NETTFeatureExtractor` subclass.
2. Accept Isaac HWC image observations and flattened skrl observations.
3. Register it with `register_encoder(...)` if it is not built in.
4. Add tests for HWC and flat observations.

External encoders can be registered without editing package code:

```python
from nett_skrl.brain.registry import register_encoder

register_encoder("MyEncoder", MyEncoder)
```

## Adding An Intrinsic Reward

Implement the adapter-compatible interface:

```python
class MyReward:
    def watch(self, observations, actions, rewards, terminated, truncated, next_observations):
        ...

    def compute(self, **samples):
        ...

    def update(self, samples=None):
        ...
```

Then register it:

```python
from nett_skrl.brain.registry import register_reward

register_reward("MyReward", MyReward)
```

Intrinsic rewards are added to extrinsic env rewards before skrl transition
storage.
