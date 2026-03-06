# CLAUDE.md

Project context for Claude Code and other LLM assistants.

## Project Identity

- **Name**: Newborn Embodied Turing Test (NETT)
- **Package**: `nett-benchmarks` (PyPI)
- **Repo**: `buildingamind/NewbornEmbodiedTuringTest`
- **Purpose**: Benchmark virtual agents in controlled-rearing conditions, comparing AI learning with newborn animal (chick) behavior in equivalent experimental setups
- **Python**: Exactly 3.10.12
- **Not compatible** with Apple Silicon (mlagents dependency)

## Directory Layout

```
src/nett/                  # Main package source
├── nett.py                # NETT orchestrator class — primary entry point
├── __init__.py            # Public API re-exports (list_* functions, NETT, analyze, map_trajectories)
├── schema.json            # JSON Schema for config validation (all config options defined here)
├── _version.py            # Package version
├── brain/                 # Learning module
│   ├── brain.py           # Brain class: train(), test(), calc_iterations()
│   ├── encoders/          # Vision encoders (BaseFeaturesExtractor subclasses)
│   ├── rewards/           # Intrinsic reward functions (BaseReward subclasses)
│   └── utils/
│       ├── validate.py    # Registry pattern: _getMapping(), _getValidator() factories
│       ├── callbacks.py   # SB3 training callbacks
│       └── performance.py # Performance metrics
├── body/                  # Sensory interface module
│   ├── body.py            # Body class: embed(), validate_env()
│   ├── wrappers/          # gym.Wrapper subclasses (binocular, dvs, retina, video, etc.)
│   └── utils/
│       └── validate.py    # Wrapper validation
├── environment/           # Unity simulation module
│   ├── environment.py     # Environment class: load(), adjust_to_agent()
│   └── utils/
│       ├── design.py      # get_experiment_design() — reads conditions from Unity executables
│       ├── ports.py       # random_port() for Unity worker processes
│       ├── wrappers.py    # GymWrapper, ZooWrapper for single/multi-agent
│       └── validate.py    # Executable path and conditions validation
├── analysis/              # Post-training analysis (standalone, usable without NETT)
│   ├── analysis.py        # dst(), timelapse(), analyze()
│   ├── feature_visualization.py
│   ├── activation_maximization.py
│   ├── trajectory.py      # map_trajectories()
│   ├── tSNE.py            # generate_tSNEs()
│   └── utils/             # R script wrappers (merge, train_viz, test_viz)
└── utils/                 # Orchestration internals
    ├── task.py            # TaskConfig, Agent, Task, run_task()
    ├── executor.py        # Singleton ProcessPoolExecutor wrapper
    ├── tasklist.py        # TaskList management
    ├── memory.py          # MemoryManager — GPU VRAM tracking and allocation
    ├── loading_bar_queue.py
    └── validate.py        # Config validation against schema.json

docs/
├── dev/                   # Developer documentation
│   ├── full_architecture.md    # Full mermaid architecture diagram
│   ├── simple_architecture.md  # Simplified mermaid diagram
│   ├── random-seeding.md       # Seeding strategy docs
│   ├── developer-notes2.md
│   └── documentation.md
└── source/                # Sphinx docs source (RST + Markdown)

examples/
├── example1.py            # Basic usage example
├── configs/
│   ├── nett_config_min.yaml   # Minimal config (just name + executable_path)
│   └── nett_config_max.yaml   # Full config with all options
└── notebooks/

tests/                     # Test directory (scaffolded, currently empty)
scripts/publish.sh         # PyPI publish script
```

## Architecture

```
NETT (orchestrator)
 ├── Reads YAML/JSON configs, validates against schema.json
 ├── Creates TaskConfig objects (one per brain × condition combination)
 ├── Manages GPU memory via MemoryManager
 └── Submits tasks to Executor (ProcessPoolExecutor)

Executor → run_task(task)
 ├── Body.embed(env, config) → applies wrapper chain to Environment
 │    └── Environment.load() → starts Unity executable as gym.Env
 ├── Brain.train(envs, config) → SB3 algorithm training loop
 └── Brain.test(envs, config) → model evaluation

Analysis (post-training, standalone)
 ├── analyze() → R scripts for train/test visualization
 ├── map_trajectories() → agent trajectory plots
 ├── feature_visualization() → CNN feature maps
 └── generate_tSNEs() → t-SNE of model parameters
```

See `docs/dev/full_architecture.md` for the complete mermaid diagram and `docs/dev/simple_architecture.md` for the simplified version.

## Key Patterns

### Config-Driven Experiments
All experiments defined via YAML/JSON validated against `src/nett/schema.json`. Only `name` and `environment.executable_path` are required. Configs are saved to output directories for reproducibility.

### Registry Pattern (Component Validation)
Brain components (encoders, algorithms, policies, rewards) and Body wrappers use a registry pattern defined in `src/nett/brain/utils/validate.py`:

- `_getMapping(sources, override)` — Scans module exports for capitalized class names, plus manual overrides, producing a `dict[str, type]`
- `_getValidator(label, baseclass, mapping)` — Returns a validator function that accepts either a string key (looked up in mapping) or a callable class (passed through)
- Users can pass string names ("PPO", "small") or custom class types directly

### Registered Components

| Category    | Names |
|-------------|-------|
| Algorithms  | A2C, ARS, DDPG, DQN, HER, MaskablePPO, PPO, QRDQN, RecurrentPPO, SAC, TD3, TQC, TRPO |
| Encoders    | small, medium, large, CNNLSTM, DinoV1, DinoV2, FrozenSimCLR, SegmentAnything, SimpleViT, ViT |
| Policies    | CnnPolicy, CnnLstmPolicy, MlpPolicy, MlpLstmPolicy, MultiInputPolicy, MultiInputLstmPolicy |
| Rewards     | closeness, completeness, closeness,completeness, unsupervised, disagreement, e3b, fabric, icm, ngu, pseudocounts, re3, ride, rnd |
| Wrappers    | binocular, dvs, multiobs, retina, video |

### Wrapper Composition
Body uses the decorator pattern with `gym.Wrapper` subclasses applied in a chain. Wrappers live in `src/nett/body/wrappers/`. The `Body.embed()` method applies wrappers, vectorizes with `SubprocVecEnv`, and adds monitoring.

### Context Managers
`Body` and `Executor` implement `__enter__`/`__exit__` for resource cleanup (closing Unity environments, shutting down process pools).

### GPU Memory Management
`MemoryManager` (in `src/nett/utils/memory.py`) tracks VRAM per device, auto-calculates memory needed per task, and queues tasks when GPUs are full. Uses `nvidia-ml-py`.

### Seed Diversification
Task seeds use: `(brain_id * 7919) % (2**31 - 1)` to avoid correlated failures from sequential seeds. See `docs/dev/random-seeding.md`.

### Parallel Execution
`Executor` is a singleton wrapping `ProcessPoolExecutor`. Tasks are submitted per brain × condition combination and run across available GPUs.

## Build & Development

```bash
# Install in development mode
pip install -e .

# Install with specific constraints (required for gym==0.21 compatibility)
pip install setuptools==65.5.0 pip==21 wheel==0.38.4

# Build Sphinx documentation
sphinx-build -M html docs/source/ docs/build/

# Lint (GitHub Actions workflow at .github/workflows/lint.yml)
# Docs deployment (GitHub Actions workflow at .github/workflows/docs.yml)
```

## Code Conventions

- **Docstrings**: Google-style (Napoleon extension for Sphinx)
- **Logging**: `logging.getLogger("nett")` base logger; modules use `logging.getLogger("nett.ModuleName")`; format: `[name] LEVEL: message`
- **Imports**: Public API re-exported through `src/nett/__init__.py`; internal modules import from each other directly
- **Type hints**: Used on public method signatures
- **File permissions**: `__init__.py` manages `/tmp/ml-agents-binaries` directory permissions for shared lab environments

## Common Extension Tasks

### Adding a New Encoder
1. Create `src/nett/brain/encoders/my_encoder.py` with a class subclassing `BaseFeaturesExtractor`
2. Import it in `src/nett/brain/encoders/__init__.py`
3. The registry auto-discovers it via `_getMapping()` (scans for capitalized names in encoder modules)
4. Optionally add a string alias in the `override` dict passed to `_getMapping()`

### Adding a New Reward Function
1. Create `src/nett/brain/rewards/my_reward.py` with a class subclassing `BaseReward`
2. Import it in `src/nett/brain/rewards/__init__.py`
3. Auto-discovered by registry, or add alias

### Adding a New Wrapper
1. Create `src/nett/body/wrappers/my_wrapper.py` with a class subclassing `gym.Wrapper`
2. Import it in `src/nett/body/wrappers/__init__.py`
3. Add the string name → class entry to the wrapper mapping in `src/nett/body/utils/validate.py`

### Adding a New Algorithm
Pass any Stable-Baselines3 or sb3-contrib algorithm class directly, or add it to the algorithm mapping in `src/nett/brain/utils/validate.py`.

## Important Constraints

- **Python 3.10.12** exactly — other versions may cause dependency conflicts
- **No Apple Silicon** — `mlagents==1.0.0` is incompatible with M1/M2/M3 chips
- **Custom mlagents fork** — Uses `mlagents_envs` from `Zach-Attach/ml-agents` (gym-plus branch)
- **Unity executables required** — Environments need pre-built Unity binaries (not included in the package)
- **R required for some analysis** — `analyze()` calls R scripts that need `tidyverse`, `argparse`, `scales`
- **`if __name__ == '__main__':` guard** — Required in user scripts because `SubprocVecEnv` spawns subprocesses

## Output Structure

```
output_dir/
├── experiment_name/
│   ├── config.yaml              # Saved config for reproducibility
│   ├── condition1/
│   │   ├── brain_1/
│   │   │   ├── logs/            # TensorBoard logs
│   │   │   ├── models/          # Saved checkpoints (.zip)
│   │   │   └── recordings/      # Episode recordings
│   │   └── brain_2/
│   └── condition2/
└── results/                     # Analysis outputs
```

## Maintenance

When adding new components (encoders, rewards, wrappers, algorithms), update the "Registered Components" table above and the corresponding section in `llms.txt`.
