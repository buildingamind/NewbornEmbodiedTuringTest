"""``schema.json``'s declared defaults must equal the Python defaults that govern.

WHY THIS TEST EXISTS. ``jsonschema`` does not apply defaults, so a value in
``schema.json`` never reaches a run — the Python default does. That makes schema
defaults pure documentation, and undetectable documentation at that: nothing errors,
nothing warns, and the file *looks* authoritative. Two real defects have come out of
exactly this gap:

* ``locomotion`` — the schema declared ``wheeled`` while the Python default was
  ``kinematic``, so every config omitting the key ran the *unvalidated* mode
  (fixed 2026-07-24; see ``docs/configuration.md``).
* ``actor_distribution``, ``input_resolution``, ``enable_neck_flexion``,
  ``enable_lateral_bending`` — all four had drifted by 2026-07-31, found only by a
  hand audit.

An audit finds these once. This test finds them every run.

WHAT IT DOES NOT CHECK. Only keys in ``_MAPPING`` are compared. ``test_every_schema_default_is_mapped``
is what keeps that honest: add a defaulted key to the schema without mapping it here and
that test fails, so the mapping cannot silently fall behind the schema.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
from pathlib import Path

import pytest

from _repo_paths import nett_isaac_dir, read_source
from nett_skrl.brain.brain import Brain
from nett_skrl.brain.config import EncoderCfg, OnPolicyAlgorithmCfg, RewardCfg
from nett_skrl.brain.models.model_cfg import ModelCfg
from nett_skrl.environment import Environment
from nett_skrl.nett import NETT

SCHEMA = json.loads((Path(__file__).resolve().parent.parent / "nett_skrl" / "schema.json").read_text())

_MISSING = object()


def _signature_default(func, name):
    param = inspect.signature(func).parameters.get(name)
    if param is None or param.default is inspect.Parameter.empty:
        return _MISSING
    return param.default


def _dataclass_default(cls, name):
    for field in dataclasses.fields(cls):
        if field.name != name:
            continue
        if field.default is not dataclasses.MISSING:
            return field.default
        if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            return field.default_factory()  # type: ignore[misc]
    return _MISSING


def _cfg_default(cls, name):
    """`FlexibleCfg` subclasses carry their defaults on ``__init__``, not as fields."""
    return _signature_default(cls.__init__, name)


# schema dotted path -> callable returning the Python default that actually governs.
# Where a key is resolved by orchestration rather than by a constructor default, the
# entry says so and the resolved value is asserted instead.
_MAPPING = {
    # --- top level: NETT.single_run kwargs ---------------------------------
    "steps_per_episode": lambda: _signature_default(NETT.single_run, "steps_per_episode"),
    "num_brains": lambda: _signature_default(NETT.single_run, "num_brains"),
    "brain_id_offset": lambda: _signature_default(NETT.single_run, "brain_id_offset"),
    "eval_freq": lambda: _signature_default(NETT.single_run, "eval_freq"),
    "task_memory": lambda: _signature_default(NETT.single_run, "task_memory"),
    "max_parallel_envs": lambda: _signature_default(NETT.single_run, "max_parallel_envs"),
    # --- environment: Environment.__init__ --------------------------------
    "environment.headless": lambda: _signature_default(Environment.__init__, "headless"),
    "environment.input_resolution": lambda: _signature_default(Environment.__init__, "input_resolution"),
    "environment.camera_fov": lambda: _signature_default(Environment.__init__, "camera_fov"),
    "environment.record_mode": lambda: _signature_default(Environment.__init__, "record_mode"),
    "environment.random_first_frame": lambda: _signature_default(Environment.__init__, "random_first_frame"),
    "environment.decision_period": lambda: _signature_default(Environment.__init__, "decision_period"),
    "environment.enable_neck_flexion": lambda: _signature_default(Environment.__init__, "enable_neck_flexion"),
    "environment.enable_lateral_bending": lambda: _signature_default(Environment.__init__, "enable_lateral_bending"),
    "environment.locomotion": lambda: _signature_default(Environment.__init__, "locomotion"),
    "environment.render_mode": lambda: _signature_default(Environment.__init__, "render_mode"),
    "environment.tracemalloc_interval": lambda: _signature_default(Environment.__init__, "tracemalloc_interval"),
    "environment.train_step_logging": lambda: _signature_default(Environment.__init__, "train_step_logging"),
    "environment.conditions": lambda: _signature_default(Environment.__init__, "conditions"),
    # --- brain: Brain.__init__ --------------------------------------------
    "brain.encoder": lambda: _signature_default(Brain.__init__, "encoder"),
    "brain.algorithm": lambda: _signature_default(Brain.__init__, "algorithm"),
    "brain.reward": lambda: _signature_default(Brain.__init__, "reward"),
    "brain.evaluation_policy": lambda: _signature_default(Brain.__init__, "evaluation_policy"),
    "brain.checkpoint_freq": lambda: _signature_default(Brain.__init__, "checkpoint_freq"),
    # --- brain.model: ModelCfg dataclass ----------------------------------
    "brain.model.hidden_sizes": lambda: _dataclass_default(ModelCfg, "hidden_sizes"),
    "brain.model.activation": lambda: _dataclass_default(ModelCfg, "activation"),
    "brain.model.initial_log_std": lambda: _dataclass_default(ModelCfg, "initial_log_std"),
    "brain.model.max_log_std": lambda: _dataclass_default(ModelCfg, "max_log_std"),
    "brain.model.clip_actions": lambda: _dataclass_default(ModelCfg, "clip_actions"),
    "brain.model.value_bound": lambda: _dataclass_default(ModelCfg, "value_bound"),
    "brain.model.orthogonal_init": lambda: _dataclass_default(ModelCfg, "orthogonal_init"),
    "brain.model.hidden_gain": lambda: _dataclass_default(ModelCfg, "hidden_gain"),
    "brain.model.output_gain": lambda: _dataclass_default(ModelCfg, "output_gain"),
    "brain.model.shared_encoder": lambda: _dataclass_default(ModelCfg, "shared_encoder"),
    "brain.model.actor_distribution": lambda: _dataclass_default(ModelCfg, "actor_distribution"),
    # --- brain.*_cfg: FlexibleCfg subclasses ------------------------------
    "brain.encoder_cfg.features_dim": lambda: _cfg_default(EncoderCfg, "features_dim"),
    "brain.encoder_cfg.trainable": lambda: _cfg_default(EncoderCfg, "trainable"),
    "brain.algorithm_cfg.learning_rate": lambda: _cfg_default(OnPolicyAlgorithmCfg, "learning_rate"),
    "brain.algorithm_cfg.rollouts": lambda: _cfg_default(OnPolicyAlgorithmCfg, "rollouts"),
    "brain.algorithm_cfg.mini_batches": lambda: _cfg_default(OnPolicyAlgorithmCfg, "mini_batches"),
    "brain.algorithm_cfg.value_loss_scale": lambda: _cfg_default(OnPolicyAlgorithmCfg, "value_loss_scale"),
    "brain.algorithm_cfg.grad_norm_clip": lambda: _cfg_default(OnPolicyAlgorithmCfg, "grad_norm_clip"),
    "brain.reward_cfg.beta": lambda: _cfg_default(RewardCfg, "beta"),
    "brain.reward_cfg.kappa": lambda: _cfg_default(RewardCfg, "kappa"),
    "brain.reward_cfg.gamma": lambda: _cfg_default(RewardCfg, "gamma"),
    "brain.reward_cfg.weight": lambda: _cfg_default(RewardCfg, "weight"),
    "brain.reward_cfg.trainable": lambda: _cfg_default(RewardCfg, "trainable"),
}

# Keys whose schema default is deliberately NOT the constructor default, with the reason.
# Each entry must justify itself; `test_no_stale_exemptions` fails if a mapped-equivalent
# key is listed here needlessly.
_EXEMPT = {
    # `body.input_resolution` is an OVERRIDE: null means "inherit environment's".
    "body.input_resolution": "null means inherit from environment.input_resolution",
    # Composite object defaults ({} / nested dicts) declare shape, not a scalar value.
    "brain.model": "object-shaped default",
    "brain.encoder_cfg": "object-shaped default",
    "brain.algorithm_cfg": "object-shaped default",
    "brain.reward_cfg": "object-shaped default",
    "brain.wandb": "object-shaped default",
    "body": "object-shaped default",
    "environment.recording": "object-shaped default",
    "recordingSection": "object-shaped default",
    # Off-policy-only algorithm_cfg keys: no default on OnPolicyAlgorithmCfg.
    "brain.algorithm_cfg.batch_size": "off-policy only",
    "brain.algorithm_cfg.memory_size": "off-policy only",
    "brain.algorithm_cfg.gradient_steps": "off-policy only",
    "brain.algorithm_cfg.learning_starts": "off-policy only",
    # Resolved by orchestration, not by a constructor default.
    "environment.reward_types": "nett.single_run infers this from the brain's reward when absent",
    "episodes.train": "nett.single_run defaults the whole episodes dict, not per key",
    "episodes.test": "nett.single_run defaults the whole episodes dict, not per key",
    "episodes.record": "nett.single_run defaults the whole episodes dict, not per key",
    "environment.recording.fps": "recording sub-config, resolved in the recording layer",
    "body.wrappers": "resolved by _make_body",
    "brain.encoder_cfg.spatial_pool": "encoder-specific passthrough, no top-level default",
    "brain.wandb.project": "wandb sub-config, resolved in the wandb layer",
    "brain.wandb.entity": "wandb sub-config, resolved in the wandb layer",
    "brain.wandb.mode": "wandb sub-config, resolved in the wandb layer",
    "brain.wandb.tags": "wandb sub-config, resolved in the wandb layer",
    "brain.wandb.notes": "wandb sub-config, resolved in the wandb layer",
    "brain.wandb.kwargs": "wandb sub-config, resolved in the wandb layer",
}


def _declared_defaults() -> dict[str, object]:
    """Every ``default:`` in the schema, keyed by dotted path."""
    found: dict[str, object] = {}

    def walk(node, path):
        if not isinstance(node, dict):
            return
        if "default" in node:
            found[path.lstrip(".")] = node["default"]
        for container in ("properties", "$defs", "definitions"):
            for name, sub in (node.get(container) or {}).items():
                walk(sub, f"{path}.{name}")
        if isinstance(node.get("items"), dict):
            walk(node["items"], path + "[]")

    walk(SCHEMA, "")
    return found


DECLARED = _declared_defaults()


# Keys whose schema default is knowingly ahead of the code, with the task that closes the
# gap. `strict=True` is the point: when the code catches up, the xfail turns into an
# unexpected PASS and this suite goes red until the entry is deleted. A pending exception
# that outlives its bug is how the original drift survived.
_PENDING_FIX = {
    "environment.camera_fov": (
        "schema declares 150.0 (repo A's ObservationCfg.fov, and what the fisheye reward "
        "geometry assumes); repo B's Environment still defaults 120.0. Closed by "
        "maintenance PLAN.md task B1, which moves the CODE to 150.0 — a deliberate "
        "behavior change, hence its own commit rather than being folded in here."
    ),
}


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            key,
            marks=(
                [pytest.mark.xfail(reason=_PENDING_FIX[key], strict=True)]
                if key in _PENDING_FIX
                else []
            ),
        )
        for key in sorted(k for k in _MAPPING if k in DECLARED)
    ],
)
def test_schema_default_matches_python_default(path):
    declared = DECLARED[path]
    actual = _MAPPING[path]()
    assert actual is not _MISSING, f"{path}: mapping points at a parameter that no longer exists"
    # Schema JSON cannot express a tuple; compare sequences by content.
    if isinstance(actual, tuple):
        actual = list(actual)
    assert declared == actual, (
        f"schema.json declares {path} = {declared!r} but the Python default that actually "
        f"governs is {actual!r}. jsonschema does not apply defaults, so the Python value "
        f"wins silently and the schema misleads every reader. Fix whichever is wrong — and "
        f"if you are changing the EFFECTIVE default, that is a behavior change: say so."
    )


def test_every_schema_default_is_mapped():
    """A new defaulted schema key must be mapped or explicitly exempted.

    Without this, the mapping quietly falls behind the schema and the check above
    degrades into testing a shrinking subset — which is how the original drift
    survived in the first place.
    """
    unaccounted = sorted(set(DECLARED) - set(_MAPPING) - set(_EXEMPT))
    assert not unaccounted, (
        "schema.json declares defaults with no Python counterpart in _MAPPING and no "
        f"entry in _EXEMPT: {unaccounted}. Add the mapping, or exempt it with a reason."
    )


def test_no_stale_exemptions():
    """An exemption that is also mapped is dead weight and hides a real comparison."""
    both = sorted(set(_EXEMPT) & set(_MAPPING))
    assert not both, f"keys are both mapped and exempted; drop the exemption: {both}"


@pytest.mark.xfail(reason=_PENDING_FIX["environment.camera_fov"], strict=True)
def test_repo_b_environment_fov_matches_repo_a():
    """repo B's ``camera_fov`` default must equal repo A's ``ObservationCfg.fov``.

    The two are independent declarations of one physical fact — the chick's fisheye
    aperture — and ``_ENV_CFG_FIELDS`` copies repo B's value into the env cfg
    *unconditionally*, so when they disagree repo B wins silently. They did disagree
    until 2026-07-31 (repo B 120.0 vs repo A 150.0): every config omitting the key ran an
    optic no experiment used, and the closeness projection is calibrated for 150°.

    ⚠ Read repo A's value from its SOURCE TEXT, not by importing it. ``nett_env_cfg``
    pulls in ``isaaclab.actuators``, which only exists once Kit has booted — an
    ``importorskip`` here would skip in the plain unit suite and this guard would never
    actually run. The source-text convention (``_repo_paths``) is what the rest of the
    cross-repo assertions in this suite use, and it needs no Kit.
    """
    source = read_source(nett_isaac_dir() / "nett_env_cfg.py")
    match = re.search(r"^\s*fov\s*:\s*float\s*=\s*([0-9.]+)", source, re.MULTILINE)
    assert match, (
        "could not find `fov: float = <value>` in repo A's nett_env_cfg.py — the field was "
        "renamed or restructured, so this cross-repo guard is stale and needs updating"
    )
    repo_a_fov = float(match.group(1))
    repo_b_fov = _signature_default(Environment.__init__, "camera_fov")
    assert repo_b_fov == repo_a_fov, (
        f"repo B Environment(camera_fov={repo_b_fov!r}) != repo A ObservationCfg.fov="
        f"{repo_a_fov!r}. repo B's value is copied into the env cfg unconditionally, so "
        f"it wins — any config omitting camera_fov silently runs repo B's number."
    )
