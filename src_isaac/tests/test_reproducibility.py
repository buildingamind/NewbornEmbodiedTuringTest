"""Reproducibility / same-seed determinism tests.

Maps to the acceptance criteria (D) in workspace/notes/09_reproducibility.md.

Two flavors, per the env constraints (Isaac Sim is not importable here):

* **Behavior tests** import ``set_seeds`` / ``TaskConfig`` from
  ``nett_skrl.runtime.task`` (package root ``src_isaac``) and run for real.
* **Source-inspection tests** read the in-Isaac wiring files as text and assert
  substrings, following the repo's established pattern
  (``isaac_lab/tests/test_camera_mount.py``).

repoB paths are computed from this file's location so the suite is relocatable;
repoA is resolved by ``_repo_paths`` (``$NETT_REPO_A`` or the sibling default).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import pytest

from nett_skrl.runtime.task import set_seeds, TaskConfig


# --- Repo-root resolution (robust to absolute checkout location) -------------
# repoA is resolved via $NETT_REPO_A or the sibling default; see _repo_paths for
# what that does and does NOT guarantee about the package on PYTHONPATH.
from _repo_paths import SRC_ISAAC as _SRC_ISAAC, nett_isaac_dir, read_source as _read

_TASK_PY = _SRC_ISAAC / "nett_skrl" / "runtime" / "task.py"
_NETT_ENV_PY = nett_isaac_dir() / "nett_env.py"


# === Criterion 1: set_seeds configures full determinism =====================

def test_set_seeds_configures_determinism_flags():
    set_seeds(1234)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["PYTHONHASHSEED"] == str(1234)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True


# === Criterion 2: same seed reproduces, different seed diverges =============

def _draw_sequences():
    t = torch.rand(5)
    n = np.random.rand(5)
    r = [random.random() for _ in range(5)]
    return t, n, r


def test_same_seed_reproduces_all_rng_streams():
    set_seeds(7)
    t1, n1, r1 = _draw_sequences()
    set_seeds(7)
    t2, n2, r2 = _draw_sequences()
    assert torch.equal(t1, t2)
    assert np.array_equal(n1, n2)
    assert r1 == r2


def test_different_seed_diverges():
    set_seeds(7)
    t1, n1, r1 = _draw_sequences()
    set_seeds(99)
    t2, n2, r2 = _draw_sequences()
    assert not torch.equal(t1, t2)
    assert not np.array_equal(n1, n2)
    assert r1 != r2


# === Criterion 3: skrl call guarded; set_seeds runnable without skrl ========

def test_set_seeds_skrl_call_is_guarded_in_source():
    src = _read(_TASK_PY)
    # The skrl import + set_seed must live inside a try/except so set_seeds
    # stays importable/usable when skrl is absent.
    assert "try:" in src
    assert "from skrl.utils import set_seed" in src
    assert "except Exception" in src
    # try/except wraps the skrl call.
    try_idx = src.index("from skrl.utils import set_seed")
    except_idx = src.index("except Exception", try_idx)
    assert src.rindex("try:", 0, try_idx) < try_idx < except_idx


def test_set_seeds_runs_without_raising(monkeypatch):
    # Simulate skrl not being installed: force the guarded import to fail.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "skrl.utils" or name.startswith("skrl"):
            raise ImportError("skrl not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Must complete without raising even though skrl import fails.
    set_seeds(42)


# === Criterion 4: TaskConfig seed derivation ================================

def _make_cfg(condition, tmp_path, **kw):
    return TaskConfig(
        condition=condition,
        output_dir=tmp_path,
        modes=["train"],
        **kw,
    )


def test_taskconfig_same_condition_same_seed(tmp_path):
    a = _make_cfg("conditionA", tmp_path)
    b = _make_cfg("conditionA", tmp_path)
    assert a.seed == b.seed


def test_taskconfig_different_condition_different_seed(tmp_path):
    a = _make_cfg("conditionA", tmp_path)
    c = _make_cfg("conditionB", tmp_path)
    assert a.seed != c.seed


def test_taskconfig_seed_is_condition_only_not_perturbed_by_offset(tmp_path):
    # The task seed is now the CONDITION base seed only; brain_id_offset no longer
    # perturbs it. The offset differentiates runs downstream via the GLOBAL brain id
    # (brain_id_offset + local id) applied to BOTH weight init (agent_factory) and the
    # per-episode env draws (nett_isaac.episode_seed) -- which is what makes single-brain
    # offset-b bit-identical to brain b of a multi-brain run. See runtime/task.py.
    base = _make_cfg("conditionA", tmp_path, brain_id_offset=0)
    shifted = _make_cfg("conditionA", tmp_path, brain_id_offset=1)
    assert base.seed == shifted.seed


# === Criterion 5: generator-isolation baseline (heart of G2/S3) =============

def test_dedicated_generator_is_invariant_to_intervening_global_draws():
    """A manually-seeded torch.Generator yields an identical sequence even when
    GLOBAL torch.rand calls happen between draws; the global RNG does not.
    """
    device = "cpu"

    # --- Dedicated generator: reproducible despite global interference ---
    g = torch.Generator(device=device)
    g.manual_seed(2024)
    draw_a = torch.rand(3, generator=g)

    g = torch.Generator(device=device)
    g.manual_seed(2024)
    # Intervening global-RNG churn between reseed and the draw.
    _ = torch.rand(17)
    _ = torch.rand(5)
    draw_b = torch.rand(3, generator=g)

    assert torch.equal(draw_a, draw_b), "dedicated generator must isolate the stream"

    # --- Global RNG: diverges under the same interference ---
    torch.manual_seed(2024)
    global_a = torch.rand(3)

    torch.manual_seed(2024)
    _ = torch.rand(17)  # an agent-side draw shifts the global stream
    global_b = torch.rand(3)

    assert not torch.equal(global_a, global_b), (
        "global RNG should diverge when intervening draws occur (proves isolation matters)"
    )


# === Criterion 6: source wiring (nett_env.py + task.py) =====================

def test_nett_env_seed_override_calls_configure_seed_deterministic():
    src = _read(_NETT_ENV_PY)
    assert "def seed(" in src
    assert "configure_seed(" in src
    assert "torch_deterministic=True" in src
    # configure_seed is called with the deterministic flag.
    seed_def_idx = src.index("def seed(")
    cfg_call_idx = src.index("configure_seed(seed, torch_deterministic=True)")
    assert cfg_call_idx > seed_def_idx


def test_nett_env_draws_pose_from_per_episode_substream():
    # Determinism upgrade: the start pose is no longer drawn from the shared
    # _reset_generator. Each env's pose comes from an INDEPENDENT per-episode substream
    # (nett_isaac.episode_seed), keyed by (global brain id, brain-local episode index),
    # and passed to reset_pose as precomputed values `u=...`. This is strictly more
    # isolated than the old dedicated generator: invariant to draw order AND to num_envs
    # (parallel==sequential). See _reset_idx.
    src = _read(_NETT_ENV_PY)
    assert "self.phase.reset_pose(" in src
    rp_idx = src.index("self.phase.reset_pose(")
    call_segment = src[rp_idx:rp_idx + 200]
    assert "u=" in call_segment                                # per-env precomputed pose
    assert "generator=self._reset_generator" not in call_segment
    # The pose values come from the episode-keyed substream on the POSE channel.
    assert "episode_rng(" in src and "CH_POSE" in src and "brain_episode_key(" in src


def test_task_set_seeds_source_references_determinism_knobs():
    src = _read(_TASK_PY)
    assert "CUBLAS_WORKSPACE_CONFIG" in src
    assert "use_deterministic_algorithms" in src
    # Guarded skrl set_seed.
    assert "set_seed" in src
    assert "from skrl.utils import set_seed" in src


# === Criterion 8 (Critic W1): warn_only is the FINAL determinism mode ========

def test_set_seeds_leaves_deterministic_in_warn_only_mode():
    """skrl.set_seed(deterministic=True) flips strict mode on; set_seeds must
    re-assert warn_only LAST so a missing deterministic kernel WARNS instead of
    crashing a long run. Regression guard for Critic Warning 1.
    """
    set_seeds(321)
    assert torch.are_deterministic_algorithms_enabled() is True
    # The decisive check: deterministic mode is enabled but in warn_only flavor.
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True


def test_set_seeds_source_reasserts_warn_only_after_skrl():
    """The ambient policy must be re-asserted AFTER the skrl set_seed call.

    ``skrl.set_seed(deterministic=True)`` flips STRICT mode on; set_seeds has to
    put the policy back afterwards. warn_only is the default; the only way to get
    strict is the NETT_STRICT_DETERMINISM diagnostic opt-in.
    """
    src = _read(_TASK_PY)
    skrl_idx = src.index("from skrl.utils import set_seed")
    reassert_idx = src.index("torch.use_deterministic_algorithms(", skrl_idx)
    assert reassert_idx > skrl_idx, "warn_only must be re-asserted after skrl set_seed"
    assert "warn_only=not strict" in src[reassert_idx:reassert_idx + 120]


# === NETT_STRICT_DETERMINISM: opt-in strict AMBIENT policy (diagnostic) ======
# The PPO update already runs strict via @strict_update, but rollout COLLECTION
# runs under the warn_only ambient policy, so a nondeterministic op there is
# silently tolerated. This knob makes it raise, so run-to-run weight divergence
# can be ATTRIBUTED to a named op instead of merely observed.


def test_strict_determinism_env_var_makes_ambient_policy_raise(monkeypatch):
    monkeypatch.setenv("NETT_STRICT_DETERMINISM", "1")
    try:
        set_seeds(321)
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is False
    finally:
        # set_seeds mutates PROCESS-GLOBAL torch state; put the default back so
        # this diagnostic cannot leak into later tests.
        monkeypatch.delenv("NETT_STRICT_DETERMINISM", raising=False)
        set_seeds(321)
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True


def test_strict_determinism_is_off_by_default(monkeypatch):
    """A long production run must not abort on an op with no deterministic kernel."""
    monkeypatch.delenv("NETT_STRICT_DETERMINISM", raising=False)
    set_seeds(321)
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True


def test_strict_determinism_context_restores_the_ambient_policy():
    """The PPO-update wrapper must RESTORE the ambient policy, not force warn_only.

    It used to hardcode ``warn_only=True`` on exit, which disarmed
    NETT_STRICT_DETERMINISM after the very first PPO update — leaving the rest of
    the run, where the divergence accumulates, silently tolerant again.
    """
    from nett_skrl.brain.skrl_patches import strict_determinism

    try:
        for ambient_warn_only in (True, False):
            torch.use_deterministic_algorithms(True, warn_only=ambient_warn_only)
            with strict_determinism():
                # Inside, the learning path is always strict.
                assert torch.is_deterministic_algorithms_warn_only_enabled() is False
            assert (
                torch.is_deterministic_algorithms_warn_only_enabled()
                is ambient_warn_only
            )
    finally:
        torch.use_deterministic_algorithms(True, warn_only=True)


def test_nett_env_seed_override_reasserts_warn_only():
    """The env's seed() override calls configure_seed (strict) then relaxes to
    warn_only, mirroring set_seeds. Regression guard for Critic Warning 1.
    """
    src = _read(_NETT_ENV_PY)
    cfg_idx = src.index("configure_seed(seed, torch_deterministic=True)")
    warn_idx = src.index("use_deterministic_algorithms(True, warn_only=True)")
    assert warn_idx > cfg_idx, "warn_only must be re-asserted after configure_seed"


# === Criterion 9 (Critic W3): skrl PRNG key seeded from the condition seed ====

def test_set_seeds_seeds_skrl_torch_key():
    """skrl_adapter reads config.torch.key for the first env reset; set_seeds must
    drive it from the condition seed so the reset is reproducible (G3).
    """
    skrl_config = pytest.importorskip("skrl").config
    set_seeds(555)
    key = skrl_config.torch.key
    key_val = int(key) if not hasattr(key, "__len__") else int(np.asarray(key).ravel()[0])
    assert key_val == 555


def test_validate_tasklist_seeds_before_load():
    """The smoke-validation path must seed before building the env (Critic W2)."""
    tasklist_py = _SRC_ISAAC / "nett_skrl" / "runtime" / "tasklist.py"
    src = _read(tasklist_py)
    seed_idx = src.index("set_seeds(config.seed)")
    load_idx = src.index("task.agent.env.load(")
    assert seed_idx < load_idx, "set_seeds must precede env.load in validate_tasklist"
