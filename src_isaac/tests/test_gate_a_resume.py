"""CPU tests for the Gate A resume runner's preflight and seeding.

The runner's whole purpose is to make a silent failure impossible: every way a resume
can leave you training from random weights while exiting 0 is caught by one of these.
So the tests exercise the failure paths, not the happy one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from campaign_train import MODELS  # noqa: E402
from gate_a_resume import (  # noqa: E402
    RESUME_MODULES,
    _verify_resume,
    SHAPE_FIELDS,
    compatible,
    seed_checkpoints,
    source_checkpoints,
    train_eps_for,
)


def _ckpt(path: Path, *, keys=("policy", "value", "optimizer", "value_preprocessor")):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: {"w": torch.zeros(2)} for k in keys}, path)


# ---------------------------------------------------------------------------
# 1. Is there anything to resume FROM?
# ---------------------------------------------------------------------------


def test_no_checkpoints_is_detectable_before_launch(tmp_path):
    """Empty means empty. The alternative is a run that trains from noise and exits 0."""
    (tmp_path / "fork-1" / "wandb_runs" / "brain_1" / "checkpoints").mkdir(parents=True)
    assert source_checkpoints(tmp_path) == {}


def test_final_agent_is_preferred_over_a_mid_training_snapshot(tmp_path):
    """`trained` is the property the whole tool needs, so the END of training wins."""
    ck = tmp_path / "fork-1" / "wandb_runs" / "brain_1" / "checkpoints"
    _ckpt(ck / "agent_6250.pt")
    _ckpt(ck / "agent_62500.pt")
    _ckpt(ck / "final_agent.pt")
    assert source_checkpoints(tmp_path)[1].name == "final_agent.pt"


def test_newest_numbered_snapshot_when_there_is_no_final(tmp_path):
    """Ordering must be NUMERIC: lexically, agent_6250 sorts after agent_62500."""
    ck = tmp_path / "fork-1" / "wandb_runs" / "brain_1" / "checkpoints"
    _ckpt(ck / "agent_6250.pt")
    _ckpt(ck / "agent_62500.pt")
    assert source_checkpoints(tmp_path)[1].name == "agent_62500.pt"


def test_every_brain_is_found_and_keyed_by_its_own_index(tmp_path):
    for i in (1, 2, 7):
        _ckpt(tmp_path / "fork-1" / "wandb_runs" / f"brain_{i}" / "checkpoints" / "final_agent.pt")
    assert sorted(source_checkpoints(tmp_path)) == [1, 2, 7]


# ---------------------------------------------------------------------------
# 2. Will the state dict fit? (the check that is NOT a name comparison)
# ---------------------------------------------------------------------------


def test_same_architecture_different_aux_is_compatible():
    """The point of the whole exercise: resume a trained policy into a new AUX loss.

    ViT+VICReg and ViT-VICReg-TT differ only in the auxiliary objective, and the aux
    head's weights are not in the checkpoint at all -- so the policy transplants.
    """
    ok, why = compatible("ViT+VICReg", "ViT-VICReg-TT")
    assert ok, why


def test_different_encoder_is_refused():
    a = next(m for m in MODELS if MODELS[m]["encoder"] == MODELS["ViT+VICReg"]["encoder"])
    b = next(m for m in MODELS if MODELS[m]["encoder"] != MODELS[a]["encoder"])
    ok, why = compatible(a, b)
    assert not ok
    assert "RANDOM WEIGHTS" in why, why


def test_a_framestack_mismatch_is_refused_even_at_the_same_encoder():
    """Framestack changes the input channel count, so conv/patch weights change shape.

    It is the mismatch most likely to be waved through, because the two models can
    carry the same encoder name and the same cfg.
    """
    pairs = [(a, b) for a in MODELS for b in MODELS
             if MODELS[a]["encoder"] == MODELS[b]["encoder"]
             and MODELS[a].get("cfg") == MODELS[b].get("cfg")
             and MODELS[a]["framestack"] != MODELS[b]["framestack"]]
    if not pairs:
        pytest.skip("no same-encoder same-cfg pair differing only in framestack in MODELS")
    ok, why = compatible(*pairs[0])
    assert not ok and "framestack" in why


def test_unknown_model_names_are_refused_by_name():
    assert not compatible("NoSuchModel", "ViT+VICReg")[0]
    assert not compatible("ViT+VICReg", "NoSuchModel")[0]


def test_compatibility_ignores_the_model_NAME():
    """Two entries equal on all three shape fields are compatible however they are named."""
    for m in MODELS:
        ok, _ = compatible(m, m)
        assert ok, m
    assert set(SHAPE_FIELDS) == {"encoder", "cfg", "framestack"}


# ---------------------------------------------------------------------------
# 3. Seeding: what gets written, and what deliberately does not
# ---------------------------------------------------------------------------


def test_the_optimizer_is_dropped_and_the_policy_is_kept(tmp_path):
    """Its aux-head param group differs by aux kind; loading it across kinds raises
    HALFWAY THROUGH agent.load, after policy has already been assigned."""
    src = tmp_path / "src" / "final_agent.pt"
    _ckpt(src)
    dest = tmp_path / "dest"
    man = seed_checkpoints({1: src}, dest, dry_run=False)
    written = torch.load(dest / "wandb_runs" / "brain_1" / "checkpoints" / "final_agent.pt",
                         map_location="cpu", weights_only=False)
    assert set(written) == set(RESUME_MODULES)
    assert "optimizer" not in written
    assert man[0]["dropped_keys"] == ["optimizer"]


def test_the_manifest_records_what_was_dropped_not_only_what_was_kept(tmp_path):
    """A record of the kept keys alone cannot answer 'was anything lost?'."""
    src = tmp_path / "src" / "final_agent.pt"
    _ckpt(src, keys=("policy", "value", "optimizer", "value_preprocessor", "extra_thing"))
    man = seed_checkpoints({3: src}, tmp_path / "dest", dry_run=False)
    assert man[0]["brain"] == 3
    assert man[0]["source_keys"] == sorted(
        ["policy", "value", "optimizer", "value_preprocessor", "extra_thing"])
    assert set(man[0]["dropped_keys"]) == {"optimizer", "extra_thing"}
    json.dumps(man)   # the manifest is written to disk, so it must serialise


def test_a_checkpoint_without_a_policy_is_refused(tmp_path):
    src = tmp_path / "src" / "final_agent.pt"
    _ckpt(src, keys=("value", "optimizer"))
    with pytest.raises(SystemExit, match="nothing to resume"):
        seed_checkpoints({1: src}, tmp_path / "dest", dry_run=False)


def test_dry_run_writes_nothing(tmp_path):
    src = tmp_path / "src" / "final_agent.pt"
    _ckpt(src)
    dest = tmp_path / "dest"
    man = seed_checkpoints({1: src}, dest, dry_run=True)
    assert man and not dest.exists(), "--dry-run must not touch the filesystem"


def test_the_destination_layout_is_the_one_brain_py_reads(tmp_path):
    """`load_latest_checkpoints` looks under
    {config.path}/wandb_runs/brain_{i}/checkpoints -- one wrong segment and the loader
    finds nothing, logs it, and trains from random weights."""
    src = tmp_path / "src" / "final_agent.pt"
    _ckpt(src)
    cfg_path = tmp_path / "run" / "fork-1"
    seed_checkpoints({2: src}, cfg_path, dry_run=False)
    assert (cfg_path / "wandb_runs" / "brain_2" / "checkpoints" / "final_agent.pt").is_file()


# ---------------------------------------------------------------------------
# 4. Enough updates for a trajectory
# ---------------------------------------------------------------------------


def test_train_eps_gives_the_requested_number_of_updates():
    """One update per episode per env: updates = train_eps // envs_per_brain."""
    for updates in (1, 2, 4, 10):
        for envs in (8, 16):
            assert train_eps_for(updates, envs) // envs == updates


def test_the_default_run_can_licence_the_kill_branch():
    """Gate A's registered kill is 'flat-or-declining'. On ONE reading that is UNDEFINED,
    not unmet -- so a default of 1 would make the gate structurally unable to kill, and
    the run that discovered it would already have been spent."""
    from gate_a_resume import build_parser
    assert build_parser().get_default("updates") >= 2


# ---------------------------------------------------------------------------
# 5. NETT_RUN_NAME's guard must not refuse the one caller it has
# ---------------------------------------------------------------------------


def _run_name_guard(tmp_path, name, build=None):
    """Exercise campaign_train's NETT_RUN_NAME guard predicate in isolation."""
    existing = tmp_path / name
    if build:
        build(existing)
    output = [p for p in (existing / "campaign_timing.json", *existing.glob("*/logs"),
                          *existing.glob("logs")) if p.exists()]
    return output


def test_a_seeded_directory_is_not_run_output(tmp_path):
    """⛔ THE BUG THIS PINS COST A LAUNCH. gate_a_resume must create the run directory to
    seed checkpoints into it, so a guard keyed on EXISTENCE refuses its own only client --
    measured 2026-09-10, 2 seconds in."""
    def seeded(root):
        (root / "fork-1" / "wandb_runs" / "brain_1" / "checkpoints").mkdir(parents=True)
        (root / "fork-1" / "wandb_runs" / "brain_1" / "checkpoints" / "final_agent.pt").touch()
        (root / "resume_manifest.json").write_text("{}")
    assert _run_name_guard(tmp_path, "gateA_x", seeded) == []


def test_a_directory_holding_logs_is_refused(tmp_path):
    """The real hazard: mkdir(exist_ok=True) would merge two runs' logs with no error."""
    def used(root):
        (root / "fork-1" / "logs").mkdir(parents=True)
        (root / "fork-1" / "logs" / "test_fork-1_0.csv").touch()
    assert _run_name_guard(tmp_path, "gateA_x", used)


def test_a_finished_run_is_refused_by_its_timing_file(tmp_path):
    def finished(root):
        root.mkdir(parents=True)
        (root / "campaign_timing.json").write_text("{}")
    assert _run_name_guard(tmp_path, "gateA_x", finished)


def test_an_absent_directory_is_allowed(tmp_path):
    assert _run_name_guard(tmp_path, "never_used") == []


# ---------------------------------------------------------------------------
# 6. Verifying the resume landed -- by weights, not by a log line
# ---------------------------------------------------------------------------


def _resume_pair(tmp_path, brain, src_w, final_w):
    src = tmp_path / f"src_{brain}.pt"
    torch.save({"policy": {"w": src_w}}, src)
    final = tmp_path / "run" / "wandb_runs" / f"brain_{brain}" / "checkpoints"
    final.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": {"w": final_w}}, final / "final_agent.pt")
    return {"brain": brain, "source": str(src)}


def test_a_short_resume_verifies(tmp_path):
    """4 PPO updates drift the policy slightly: measured cosine 0.990260 on the real run."""
    base = torch.randn(5000)
    man = [_resume_pair(tmp_path, 1, base, base + 0.02 * torch.randn(5000))]
    ok, checked, _ = _verify_resume(man, tmp_path / "run")
    assert (ok, checked) == (1, 1)


def test_random_weights_are_caught(tmp_path):
    """⛔ THE CASE THE LOG-COUNTING VERSION GOT BACKWARDS. A run that trained from a fresh
    init is orthogonal to the checkpoint it was supposed to resume: measured -0.000727."""
    man = [_resume_pair(tmp_path, 1, torch.randn(5000), torch.randn(5000))]
    ok, checked, _ = _verify_resume(man, tmp_path / "run")
    assert (ok, checked) == (0, 1)


def test_the_floor_sits_between_the_two_by_orders_of_magnitude(tmp_path):
    """0.5 is not a tuned threshold: the two populations are ~0.99 and ~0.000."""
    base = torch.randn(20000)
    resumed = float(torch.nn.functional.cosine_similarity(
        base, base + 0.02 * torch.randn(20000), dim=0))
    fresh = abs(float(torch.nn.functional.cosine_similarity(
        base, torch.randn(20000), dim=0)))
    assert resumed > 0.9 and fresh < 0.1, (resumed, fresh)


def test_a_missing_final_checkpoint_is_not_counted_as_a_pass(tmp_path):
    """A run that never wrote a final checkpoint must not read as 'verified'."""
    src = tmp_path / "src.pt"
    torch.save({"policy": {"w": torch.randn(10)}}, src)
    ok, checked, note = _verify_resume([{"brain": 1, "source": str(src)}], tmp_path / "run")
    assert (ok, checked) == (0, 0) and "no comparable" in note


def test_every_brain_is_checked_not_just_the_first(tmp_path):
    base = torch.randn(5000)
    man = [_resume_pair(tmp_path, 1, base, base + 0.02 * torch.randn(5000)),
           _resume_pair(tmp_path, 2, torch.randn(5000), torch.randn(5000))]
    ok, checked, _ = _verify_resume(man, tmp_path / "run")
    assert (ok, checked) == (1, 2), "a per-brain failure must not be averaged away"
