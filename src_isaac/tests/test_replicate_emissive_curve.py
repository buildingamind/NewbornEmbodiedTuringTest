"""The emissive-curve replication recipe must stay identical to the published one.

The published curve (e300 2/8, e1000 7/8, e2000 6/8) came from
``train_binding_8brain.py`` with three hand edits and ``NETT_CHAMBER_VARIANT``
exported. ``examples/replicate_emissive_curve.py`` encodes that recipe so it can
be re-run; these tests pin the parts of it that would silently invalidate a
replication if they drifted.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

# The examples run as scripts from their own directory (`from _paths import ...`),
# so reproduce that import context rather than treating them as a package.
_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import replicate_emissive_curve as rec  # noqa: E402
import train_binding_8brain as canonical  # noqa: E402


class TestRecipeDeltas:
    """Exactly four deltas from the canonical config — no more, no fewer."""

    @pytest.fixture
    def cfg(self) -> dict:
        return rec.build_config(1000, 8, rec.CURVE_EPISODES_TRAIN)

    def test_wandb_is_disabled(self, cfg):
        assert cfg["brain"]["wandb"]["mode"] == "disabled"

    def test_locomotion_is_pinned_not_inherited(self, cfg):
        """The canonical config omits `locomotion`, so it follows the RUNTIME
        DEFAULT — and that default flipped kinematic -> wheeled hours after the
        curve was measured. A replication that inherits it changes the physics
        as well as the seed and can attribute a difference to neither."""
        assert "locomotion" not in canonical.CONFIG["environment"]
        assert cfg["environment"]["locomotion"] == rec.PUBLISHED_LOCOMOTION
        assert rec.PUBLISHED_LOCOMOTION == "kinematic"

    def test_locomotion_is_overridable(self):
        cfg = rec.build_config(1000, 8, 1000, locomotion="wheeled")
        assert cfg["environment"]["locomotion"] == "wheeled"

    def test_training_budget_is_the_curve_budget(self, cfg):
        # Binding does not bind below ~1000 episodes; a shorter budget yields
        # chance/side-lock at EVERY brightness and says nothing about the curve.
        assert cfg["episodes"]["train"] == 1000
        assert rec.CURVE_EPISODES_TRAIN == 1000

    def test_brain_id_offset_is_carried(self, cfg):
        assert cfg["brain_id_offset"] == 8

    def test_nothing_else_changed(self, cfg):
        """Any other difference makes the replication a different experiment."""
        expected = dict(canonical.CONFIG)
        differing = {
            k for k in set(cfg) | set(expected)
            if cfg.get(k) != expected.get(k)
        }
        # `name` carries a timestamp, so it always differs.
        assert differing == {"name", "episodes", "brain", "brain_id_offset",
                             "environment"}
        # ...and within `environment`, only the pinned locomotion.
        assert {
            k for k in set(cfg["environment"]) | set(expected["environment"])
            if cfg["environment"].get(k) != expected["environment"].get(k)
        } == {"locomotion"}
        # ...and within `brain`, only wandb.
        assert {
            k for k in set(cfg["brain"]) | set(expected["brain"])
            if cfg["brain"].get(k) != expected["brain"].get(k)
        } == {"wandb"}
        assert {
            k for k in cfg["brain"]["wandb"]
            if cfg["brain"]["wandb"][k] != expected["brain"]["wandb"][k]
        } == {"mode"}

    def test_canonical_config_is_not_mutated(self):
        rec.build_config(300, 16, 1000)
        assert canonical.CONFIG["episodes"]["train"] == 2000
        assert canonical.CONFIG["brain"]["wandb"]["mode"] == "online"
        assert "brain_id_offset" not in canonical.CONFIG


class TestExperimentInvariants:
    def test_brain_id_offset_is_a_real_run_parameter(self):
        """A typo'd key would be swallowed by **kwargs and silently do nothing —
        every replication would then be a determinism replay of offset 0."""
        from nett_skrl import NETT

        assert "brain_id_offset" in inspect.signature(NETT.single_run).parameters

    def test_env_count_is_a_valid_square_grid(self):
        """num_envs must land in [k^2-k+1, k^2] or the fisheye is distorted
        (Isaac #488). res256 pins this at 16 = 4^2; changing brains or envs
        without rechecking the product silently corrupts the observations."""
        n = rec.build_config(1000, 8, 1000)["max_parallel_envs"]
        k = int(n ** 0.5) + (0 if int(n ** 0.5) ** 2 == n else 1)
        assert k * k - k + 1 <= n <= k * k, f"{n} envs is not a valid square grid"

    def test_variant_name_matches_the_baked_assets(self):
        assert rec.variant_name(1000) == "chamber_e1000.usdc"
        assert rec.variant_name(1000, "realistic") == "chamber_re1000.usdc"
        from _repo_paths import repo_a_root

        assets = repo_a_root() / "isaac_lab" / "assets" / "chamber"
        if not assets.is_dir():
            pytest.skip(f"repoA chamber assets not found at {assets}")
        for style in ("flat", "realistic"):
            for point in rec.MEASURED_POINTS:
                assert (assets / rec.variant_name(point, style)).exists(), (
                    f"no baked {style} chamber for measured point e{point}"
                )

    def test_chamber_style_defaults_to_the_published_one(self):
        """The style is pinned for the same reason locomotion is: the published
        curve ran in the flat chamber, and a run that silently picked up the
        restyle would be comparing different stimuli, not different seeds."""
        assert rec.PUBLISHED_STYLE == "flat"
        assert rec.variant_name(300) == rec.variant_name(300, rec.PUBLISHED_STYLE)

    def test_restyled_runs_record_their_style_in_the_run_name(self):
        """The 2026-07-24 curve runs recorded their chamber variant NOWHERE in
        their artifacts — the directory name was the only provenance. A restyled
        run must therefore not share a name with a flat run at the same point."""
        flat = rec.build_config(1000, 8, 1000, style="flat")["name"]
        real = rec.build_config(1000, 8, 1000, style="realistic")["name"]
        assert flat.startswith("binding_e1000_off8")
        assert real.startswith("binding_re1000_off8")


class TestScoring:
    def test_primary_endpoint_is_the_learn_fraction(self, tmp_path):
        """Scoring must read the verdict counts, not correct_pct_mean — the
        scalar cannot separate a side-locked brain from a wandering one."""
        analysis = tmp_path / "analysis"
        analysis.mkdir()
        (analysis / "summary.json").write_text(json.dumps({
            "test": {"Object1": {"rest": {
                "correct_pct_mean": 0.5,
                "learn_fraction": 0.25,
                "verdict_counts": {"LEARN": 2, "SIDE-LOCK": 6, "chance": 0, "n/a": 0},
                "n_brains": 8,
                "n_brains_immobile": 0,
                "side_preference_abs_mean": 0.9,
            }}}
        }))
        scores = rec.score(analysis)["Object1"]
        assert scores["learn_fraction"] == 0.25
        assert scores["verdict_counts"]["SIDE-LOCK"] == 6
        # correct_pct_mean is reported alongside, never as the endpoint.
        assert scores["correct_pct_mean"] == 0.5

    def test_missing_rest_condition_yields_no_score(self, tmp_path):
        analysis = tmp_path / "analysis"
        analysis.mkdir()
        (analysis / "summary.json").write_text(
            json.dumps({"test": {"Object1": {"1color": {"learn_fraction": 1.0}}}})
        )
        assert rec.score(analysis) == {}
