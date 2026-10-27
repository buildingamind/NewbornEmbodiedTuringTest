"""The two protocol DEFAULTS changed on 2026-07-30, and defaults are science here.

A default that silently reverts costs a re-baseline, because both of these change what a
number MEANS rather than how noisy it is:

* ``NETT_EVAL_STOCHASTIC`` -> sampled vs mean action at test. With the fixed test start
  pose the mean makes every episode of a condition a bit-identical replay, so a condition
  can only score 0 or 1 and a weak-but-real preference reads as exactly chance.
* ``NETT_HIDDEN_SIZES`` -> the policy head. It stays LINEAR in the campaign driver: the
  Unity archive's head effect (``Compendium1.2.15``, linear 0.516 -> [64,64] 0.724, n=10)
  did NOT transfer -- matched Isaac arms gave 0.804 vs 0.742 at Welch p=0.51. It was
  briefly defaulted to [64,64] on 2026-07-30 and reverted the same day.

The campaign-driver checks are source-text tests because importing ``campaign_train``
needs Isaac on the path; that is the same constraint as ``test_campaign_retest.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from nett_skrl.brain.trainer import eval_stochastic_enabled

_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"


def _train_source() -> str:
    return _TRAIN.read_text()


def test_eval_is_stochastic_by_default(monkeypatch):
    monkeypatch.delenv("NETT_EVAL_STOCHASTIC", raising=False)
    assert eval_stochastic_enabled() is True, (
        "test-time actions must be SAMPLED by default (2026-07-30); the policy mean "
        "collapses every episode of a condition into one bit-identical replay")


def test_the_mean_action_is_still_reachable(monkeypatch):
    """The old protocol must stay available, or the 9-model sweep is unreproducible."""
    monkeypatch.setenv("NETT_EVAL_STOCHASTIC", "0")
    assert eval_stochastic_enabled() is False


def test_eval_stochastic_accepts_the_word_forms(monkeypatch):
    for value in ("true", "TRUE", "yes", "1"):
        monkeypatch.setenv("NETT_EVAL_STOCHASTIC", value)
        assert eval_stochastic_enabled() is True, value
    for value in ("false", "no", "0", ""):
        monkeypatch.setenv("NETT_EVAL_STOCHASTIC", value)
        assert eval_stochastic_enabled() is False, value


def test_campaign_train_reads_the_flag_through_the_accessor():
    """One source of truth: the driver RECORDS this flag as the run's provenance."""
    src = _train_source()
    assert "eval_stochastic_enabled()" in src
    assert 'os.environ.get("NETT_EVAL_STOCHASTIC"' not in src, (
        "campaign_train must not re-parse NETT_EVAL_STOCHASTIC with its own default "
        "literal -- it would record a protocol other than the one trainer.py applied")


def test_entropy_is_overridable_but_defaults_to_the_validated_value():
    """0.01 is the SB3/Unity replication value and must not drift when the knob is used.

    It is exposed (NETT_ENTROPY) because a side-locked policy is a collapsed one, and the
    driver must RECORD it -- a run's numbers are not interpretable without it.
    """
    src = _train_source()
    assert 'os.environ.get("NETT_ENTROPY", "0.01")' in src
    # Twice: once in the PPO config, once in the timing JSON. A hardcoded 0.01 left in the
    # config would silently ignore the knob; a missing JSON entry would leave the run
    # unlabelled, which is the provenance failure NETT_EVAL_STOCHASTIC already cost once.
    assert src.count('"entropy_loss_scale": entropy') == 2, (
        "entropy must both DRIVE the PPO config and be RECORDED in campaign_timing.json")


def test_campaign_train_defaults_to_a_64_64_policy_head():
    """owner 2026-10-27: policy/value head default [64,64] (was LINEAR, "").

    Workspace PROGRAM_BIND16 HEADRL: on frozen offline-CLTT encoders the linear head gave 0 / 6 binders under RL
    and [64, 64] 6 / 6. Linear stays one env value away ("[]"); unset or "" is now [64, 64], except beside
    NETT_FEATURE_ATTN, whose attention head stands alone as before. (Reverted once on 2026-07-30 at Welch p=0.51,
    n=8/arm -- SIDE_LOCK_INVESTIGATION Phase 5, NEXT_STEPS.md §G5.)
    """
    src = _train_source()
    assert 'os.environ.get("NETT_HIDDEN_SIZES", "").strip()' in src
    assert 'hidden_sizes = [] if os.environ.get("NETT_FEATURE_ATTN", "").strip() else [64, 64]' in src, (
        "the campaign driver's policy head defaults to [64, 64] (owner 2026-10-27); the old linear default is \"[]\"")
    assert 'if _hs.strip() else []' not in src, "the pre-10-27 linear-default parse is back"


def _hidden_sizes_parse():
    """The driver's own `_hs = ...` read and the `if not _hs:` block after it, as source."""
    for node in ast.walk(ast.parse(_train_source())):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body[:-1]):
            if isinstance(stmt, ast.Assign) and ast.unparse(stmt.targets[0]) == "_hs":
                nxt = body[i + 1]
                assert isinstance(nxt, ast.If) and ast.unparse(nxt.test) == "not _hs", ast.unparse(nxt)
                assigned = {ast.unparse(n.targets[0]) for n in ast.walk(nxt) if isinstance(n, ast.Assign)}
                assert "hidden_sizes" in assigned, assigned
                return ast.unparse(stmt) + "\n" + ast.unparse(nxt)
    raise AssertionError("no `_hs = ...` read in campaign_train.py")


def test_the_hidden_sizes_parse_covers_linear_and_multi_layer(monkeypatch):
    """Evaluates the driver's own parse (the env read and its `if` block), so no branch can rot.

    owner 2026-10-27: head default [64,64]. Was: "" = linear (the default), "64,64" =
    a head arm. Now unset / "" = [64, 64], "[]" = linear, "64,64" and "[64,64]" both parse, and NETT_FEATURE_ATTN
    with NETT_HIDDEN_SIZES unset keeps the attention head alone (no trunk). An explicit size beside
    NETT_FEATURE_ATTN parses as given here and is refused later by feature_attn_from_env.
    """
    import os

    parse = _hidden_sizes_parse()
    cases = ((None, None, [64, 64]), ("", None, [64, 64]), ("  ", None, [64, 64]), ("[]", None, []),
             ("64,64", None, [64, 64]), ("[64,64]", None, [64, 64]), ("[64, 64]", None, [64, 64]),
             ("64, 64, 64", None, [64, 64, 64]), ("128", None, [128]),
             (None, "1", []), ("", "1", []), ("64,64", "1", [64, 64]))
    for hidden, attn, expected in cases:
        for name, value in (("NETT_HIDDEN_SIZES", hidden), ("NETT_FEATURE_ATTN", attn)):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        scope = {}
        exec(parse, {"os": os}, scope)  # noqa: S102 - the driver's own lines, read from disk
        assert scope["hidden_sizes"] == expected, (hidden, attn)


def test_locomotion_is_opt_in_and_the_default_body_is_unchanged():
    """NETT_LOCOMOTION (2026-09-27, researcher2) exists to run Unity rA10's kinematic body.

    Unset, the driver must NOT pass ``locomotion`` at all, so ``Environment``'s own default
    governs and every existing arm keeps its body. A literal default here (even "wheeled")
    would be a second source of truth that can drift from the constructor's.
    """
    import inspect
    from nett_skrl.environment.environment import Environment

    src = _train_source()
    assert '**({"locomotion": os.environ["NETT_LOCOMOTION"]}' in src
    assert 'if os.environ.get("NETT_LOCOMOTION") else {})' in src
    assert 'os.environ.get("NETT_LOCOMOTION", ' not in src, "no default literal in the driver"
    assert inspect.signature(Environment.__init__).parameters["locomotion"].default == "wheeled"


def test_kinematic_is_passed_to_the_cfg_constructor_not_set_afterwards():
    """U13-KIN FIT probe, 2026-09-28: a cfg built with the default MotorCfg has already run
    the wheeled overrides in __post_init__ (speed 1.5, radius 3.25), and setting
    locomotion="kinematic" afterwards does not undo them -- every test episode started at
    y=-14.500 instead of Unity's -15. The construction site must hand the mode to
    NETTEnvCfg itself, and only when it is not the wheeled default.
    """
    import inspect
    from nett_skrl.environment import environment as env_mod

    src = inspect.getsource(env_mod)
    guard = 'if getattr(self, "locomotion", "wheeled") != "wheeled":'
    passed = 'cfg_kwargs["motor"] = MotorCfg(locomotion=self.locomotion)'
    built = "cfg = NETTEnvCfg(**cfg_kwargs)"
    assert src.count(built) == 1, "exactly one NETTEnvCfg construction site"
    assert guard in src and passed in src
    assert src.index(guard) < src.index(passed) < src.index(built), "mode must reach the constructor"
