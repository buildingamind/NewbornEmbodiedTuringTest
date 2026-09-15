"""The 14-condition wave: every row must RESOLVE, and every manipulation must REACH the arm.

⛔ WHY THIS FILE EXISTS. A queue row naming a knob that nothing reads launches happily, logs
the experimental label, and runs the CONTROL. There is no symptom: the arm trains, scores, and
lands in the results table wearing a name for a manipulation that never happened. It was not
hypothetical -- row 03 of this wave was first written as `env: {NETT_BODY_WRAPPERS: lumnorm}`,
and NETT_BODY_WRAPPERS is read NOWHERE in nett_skrl. Body wrappers come from the MODELS entry,
so the model NAME has to carry the manipulation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "examples"))
_spec = importlib.util.spec_from_file_location("_campaign_train", SRC / "examples" / "campaign_train.py")
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)

from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES  # noqa: E402
from nett_skrl.body.wrappers.registry import wrapper_list  # noqa: E402

#: (model key, env knobs) for the 14 conditions, as the proposal rows declare them.
WAVE = [
    ("CNN2F", {}),
    ("CNN2F", {"NETT_DECOUPLE_ENCODER": "1"}),
    ("CNN2F+LumNorm", {}),
    ("CNN2F+CLTT-Ref", {"NETT_AUX_CLTT_REF_DIAG": "1", "NETT_AUX_BATCH": "128"}),
    ("CNN2F+CLTT-Ref", {"NETT_AUX_CLTT_REF_DIAG": "1", "NETT_DECOUPLE_ENCODER": "1", "NETT_AUX_BATCH": "128"}),
    ("CNN2F+EoO", {"NETT_AUX_BATCH": "128"}),
    ("CNN2F+EoO", {"NETT_DECOUPLE_ENCODER": "1", "NETT_AUX_BATCH": "128"}),
    ("CNN2F+GWM", {"NETT_AUX_BATCH": "128"}),
    ("CNN2F+GWM", {"NETT_DECOUPLE_ENCODER": "1", "NETT_AUX_BATCH": "128"}),
    ("CNN+EoO-Dual", {"NETT_EXPERT_FLOW": "1", "NETT_AUX_BATCH": "128"}),
    ("CNN+GWM-Dual", {"NETT_EXPERT_FLOW": "1", "NETT_AUX_BATCH": "128"}),
    ("CNN+GWM-Dual", {"NETT_EXPERT_FLOW": "1", "NETT_DECOUPLE_ENCODER": "1", "NETT_AUX_BATCH": "128"}),
    ("CNN2F+GWM-Seg", {"NETT_EXPERT_FLOW": "1"}),
    ("CNN2F+SlotContrast", {"NETT_SLOTC_SLOTS": "4", "NETT_SLOTC_DIAG": "1", "NETT_AUX_BATCH": "128"}),
]

_PKG_TEXT = "\n".join(p.read_text() for p in (SRC / "nett_skrl").rglob("*.py"))


def test_there_are_fourteen_conditions():
    """⛔ Establish the n. An empty list passes every parametrized assertion below it."""
    assert len(WAVE) == 14


@pytest.mark.parametrize("model,env", WAVE, ids=[f"{i+1:02d}-{m}" for i, (m, _) in enumerate(WAVE)])
def test_the_condition_resolves_end_to_end(model, env):
    assert model in ct.MODELS, f"{model!r} is not a MODELS key"
    spec = ct.MODELS[model]
    aux = spec.get("aux")
    if aux:
        assert aux in AUX_LOSSES, f"{model}: aux {aux!r} is not registered in AUX_LOSSES"
    for w in ct.segmentation_wrappers(spec):
        assert w in wrapper_list, f"{model}: wrapper {w!r} is not in the registry"
    for knob in env:
        assert knob in _PKG_TEXT, (
            f"{model}: env knob {knob} is READ NOWHERE in nett_skrl, so a row setting it "
            f"would run the control under an experimental label")


def test_the_encoder_is_held_constant_across_the_whole_wave():
    """The wave's design is one-factor contrasts against CNN2F. A trunk that differed in one
    condition would confound that condition's aux loss with an architecture change."""
    encoders = {ct.MODELS[m]["encoder"] for m, _ in WAVE}
    assert encoders == {"nature_cnn"}, f"wave spans more than one trunk: {encoders}"


def test_every_condition_is_framestacked():
    """Every aux loss in the wave needs a temporal pair, and the only pair available is the
    stacked channel axis -- there is no next_observations in the PPO sample tuple."""
    for model, _ in WAVE:
        assert ct.MODELS[model]["framestack"] is True, f"{model} is not framestacked"


def test_lumnorm_is_innermost_so_downstream_consumers_see_standardised_frames():
    """⛔ ORDER IS THE MANIPULATION. lumnorm after framestack would standardise a STACK; the
    point is to standardise each raw frame before anything else -- segmenters included -- sees it."""
    order = ct.segmentation_wrappers(ct.MODELS["CNN2F+LumNorm"])
    assert order[0] == "lumnorm", order
    assert order.index("lumnorm") < order.index("framestack"), order


def test_the_control_and_the_lumnorm_arm_differ_in_exactly_one_thing():
    control, lum = ct.MODELS["CNN2F"], ct.MODELS["CNN2F+LumNorm"]
    differing = {k for k in set(control) | set(lum) if control.get(k) != lum.get(k)}
    assert differing == {"pre"}, f"expected only `pre` to differ; got {differing}"


def test_there_is_no_environment_hook_for_body_wrappers():
    """The regression test for the actual mistake: if someone adds an env path for wrappers,
    this fails and the wave's rows must be revisited, because a row could then name a
    manipulation that silently does nothing when the spelling is wrong."""
    assert "NETT_BODY_WRAPPERS" not in _PKG_TEXT
    assert "NETT_WRAPPERS" not in _PKG_TEXT


def test_every_aux_condition_pins_its_batch_to_the_same_value():
    """⛔ B IS NOT ONE NUMBER UNLESS SOMETHING MAKES IT ONE.

    NETT_AUX_BATCH is read at 13 sites with FOUR different defaults. Unpinned, this wave would
    have run cltt_ref at 512 and every other aux loss at 32 -- an 8x spread across rows whose
    entire design is to differ only in the objective, confounding the 3x2 with batch size. And
    every level claim about a contrastive objective depends on B: NT-Xent chance is ln(2B-1).

    This asserts the PROPERTY (every aux row pins it, all to one value) rather than restating
    the values, so a new aux condition added without a pin fails here instead of silently
    inheriting whichever default its loss happens to carry.
    """
    pinned = {}
    for model, env in WAVE:
        if not ct.MODELS[model].get("aux"):
            assert "NETT_AUX_BATCH" not in env, (
                f"{model} declares no aux loss but pins NETT_AUX_BATCH -- nothing would read it")
            continue
        assert "NETT_AUX_BATCH" in env, (
            f"{model} declares aux={ct.MODELS[model]['aux']!r} and does not pin NETT_AUX_BATCH; "
            f"it would take that loss's own default, which is not the same across losses")
        pinned[model] = env["NETT_AUX_BATCH"]
    assert len(set(pinned.values())) == 1, f"aux batch differs across the wave: {pinned}"


def test_the_pinned_batch_is_reachable_at_this_protocol():
    """⛔ cltt_ref's OWN DEFAULT IS UNREACHABLE HERE, which is why a pin was needed rather than
    merely tidy. memory_size = rollouts//scope = 8000//32 = 250; cltt_ref's offsets are (1,2), so
    avail = 248 and B can never exceed it. Left at its 512 default the arm runs ~248 while its
    config records 512 -- a resolved value diverging silently from the passed one.

    A pin ABOVE the ceiling would recreate exactly that, so the pin is checked against it.
    """
    rollouts, scope, max_offset = 8000, 32, 2
    ceiling = rollouts // scope - max_offset
    assert ceiling == 248
    pin = int(dict(WAVE)["CNN2F+CLTT-Ref"]["NETT_AUX_BATCH"])
    assert pin <= ceiling, f"pinned B={pin} exceeds the reachable ceiling {ceiling}"
    assert pin >= 2, "agent_factory refuses B < 2: an empty mean is NaN forward, ZERO backward"
