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
    ("CNN2F+CLTT-Ref", {"NETT_AUX_CLTT_REF_DIAG": "1"}),
    ("CNN2F+CLTT-Ref", {"NETT_AUX_CLTT_REF_DIAG": "1", "NETT_DECOUPLE_ENCODER": "1"}),
    ("CNN2F+EoO", {}),
    ("CNN2F+EoO", {"NETT_DECOUPLE_ENCODER": "1"}),
    ("CNN2F+GWM", {}),
    ("CNN2F+GWM", {"NETT_DECOUPLE_ENCODER": "1"}),
    ("CNN+EoO-Dual", {"NETT_EXPERT_FLOW": "1"}),
    ("CNN+GWM-Dual", {"NETT_EXPERT_FLOW": "1"}),
    ("CNN+GWM-Dual", {"NETT_EXPERT_FLOW": "1", "NETT_DECOUPLE_ENCODER": "1"}),
    ("CNN2F+GWM-Seg", {"NETT_EXPERT_FLOW": "1"}),
    ("CNN2F+SlotContrast", {"NETT_SLOTC_SLOTS": "4", "NETT_SLOTC_DIAG": "1"}),
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
