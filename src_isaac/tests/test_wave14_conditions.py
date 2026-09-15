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
    ("CNN2F", {"NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN2F", {"NETT_DECOUPLE_ENCODER": "1", "NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN2F+LumNorm", {"NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN2F+CLTT-Ref", {"NETT_AUX_CLTT_REF_DIAG": "1", "NETT_AUX_BATCH": "240", "NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN2F+CLTT-Ref", {"NETT_AUX_CLTT_REF_DIAG": "1", "NETT_DECOUPLE_ENCODER": "1", "NETT_AUX_BATCH": "240", "NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN2F+EoO", {"NETT_AUX_BATCH": "32", "NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN2F+EoO", {"NETT_DECOUPLE_ENCODER": "1", "NETT_AUX_BATCH": "32", "NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN2F+GWM", {"NETT_AUX_BATCH": "32", "NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN2F+GWM", {"NETT_DECOUPLE_ENCODER": "1", "NETT_AUX_BATCH": "32", "NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN+EoO-Dual", {"NETT_EXPERT_FLOW": "1", "NETT_AUX_BATCH": "32", "NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN+GWM-Dual", {"NETT_EXPERT_FLOW": "1", "NETT_AUX_BATCH": "32", "NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN+GWM-Dual", {"NETT_EXPERT_FLOW": "1", "NETT_DECOUPLE_ENCODER": "1", "NETT_AUX_BATCH": "32", "NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN2F+GWM-Seg", {"NETT_EXPERT_FLOW": "1", "NETT_MEMORY_DEVICE": "cuda:0"}),
    ("CNN2F+SlotContrast", {"NETT_SLOTC_SLOTS": "4", "NETT_SLOTC_DIAG": "1", "NETT_AUX_BATCH": "32", "NETT_MEMORY_DEVICE": "cuda:0"}),
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


def test_every_aux_condition_pins_its_batch():
    """Every aux row states its B; a row with no aux loss must not, since nothing would read it."""
    for model, env in WAVE:
        if ct.MODELS[model].get("aux"):
            assert "NETT_AUX_BATCH" in env, (
                f"{model} declares aux={ct.MODELS[model]['aux']!r} and does not pin "
                f"NETT_AUX_BATCH; it would silently take that loss's own default")
        else:
            assert "NETT_AUX_BATCH" not in env, (
                f"{model} declares no aux loss but pins NETT_AUX_BATCH -- nothing reads it")


def test_the_batch_is_standardised_WITHIN_a_loss_and_not_across_losses():
    """⛔ B IS NOT COMMENSURABLE ACROSS LOSSES, so equalising it controls nothing.

    An earlier version of this wave pinned every aux row to one value, on the reasoning that
    an unequal B would confound the aux-family comparison. That was wrong, and the reason is
    worth keeping: **B names a different quantity in each loss.**

        cltt_ref       negatives per anchor = 2B - 2          (a contrastive denominator)
        slot_contrast  negatives = B x K                      (slots merge into the batch dim)
        eoo / gwm      NO negatives at all                    (a photometric minibatch)

    So equal B does NOT give equal negatives -- at B=32 with K=4, slot_contrast has 128
    negatives where cltt_ref would have 62 -- and for the photometric losses the contrastive
    framing does not apply in the first place. Equalising a number that means three different
    things buys the APPEARANCE of control and mis-specifies at least one arm to get it. B is a
    component of each loss's own specification, not a nuisance parameter to be held fixed.

    The invariant that DOES matter is per-loss constancy, which is what this asserts.
    """
    by_loss: dict[str, set[str]] = {}
    for model, env in WAVE:
        aux = ct.MODELS[model].get("aux")
        if aux:
            by_loss.setdefault(aux, set()).add(env["NETT_AUX_BATCH"])
    assert by_loss, "no aux conditions found -- the assertion below would be vacuous"
    for aux, values in by_loss.items():
        assert len(values) == 1, f"aux {aux!r} runs at more than one batch across the wave: {values}"
    # And the wave must actually span more than one value, or this test proves nothing.
    assert len({next(iter(v)) for v in by_loss.values()}) > 1, (
        "every loss ended up at the same B -- this test cannot distinguish "
        "'standardised within loss' from 'standardised across losses'")


def test_the_routing_pairs_hold_their_batch_identical():
    """⛔ THE ONE PLACE B MUST MATCH IS INSIDE A PAIR. rows 04/05, 06/07, 08/09 and 11/12 differ
    ONLY in gradient routing, so a B that moved with the routing would confound the wave's
    central contrast with batch size -- the real version of the concern that wrongly motivated
    a global pin."""
    pairs = [(3, 4), (5, 6), (7, 8), (10, 11)]     # 0-indexed into WAVE
    for i, j in pairs:
        (mi, ei), (mj, ej) = WAVE[i], WAVE[j]
        assert mi == mj, f"pair ({i},{j}) is not the same model: {mi} vs {mj}"
        assert ei["NETT_AUX_BATCH"] == ej["NETT_AUX_BATCH"], (
            f"{mi}: standard B={ei['NETT_AUX_BATCH']} vs decoupled B={ej['NETT_AUX_BATCH']}")
        assert ("NETT_DECOUPLE_ENCODER" in ej) and ("NETT_DECOUPLE_ENCODER" not in ei), (
            f"pair ({i},{j}) is not a routing contrast")


def test_the_pinned_batch_is_reachable_at_this_protocol():
    """⛔ cltt_ref's OWN DEFAULT IS UNREACHABLE HERE, which is why it is the one loss whose value
    had to change rather than simply be stated. memory_size = rollouts//scope = 8000//32 = 250;
    its offsets are (1,2), so avail = 248 and B can never exceed it. Left at its 512 default the
    arm runs ~248 while its config records 512 -- a resolved value diverging silently from the
    passed one. A pin ABOVE the ceiling recreates exactly that, so every pin is checked."""
    rollouts, scope, max_offset = 8000, 32, 2
    ceiling = rollouts // scope - max_offset
    assert ceiling == 248
    for model, env in WAVE:
        if not ct.MODELS[model].get("aux"):
            continue
        b = int(env["NETT_AUX_BATCH"])
        assert 2 <= b <= ceiling, (
            f"{model}: B={b} is outside [2, {ceiling}] -- agent_factory refuses <2 (an empty mean "
            f"is NaN forward and ZERO backward), and above the ceiling the config would record a "
            f"value the run cannot reach")


def test_every_row_places_the_rollout_buffer_on_the_gpu():
    """OWNER DECISION 2026-09-15: the buffer goes on VRAM, not host RAM.

    ⛔ cuda:0, NEVER a bare "cuda". agent_factory compares torch.device(mem) != torch.device(dev),
    and torch.device("cuda") != torch.device("cuda:0") is TRUE -- a bare value silently selects
    the CPU hybrid path, i.e. the exact opposite of the decision, with nothing reporting it.
    ⛔ AND NEVER cuda:1..cuda:7 EITHER, WHICH IS THE LIKELIER EDIT. The owner asked for the wave
    spread over eight cards, two rows each, and the obvious-looking way to write that -- bumping
    this value per row -- CRASHES every row not on card 0. launch_arm.sh:284 exports
    CUDA_VISIBLE_DEVICES="$GPU" and NETT_DEVICE="$GPU" together, so each arm sees exactly ONE card
    and that card is index 0 inside the process. Measured: under CUDA_VISIBLE_DEVICES=3,
    device_count() is 1, cuda:0 is the card nvidia-smi calls 3, and cuda:3 raises
    "RuntimeError: CUDA error: invalid device ordinal". nett_skrl/runtime/device.py is the module
    written for this collision; torch_device_index() returns 0 whenever one card is visible.
    ⇒ "cuda:0" here does not mean "card 0". It means "the card this arm was given". The card is a
    LAUNCH argument (launch_arm.sh --gpu N), not row content -- the per-row assignment lives in
    the plan's section 6c and the queue header, where a scheduling fact belongs.
    ⚠ The failure modes are asymmetric: a physical index WITH the pin errors loudly, but a
    non-zero index WITHOUT the pin does not error at all -- Kit's usdrt scenegraph hangs at the
    Fabric XFormPrimView, 26-71 min observed and indefinite in principle.
    """
    for model, env in WAVE:
        got = env.get("NETT_MEMORY_DEVICE")
        assert got == "cuda:0", (
            f"{model}: NETT_MEMORY_DEVICE={got!r}, want 'cuda:0'. If this was an attempt to "
            f"spread the wave across cards, that belongs in `launch_arm.sh --gpu N` -- {got!r} "
            f"would raise 'invalid device ordinal' inside the arm, because the launcher pins the "
            f"card with CUDA_VISIBLE_DEVICES and the only visible device is always cuda:0.")


def _queue_rows():
    """The wave's rows from the fleet workspace, or None when it is not on this host.

    The workspace is a SEPARATE repository, so it is present on fleet nodes and absent from a
    bare clone. Returning None (-> skip) rather than failing keeps this suite runnable anywhere,
    while still reconciling wherever the file exists.
    """
    import os
    for base in (os.environ.get("NETT_WORKSPACE"),
                 Path.home() / "code" / "isaac" / "NETT_Global_Workspace"):
        if not base:
            continue
        f = Path(base) / "queue" / "proposals.yaml"
        if f.exists():
            import yaml
            rows = yaml.safe_load(f.read_text())
            wave = [r for r in rows if str(r.get("id", "")).startswith("proposed-wave14")]
            return wave or None
    return None


def test_WAVE_matches_the_queue_rows_it_duplicates():
    """⛔ THIS LIST IS A SECOND COPY OF THE QUEUE ROWS, AND THAT IS THE HAZARD IT MUST ANSWER FOR.

    Two places stating the same fact diverge the moment one is edited alone. It already happened
    on this wave: row 03 was repointed from an env flag to the CNN2F+LumNorm model entry, and the
    header that classified the rows was left asserting the pre-correction world. A duplicate is
    acceptable only with a reconciliation, so here it is.

    Skips where the workspace is absent -- and says so, rather than passing quietly, because a
    silent skip is how a reconciliation stops running without anyone noticing.
    """
    rows = _queue_rows()
    if rows is None:
        pytest.skip("fleet workspace not on this host; nothing to reconcile against "
                    "(set NETT_WORKSPACE to point at it)")
    assert len(rows) == len(WAVE), f"queue has {len(rows)} wave rows, WAVE has {len(WAVE)}"
    for row, (model, env) in zip(rows, WAVE):
        assert row["model"] == model, f"{row['id']}: queue says {row['model']!r}, WAVE says {model!r}"
        # Compare as strings: YAML yields ints for 1/32/240, the environment only ever sees text.
        got = {k: str(v) for k, v in (row["env"] or {}).items()}
        assert got == env, f"{row['id']}: queue env {got} != WAVE env {env}"
