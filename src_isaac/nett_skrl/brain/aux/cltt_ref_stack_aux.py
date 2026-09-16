"""CLTT-Ref with WHOLE temporally-offset stacks, for encoders that read motion.

⭐⭐⭐ WHY THIS EXISTS. ``cltt_ref`` is reference-faithful and, as a direct consequence, STATIC:
``cltt_views.current_frame_stack`` takes the observation's CURRENT RGB frame and repeats it across
the host encoder's T slots, so the two positive views can never share a frame. Its own docstring
names the cost -- *"it is a static view, so a motion encoder sees no within-view motion"* -- and for
``compact_3dcnn`` that cost is total: its first ``Conv3d`` has kernel depth = num_frames and then
``.squeeze(2)``, so the layer that exists ONLY to read motion is handed a constant in every
auxiliary view. The arm would train the spatial pathway and nothing else.

⇒ This variant keeps every other thing about ``cltt_ref`` identical -- the projector, the
temperature, the reset-aware contiguous sampler, one-stream negatives, backbone gradients, the
NT-Xent with an exactly masked diagonal -- and changes exactly one line: the views are the WHOLE
prepared stacks, unmodified. The positive pair is then two disjoint MOTION SEGMENTS,
``[f(t-1), f(t)]`` against ``[f(t+1), f(t+2)]``, and the temporal kernel receives real gradient.

⛔ AND THE GUARD THE STATIC ADAPTER MADE UNNECESSARY IS NECESSARY AGAIN. With whole stacks,
``stack(t)`` and ``stack(t+1)`` SHARE a frame at T=2, which lets the encoder match on a literally
identical frame instead of learning anything. Offsets must be multiples of the stack depth. The
depth is DISCOVERED at runtime by the parent (``shape[1] // 3``), never passed in, so the guard
fires on the realised geometry rather than on a declared one. Default offsets are therefore "2,4",
the values the pre-``088a785`` implementation used, not the reference's "1,2".

⚠ WHAT THIS IS NOT. It is NOT more faithful to Schneider et al.; it is LESS. The reference contrasts
single frames and this contrasts short clips, so a ``*-CLTT-Stack`` result may not be compared to a
``*-CLTT-Ref`` result as though one were a better-run version of the other. They are two objectives.
The pair exists to separate "a contrastive objective helps this encoder" from "a contrastive
objective *over motion* helps this encoder", which no single arm can do.

⚠ AND THE HAZARD ``cltt_ref`` ALREADY CARRIES IS NOT REMOVED, IT IS MOVED. Contiguous one-stream
negatives make every negative a near-in-time frame of the SAME two-object world; NT-Xent then pushes
apart views that may show the same object at a different viewpoint. Over whole stacks the anchors
are clips rather than frames, so the near-duplicate pressure is if anything HIGHER. If this variant
trails ``cltt_ref``, that is the first place to look, and ``vicreg`` (no negatives) is the designed
alternative.
"""

from __future__ import annotations

from .cltt_ref_aux import CLTTReferenceAuxLoss


class CLTTReferenceStackAuxLoss(CLTTReferenceAuxLoss):
    """``cltt_ref`` with whole-stack views and a stack-aligned offset guard."""

    OFFSETS_ENV = "NETT_AUX_CLTT_STACK_OFFSETS"
    DEFAULT_OFFSETS = "2,4"

    def _make_views(self, views, encoder):
        # ⛔ THE GUARD IS HERE AND NOT IN __init__ BECAUSE T IS NOT KNOWN AT __init__.
        # The parent discovers the stack depth from the first prepared batch. A guard that ran
        # earlier would have to trust a DECLARED num_frames, and a declared depth that disagrees
        # with the realised one is exactly the failure this is protecting against.
        # [[a-guard-protects-the-line-it-is-on]]
        T = self.num_frames
        if T and any(k % T for k in self.offsets):
            raise ValueError(
                f"{type(self).__name__}: realised stack depth T={T}, offsets={self.offsets}. "
                "With whole-stack views, an offset that is not a multiple of T makes the two "
                "positive views share a literally identical frame, which the encoder can match "
                "on without learning anything. Set "
                f"{self.OFFSETS_ENV} to multiples of T={T} (e.g. "
                f"{','.join(str(T * i) for i in (1, 2))})."
            )
        return list(views)
