"""Memory that stores rollout tensors on a cheap, large device (CPU) but returns
reads on the compute device (GPU).

Why: at high ``input_resolution`` (e.g. 256, 2-frame) the per-brain PPO state
buffer is ``rollouts * C*H*W * 4`` bytes ≈ 12.6 GiB. Eight brains cannot keep
that on a 23 GB GPU, so the buffer must live in CPU RAM. But skrl's PPO
``update`` computes GAE and the minibatch losses by mixing tensors read straight
from the memory with on-GPU model/preprocessor outputs, which raises
``Expected all tensors to be on the same device (cuda:0 and cpu)``.

``HybridDeviceMemory`` keeps storage on CPU (small footprint on the GPU) while
``get_tensor_by_name`` and ``sample`` return tensors moved to the compute
device, so the whole PPO update stays on-device. Writes (`set_tensor_by_name`,
`add_samples`) already use ``Tensor.copy_``, which moves cross-device for free,
so they need no override. The big state minibatch is moved to the GPU exactly
when sampled — the same host→device transfer the encoder's forward would do
anyway — so there is no extra copy beyond what training already requires.
"""

from __future__ import annotations

import os

import torch
from skrl.memories.torch import RandomMemory


def uint8_buffer_enabled() -> bool:
    """NETT_UINT8_BUFFER: image observations stored as uint8 in the rollout buffer, on the GPU AND on the
    CPU (owner, workspace DECISIONS §103: "Gpu and cpu both use uint8 for observation buffer"). DEFAULT ON;
    "0" = the old float32 storage on both paths."""
    return os.environ.get("NETT_UINT8_BUFFER", "1") != "0"


def _space_is_0_255(size) -> bool:
    """True iff ``size`` is an image Space whose upper bound exceeds 1 -- the case in which
    ``prepare_image_tensor`` divides a FLOAT observation by 255, exactly as it divides a uint8 one. For a
    space bounded at 1 the float path does NOT divide, so uint8 storage would change the encoder input."""
    high = getattr(size, "high", None)
    if high is None:
        return False
    import numpy as np
    return float(np.max(high)) > 1.0


class _Uint8ImagesMixin:
    """Stores the image observation tensors as uint8 -- only when lossless, and refuses a lossy write.

    Transparent because the only consumer, ``prepare_image_tensor``, maps uint8 x -> x/255 and a float x
    of a 0-255 space -> x/255 too. Two guards keep it transparent instead of assuming it:
    * a Space bounded at 1 (float path does not divide) keeps float storage (``_space_is_0_255``);
    * every write of a floating image tensor into a uint8 slot must be integer-valued in [0, 255], or
      ``add_samples`` raises -- ``copy_`` would otherwise truncate silently.
    """

    _IMAGE_NAMES = ("observations", "next_observations", "states", "next_states")

    def create_tensor(self, name, *, size, dtype=None, keep_dimensions=False) -> bool:
        # size is a gymnasium.Space for image tensors and an int (1) for scalars.
        if name in self._IMAGE_NAMES and size is not None and not isinstance(size, int):
            if _space_is_0_255(size):
                dtype = torch.uint8
                self._uint8_names = getattr(self, "_uint8_names", set()) | {name}
        return super().create_tensor(name, size=size, dtype=dtype, keep_dimensions=keep_dimensions)

    def add_samples(self, **tensors) -> None:
        for name in getattr(self, "_uint8_names", ()):
            t = tensors.get(name)
            if t is not None and t.is_floating_point():
                bad = (t != t.round()) | (t < 0) | (t > 255)
                if bool(bad.any()):
                    v = t[bad].flatten()[:3].tolist()
                    raise ValueError(
                        f"uint8 rollout buffer: '{name}' carries values that are not integers in [0, 255] "
                        f"(e.g. {v}); storing them would truncate. Set NETT_UINT8_BUFFER=0 for this config.")
        return super().add_samples(**tensors)


class _LazyIndexedBatches:
    """skrl's minibatch list, indexed out of the buffer ONE minibatch at a time (owner, workspace DECISIONS §117
    addendum 4: "Code fix to stop duplication of buffer").

    ``RandomMemory.sample_by_index`` builds every minibatch up front -- ``tensors_view[name][batch]`` for all
    ``mini_batches`` batches -- and advanced indexing copies, so the update holds a second, permuted copy of the
    whole rollout beside the buffer (9.6 GB for one 448x280 RGB brain at 25.6k). This holds only the index splits
    and indexes when a minibatch is asked for, so at most the minibatch in use is live. Same ``indexes``, same
    ``np.array_split``, same per-name indexing: each minibatch is bit-identical to skrl's (tests/test_lazy_sample.py).
    Safe because nothing writes the buffer between ``sample`` and the end of the minibatch loop (GAE runs before).
    """

    def __init__(self, memory, names, batches) -> None:
        self._memory = memory
        self._names = list(names)
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, i):
        m, batch = self._memory, self._batches[i]
        return [m.tensors_view[name][batch] if name in m.tensors else None for name in self._names]

    def __iter__(self):
        for i in range(len(self._batches)):
            yield self[i]


class _LazySampleMixin:
    """``sample_by_index`` returning ``_LazyIndexedBatches`` when there is more than one minibatch.
    One minibatch is the whole rollout either way, so that case keeps skrl's eager path."""

    def sample_by_index(self, names, *, indexes, mini_batches=1):
        if mini_batches > 1:
            import numpy as np
            return _LazyIndexedBatches(self, names, np.array_split(indexes, mini_batches))
        return super().sample_by_index(names, indexes=indexes, mini_batches=mini_batches)


class HybridDeviceMemory(_LazySampleMixin, RandomMemory):
    """``RandomMemory`` stored on ``device`` (CPU) but read back on ``compute_device``."""

    def __init__(self, *args, compute_device, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._compute_device = torch.device(compute_device)

    def get_tensor_by_name(self, name: str) -> torch.Tensor:
        return super().get_tensor_by_name(name).to(self._compute_device)

    def sample(self, names, *, batch_size, mini_batches=1, sequence_length=1):
        batches = super().sample(
            names, batch_size=batch_size, mini_batches=mini_batches, sequence_length=sequence_length
        )
        # Move each minibatch to the GPU LAZILY (one at a time during iteration),
        # not all upfront — otherwise the full rollout buffer (~12.6 GiB/brain at
        # res256) lands on the GPU at once and OOMs. The PPO update consumes this
        # as ``for (...) in memory.sample(...)``, so only the current minibatch's
        # GPU copy is live; the previous one is freed as the loop advances.
        return _LazyDeviceBatches(batches, self._compute_device)


class _LazyDeviceBatches:
    """Iterable of minibatches that moves each to ``device`` only when yielded."""

    def __init__(self, batches, device) -> None:
        self._batches = batches
        self._device = device

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        dev = self._device
        for batch in self._batches:
            yield [None if t is None else t.to(dev) for t in batch]


class Uint8StatesMemory(_LazySampleMixin, _Uint8ImagesMixin, RandomMemory):
    """``RandomMemory`` (on the compute device) that stores the image **observation** buffers as uint8.

    Image observations are kept at 0–255 all the way to the encoder, whose
    ``prepare_image_tensor`` does the ``/255`` float conversion at the model
    input. skrl's PPO defaults these buffers to ``float32`` (4 bytes/elem), so at
    res256/2-frame the VRAM buffer is ~12.6 GiB/brain and must spill to CPU.
    Storing them as uint8 cuts the buffer to ~1/4; see ``_Uint8ImagesMixin`` for
    why it is numerically transparent and the two guards that enforce it.

    NOTE: in this skrl version the image lives in ``observations`` /
    ``next_observations`` (the policy input); ``states`` / ``next_states`` are the
    asymmetric-critic slots and are empty when ``state_space is None`` (NETT's
    case). We coerce all four image-observation names, but ONLY when the tensor
    actually carries image data (``size`` is a real Space, not a scalar/None) —
    scalars (rewards, values, log_prob, returns, advantages) and actions keep
    their float dtype.
    """


class Uint8HybridDeviceMemory(_Uint8ImagesMixin, HybridDeviceMemory):
    """The CPU-stored hybrid buffer with uint8 image observations (DECISIONS §103, default on).

    1/4 the host RAM of the float32 hybrid (two-frame 256x160 at 25.6k rollouts: 23.4 -> 5.9 GiB/brain)
    and 1/4 the bytes per minibatch host->device copy. The uint8 minibatch is cast to float on the GPU by
    ``prepare_image_tensor``, as on the GPU-buffer path.
    """


def resolve_memory_device(device) -> str:
    """Where the PPO rollout buffer lives. Rule: **follow the compute device.**

    The buffer belongs wherever the compute is. On a GPU run it stays on that GPU, so
    there is no host<->device copy per minibatch. On a CPU run it stays on the CPU --
    a CPU run must never push its buffer onto a GPU it is not otherwise using.
    Returning ``str(device)`` gives both behaviours from one rule.

    Why not the literal "cuda": ``torch.device("cuda") != torch.device("cuda:0")``, so
    a bare "cuda" fails the equality check in the caller and silently selects the CPU
    hybrid path -- the exact opposite of asking for an on-GPU buffer. Always resolve to
    the concrete device string.

    History: the buffer used to default to CPU unconditionally. That was a workaround
    for res256 (~12.6 GiB/brain at 2-frame, rollouts=8000, which will not fit beside
    Isaac on a 23 GB card), but at the res128 design point it paid a transfer stall for
    nothing and put ~25 GB/proc of host RAM on a 16-job run (408 GB total).

    MEASURED VRAM BUDGET (res128, 2-frame, 500 steps x 16 envs, states+next_states):

        float32 buffer = 5.86 GB/job -> 2 jobs/GPU = 22.7 GB of 23.0   OOM RISK
        uint8   buffer = 1.46 GB/job -> 2 jobs/GPU = 13.9 GB of 23.0   fits

    So an on-GPU buffer REQUIRES the uint8 image states (NETT_UINT8_BUFFER, default on; since
    DECISIONS §103 it also governs the CPU buffer) -- the two are coupled and must be changed together. At higher resolutions re-check
    that arithmetic and set NETT_MEMORY_DEVICE=cpu if it no longer fits.

    NETT_MEMORY_DEVICE overrides explicitly (e.g. "cpu" to spill a large buffer off a
    small card).
    """
    override = os.environ.get("NETT_MEMORY_DEVICE")
    if override:
        return override.strip()
    return str(device)
