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

import torch
from skrl.memories.torch import RandomMemory


class HybridDeviceMemory(RandomMemory):
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


class Uint8StatesMemory(RandomMemory):
    """``RandomMemory`` that stores the image **observation** buffers as uint8.

    Image observations are kept at 0–255 all the way to the encoder, whose
    ``prepare_image_tensor`` does the ``/255`` float conversion at the model
    input. skrl's PPO defaults these buffers to ``float32`` (4 bytes/elem), so at
    res256/2-frame the VRAM buffer is ~12.6 GiB/brain and must spill to CPU.
    Storing them as uint8 is **numerically transparent** (values are integer
    0–255; the only consumer, the encoder, normalizes at input; skrl's
    state/observation preprocessors are the identity here) and cuts the buffer to
    ~1/4, letting the 2-frame buffer sit on-GPU (no CPU<->GPU transfer stall).

    NOTE: in this skrl version the image lives in ``observations`` /
    ``next_observations`` (the policy input); ``states`` / ``next_states`` are the
    asymmetric-critic slots and are empty when ``state_space is None`` (NETT's
    case). We coerce all four image-observation names, but ONLY when the tensor
    actually carries image data (``size`` is a real Space, not a scalar/None) —
    scalars (rewards, values, log_prob, returns, advantages) and actions keep
    their float dtype.
    """

    _IMAGE_NAMES = ("observations", "next_observations", "states", "next_states")

    def create_tensor(self, name, *, size, dtype=None, keep_dimensions=False) -> bool:
        # size is a gymnasium.Space for image tensors and an int (1) for scalars.
        if name in self._IMAGE_NAMES and size is not None and not isinstance(size, int):
            dtype = torch.uint8
        return super().create_tensor(
            name, size=size, dtype=dtype, keep_dimensions=keep_dimensions
        )
