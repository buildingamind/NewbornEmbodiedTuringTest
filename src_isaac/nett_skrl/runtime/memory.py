"""GPU memory accounting helpers backed by NVML.

Used by the NETT scheduler to (a) pick the least-loaded GPU at submission time,
(b) gate task submission on whether the most-free GPU still has room for the
estimated ``task_memory``, and (c) supply a live free-memory baseline for the
dry-run estimator.
"""

from __future__ import annotations

import threading
from typing import Optional

from pynvml import (
    c_nvmlMemory_t,
    c_nvmlMemory_v2_t,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
    nvmlShutdown,
)


def singleton(cls):
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    def reset_instance():
        with lock:
            instances.pop(cls, None)

    get_instance.reset_instance = reset_instance
    return get_instance


@singleton
class MemoryManager:
    def __init__(self) -> None:
        nvmlInit()

    def close(self) -> None:
        nvmlShutdown()

    @staticmethod
    def get_memory_status(device_id: int) -> c_nvmlMemory_t | c_nvmlMemory_v2_t:
        return nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_id))

    def get_free_memory(self, device_id: int) -> float:
        return self.get_memory_status(device_id).free

    def get_most_free_gpu(self, devices: list[int]) -> tuple[int, float]:
        if not devices:
            raise ValueError("devices list cannot be empty")
        max_memory = -1.0
        max_device = devices[0]
        for device in devices:
            mem = self.get_free_memory(device)
            if mem > max_memory:
                max_memory = mem
                max_device = device
        return max_device, max_memory

    @staticmethod
    def validate_devices(devices: Optional[list[int]]) -> list[int]:
        available = list(range(nvmlDeviceGetCount()))
        if devices is None:
            return available
        if isinstance(devices, list) and set(devices).issubset(available):
            return devices
        raise ValueError(
            f"Custom device list lists unknown devices. Available devices are: {available}"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        MemoryManager.reset_instance()
        return False
