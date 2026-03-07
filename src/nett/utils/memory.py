from typing import Optional
from pynvml import (
    nvmlInit,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetCount,
    c_nvmlMemory_t,
    c_nvmlMemory_v2_t,
    nvmlShutdown,
)


import threading


def singleton(cls):
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

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
        maxMemory: int = 0
        maxMemoryDevice: int = None
        for device in devices:
            memory = self.get_free_memory(device)
            if memory > maxMemory:
                maxMemory = memory
                maxMemoryDevice = device
        return maxMemoryDevice, maxMemory

    @staticmethod
    def validate_devices(devices: Optional[list[int]]) -> list[int]:
        available_devices: list[int] = list(range(nvmlDeviceGetCount()))
        if devices is None:
            return available_devices
        elif isinstance(devices, list) and set(devices).issubset(available_devices):
            return devices
        else:
            raise ValueError(
                f"Custom device list lists unknown devices. Available devices are: {available_devices}"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # Don't suppress exceptions
