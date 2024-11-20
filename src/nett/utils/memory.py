from typing import Optional
from pynvml import (
    nvmlInit,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetCount,
    c_nvmlMemory_t,
    c_nvmlMemory_v2_t,
)


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class MemoryManager:
    def __init__():
        nvmlInit()

    @staticmethod
    def get_memory_status(device_id: int) -> c_nvmlMemory_t | c_nvmlMemory_v2_t:
        return nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_id))

    def get_free_memory(self, device_id: int) -> int:
        return self.get_memory_status(device_id).free

    def get_used_memory(self, device_id: int) -> int:
        return self.get_memory_status(device_id).used

    def get_most_free_gpu(self, devices: list[int]) -> int:
        maxMemory: int = 0
        maxMemoryDevice: int = None
        for device in devices:
            memory = self.get_free_memory(device)
        if memory > maxMemory:
            maxMemory = memory
            maxMemoryDevice = device
        return maxMemoryDevice

    def get_free_memory_by_device(self, devices: list[int]) -> list[dict[str, int]]:
        return [
            {"device": device, "memory": self.get_free_memory(device)}
            for device in devices
        ]

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
