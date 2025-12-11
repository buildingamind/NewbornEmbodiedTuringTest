"""GPU memory management utilities using NVIDIA Management Library."""

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


def singleton(cls):
    """
    Decorator to ensure a class has only one instance.

    Args:
        cls: The class to decorate.

    Returns:
        Function that returns the singleton instance.
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class MemoryManager:
    """
    Manages GPU memory allocation and monitoring using NVIDIA Management Library.

    Provides methods to query GPU memory status and select devices with most available memory.
    Implements singleton pattern to ensure only one instance exists.
    """

    def __init__(self) -> None:
        """Initialize the NVIDIA Management Library."""
        nvmlInit()

    def close(self) -> None:
        """Shutdown the NVIDIA Management Library."""
        nvmlShutdown()

    @staticmethod
    def get_memory_status(device_id: int) -> c_nvmlMemory_t | c_nvmlMemory_v2_t:
        """
        Get memory status for a specific GPU device.

        Args:
            device_id: GPU device index.

        Returns:
            Memory information structure containing total, free, and used memory.
        """
        return nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_id))

    def get_free_memory(self, device_id: int) -> float:
        """
        Get free memory for a specific GPU device.

        Args:
            device_id: GPU device index.

        Returns:
            Amount of free memory in bytes.
        """
        return self.get_memory_status(device_id).free

    def get_most_free_gpu(self, devices: list[int]) -> tuple[int, float]:
        """
        Find the GPU with the most free memory from a list of devices.

        Args:
            devices: List of GPU device indices to check.

        Returns:
            Tuple of (device_id, free_memory) for the device with most free memory.
        """
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
        """
        Validate that requested GPU devices are available.

        Args:
            devices: List of GPU device indices to validate, or None to return all available.

        Returns:
            List of valid device indices.

        Raises:
            ValueError: If any requested device is not available.
        """
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
        if exc_type is None:
            return False
        # An exception occurred
        print(f"Exception type: {exc_type}")
        print(f"Exception value: {exc_val}")
        # Optionally print traceback using traceback module
        import traceback

        traceback.print_tb(exc_tb)
        return True
