import torch

from netts import Brain, Body, Environment
from netts.utils.gpu import getFirstAvailable

class Benchmark:
    def __init__(self, Brain, Body, Environment) -> None:
        pass
    
    # TO DO (v0.3): Make memory dynamic, depending on the size of the model
    # this is because different models require different GPU memory
    def run(self, save_dir: str, device: list[int] | int | None = None):
        self.save_dir = save_dir
        self.device_num = getFirstAvailable(attempts=5, interval=5, maxMemory=0.5, verbose=True)
        print(self.device_num)
        torch.cuda.set_device(self.device_num[0])
        assert torch.cuda.current_device() == self.device_num[0]

    # TO DO (ideally before v1.0)
    def publish(self):
        raise NotImplementedError