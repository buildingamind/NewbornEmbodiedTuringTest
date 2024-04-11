from abc import abstractmethod

class BaseReward:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def calculate(self, policy, observations, actions, rewards):
        raise NotImplementedError("Must be implemented by the child class")