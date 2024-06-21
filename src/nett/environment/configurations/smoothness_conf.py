from environment.configs import NETTConfig

class Smoothness(NETTConfig):
    """
    NETT configuration for Smoothness.

    Args:
        object (str or list[str], optional): The object(s) to be used. Defaults to ["obj1"].
        temporal (str or list[str], optional): The temporal condition(s) to be used. Defaults to ["norm", "scram"].

    Attributes:
        num_conditions (int): The number of conditions for the configuration.
    """

    def __init__(self,
                 object: str | list[str] = ["obj1"],
                 temporal: str | list[str] = ["norm", "scram"]) -> None:
        """Constructor method
        """
        super().__init__(param_defaults=self.defaults,
                         object=object,
                         temporal=temporal)

    @property
    def num_conditions(self) -> int:
        """
        Get the number of conditions for the configuration.

        Returns:
            int: The number of conditions.
        """
        return 5
