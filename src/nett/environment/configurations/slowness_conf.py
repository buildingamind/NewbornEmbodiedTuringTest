from environment.configs import NETTConfig

class Slowness(NETTConfig):
    """
    NETT configuration for Slowness.

    Args:
        experiment (str | list[int], optional): The experiment(s) to be used. Defaults to [1, 2].
        object (str | list[str], optional): The object(s) to be used. Defaults to ["obj1", "obj2"].
        speed (str | list[str], optional): The speed(s) to be used. Defaults to ["slow", "med", "fast"].

    Raises:
        ValueError: If any parameter value is not a value or subset of the default values.
    """

    def __init__(self,
                 experiment: str | list[int] = [1, 2],
                 object: str | list[str] = ["obj1", "obj2"],
                 speed: str | list[str] = ["slow", "med", "fast"]) -> None:
        """Constructor method
        """
        super().__init__(param_defaults=self.defaults,
                         experiment=experiment,
                         object=object,
                         speed=speed)

    @property
    def num_conditions(self) -> int:
        """
        Get the number of conditions for the configuration.

        Returns:
            int: The number of conditions.
        """
        if self.params["experiment"] == "1":
            return 5
        return 13
