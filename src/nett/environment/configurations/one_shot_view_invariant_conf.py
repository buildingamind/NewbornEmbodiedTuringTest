from environment.configs import NETTConfig

class OneShotViewInvariant(NETTConfig):
    """
    NETT configuration for One-Shot View Invariant.

    Args:
        object (str | list[str]): The object(s) to be used. Defaults to ["fork", "ship"].
        range (str | list[str]): The range(s) to be used. Defaults to ["360", "small", "1"].
        view (str | list[str]): The view(s) to be used. Defaults to ["front", "side"].

    Raises:
        ValueError: If any parameter value is not a value or subset of the default values.
    """

    def __init__(self,
                 object: str | list[str] = ["fork", "ship"],
                 range: str | list[str] = ["360", "small", "1"],
                 view: str | list[str] = ["front", "side"]) -> None:
        """Constructor method
        """
        super().__init__(param_defaults=self.defaults,
                         object=object,
                         range=range,
                         view=view)

    @property
    def num_conditions(self) -> int:
        """
        Get the number of conditions for the configuration.

        Returns:
            int: The number of conditions.
        """
        return 50
