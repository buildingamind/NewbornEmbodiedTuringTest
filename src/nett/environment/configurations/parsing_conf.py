from environment.configs import NETTConfig

class Parsing(NETTConfig):
    """
    NETT configuration for Parsing.

    Args:
        background (str | list[str], optional): The background(s) to be used. Defaults to ["A", "B", "C"].
        object (str | list[str], optional): The object(s) to be used. Defaults to ["ship", "fork"].
    """

    def __init__(self,
                 background: str | list[str] = ["A", "B", "C"],
                 object: str | list[str] = ["ship", "fork"]) -> None:
        """Constructor method
        """
        super().__init__(param_defaults=self.defaults,
                         background=background,
                         object=object)

    @property
    def num_conditions(self) -> int:
        """
        Get the number of conditions for the configuration.

        Returns:
            int: The number of conditions.
        """
        return 56