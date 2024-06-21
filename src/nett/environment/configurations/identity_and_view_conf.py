from environment.configs import NETTConfig

class IdentityAndView(NETTConfig):
    """
    NETT configuration for Identity and View.

    Args:
        object (str | list[str]): The object(s) to be used. Defaults to ["object1", "object2"].
        rotation (str | list[str]): The rotation(s) to be used. Defaults to ["horizontal", "vertical"].

    Raises:
        ValueError: If any parameter value is not a value or subset of the default values.
    """

    def __init__(self,
                 object: str | list[str] = ["object1", "object2"],
                 rotation: str | list[str] = ["horizontal", "vertical"]) -> None:
        """Constructor method
        """
        super().__init__(param_defaults=self.defaults,
                         object=object,
                         rotation=rotation)

    @property
    def num_conditions(self) -> int:
        """
        Get the number of conditions for the configuration.

        Returns:
            int: The number of conditions.
        """
        return 18
