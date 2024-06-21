from environment.configs import NETTConfig

class ViewInvariant(NETTConfig):
    """
    NETT configuration for Binding.

    Args:
        object (str | list[str]): The object(s) to be used. Defaults to ["object1", "object2"].

    Raises:
        ValueError: If any parameter value is not a value or subset of the default values.
    """

    def __init__(self,
                 object: str | list[str] = ["ship", "fork"],
                 view: str | list[str] = ["front", "side"]) -> None:
        """Constructor method
        """
        super().__init__(param_defaults=self.defaults,
                         object=object, view = view)

    @property
    def num_conditions(self) -> int:
        """
        Get the number of conditions for the configuration.

        Returns:
            int: The number of conditions.
        """
        if self.view.lower()=="front":
            return 50
        return 26
        