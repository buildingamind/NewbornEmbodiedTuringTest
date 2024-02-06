"""This module contains the NETT configurations for different experiments."""

import sys
import inspect
from abc import ABC, abstractmethod
from itertools import product

# the naming is confusing since it is used for train or test too.
class NETTConfig(ABC):
    """
    Abstract base class for NETT configurations.
    """

    def __init__(self, param_defaults: dict[str, str], **params) -> None:
        """
        Initializes a NETT configuration.

        Args:
            param_defaults (dict[str, str]): A dictionary of parameter defaults.
            **params: Keyword arguments representing the configuration parameters.

        Raises:
            ValueError: If any parameter value is not a value or subset of the default values.
        """
        self.param_defaults = param_defaults
        self.params = self._validate_params(params)
        self.conditions = self._create_conditions_from_params(self.params)

    def _create_conditions_from_params(self, params: dict[str, str]) -> list[str]:
        """
        Creates conditions from the configuration parameters.

        Args:
            params (dict[str, str]): The configuration parameters.

        Returns:
            list[str]: A list of conditions.

        """
        combination_params = list(product(*params.values()))
        conditions = ["-".join(combination).lower() for combination in combination_params]
        return conditions

    def _normalize_params(self, params: dict[str, str | int | float]) -> dict[str, str]:
        """
        Normalizes the configuration parameters.

        Args:
            params (dict[str, str | int | float]): The configuration parameters.

        Returns:
            dict[str, str]: The normalized configuration parameters.

        """
        params = {param: (value if isinstance(value, list) else [value]) for param, value in params.items()}
        params = {param: [str(item) for item in value] for param, value in params.items()}
        return params

    def _validate_params(self, params: dict[str, str]):
        """
        Validates the configuration parameters.

        Args:
            params (dict[str, str]): The configuration parameters.

        Returns:
            dict[str, str]: The validated configuration parameters.

        Raises:
            ValueError: If any parameter value is not a value or subset of the default values.

        """
        params = self._normalize_params(params)
        for (values, default_values) in zip(params.values(), self.param_defaults.values()):
            if not set(values) <= set(default_values):
                raise ValueError(f"{values} should be a value or subset of {default_values}")
        return params

    @property
    def defaults(self):
        """
        Gets the default values of the configuration parameters.

        Returns:
            dict[str, Any]: A dictionary of parameter defaults.

        """
        signature = inspect.signature(self.__init__)
        return {param: value.default for param, value in signature.parameters.items()
                if value.default is not inspect.Parameter.empty}

    @property
    @abstractmethod
    def num_conditions(self):
        """
        Gets the number of conditions for the configuration.

        Returns:
            int: The number of conditions.

        """
        pass


class IdentityAndView(NETTConfig):
    """
    NETT configuration for Identity and View.
    """

    def __init__(self,
                 object: str | list[str] = ["object1", "object2"],
                 rotation: str | list[str] = ["horizontal", "vertical"]) -> None:
        """
        Initializes an Identity and View configuration.

        Args:
            object (str | list[str], optional): The object(s) to be used. Defaults to ["object1", "object2"].
            rotation (str | list[str], optional): The rotation(s) to be used. Defaults to ["horizontal", "vertical"].

        """
        super().__init__(param_defaults=self.defaults,
                         object=object,
                         rotation=rotation)

    @property
    def num_conditions(self):
        """
        Gets the number of conditions for the configuration.

        Returns:
            int: The number of conditions.

        """
        return 18


class Binding(NETTConfig):
    """
    NETT configuration for Binding.
    """

    def __init__(self,
                 object: str | list[str] = ["object1", "object2"]) -> None:
        """
        Initializes a Binding configuration.

        Args:
            object (str | list[str], optional): The object(s) to be used. Defaults to ["object1", "object2"].

        """
        super().__init__(param_defaults=self.defaults,
                         object=object)

    @property
    def num_conditions(self):
        """
        Gets the number of conditions for the configuration.

        Returns:
            int: The number of conditions.

        """
        return 50


class Parsing(NETTConfig):
    """
    NETT configuration for Parsing.
    """

    def __init__(self,
                 background: str | list[str] = ["A", "B", "C"],
                 object: str | list[str] = ["ship", "fork"]) -> None:
        """
        Initializes a Parsing configuration.

        Args:
            background (str | list[str], optional): The background(s) to be used. Defaults to ["A", "B", "C"].
            object (str | list[str], optional): The object(s) to be used. Defaults to ["ship", "fork"].

        """
        super().__init__(param_defaults=self.defaults,
                         background=background,
                         object=object)

    @property
    def num_conditions(self):
        """
        Gets the number of conditions for the configuration.

        Returns:
            int: The number of conditions.

        """
        return 56


class Slowness(NETTConfig):
    """
    NETT configuration for Slowness.
    """

    def __init__(self,
                 experiment: str | list[int] = [1, 2],
                 object: str | list[str] = ["obj1", "obj2"],
                 speed: str | list[str] = ["slow", "med", "fast"]) -> None:
        """
        Initializes a Slowness configuration.

        Args:
            experiment (str | list[int], optional): The experiment(s) to be used. Defaults to [1, 2].
            object (str | list[str], optional): The object(s) to be used. Defaults to ["obj1", "obj2"].
            speed (str | list[str], optional): The speed(s) to be used. Defaults to ["slow", "med", "fast"].

        """
        super().__init__(param_defaults=self.defaults,
                         experiment=experiment,
                         object=object,
                         speed=speed)

    @property
    def num_conditions(self):
        """
        Gets the number of conditions for the configuration.

        Returns:
            int: The number of conditions.

        """
        if self.params["experiment"] == "1":
            return 5
        return 13


class Smoothness(NETTConfig):
    """
    NETT configuration for Smoothness.
    """

    def __init__(self,
                 object: str | list[str] = ["obj1"],
                 temporal: str | list[str] = ["norm", "scram"]) -> None:
        """
        Initializes a Smoothness configuration.

        Args:
            object (str | list[str], optional): The object(s) to be used. Defaults to ["obj1"].
            temporal (str | list[str], optional): The temporal condition(s) to be used. Defaults to ["norm", "scram"].

        """
        super().__init__(param_defaults=self.defaults,
                         object=object,
                         temporal=temporal)

    @property
    def num_conditions(self):
        """
        Gets the number of conditions for the configuration.

        Returns:
            int: The number of conditions.

        """
        return 5


class OneShotViewInvariant(NETTConfig):
    """
    NETT configuration for One-Shot View Invariant.
    """

    def __init__(self,
                 object: str | list[str] = ["fork", "ship"],
                 range: str | list[str] = ["360", "small", "1"],
                 view: str | list[str] = ["front", "side"]) -> None:
        """
        Initializes a One-Shot View Invariant configuration.

        Args:
            object (str | list[str], optional): The object(s) to be used. Defaults to ["fork", "ship"].
            range (str | list[str], optional): The range(s) to be used. Defaults to ["360", "small", "1"].
            view (str | list[str], optional): The view(s) to be used. Defaults to ["front", "side"].

        """
        super().__init__(param_defaults=self.defaults,
                         object=object,
                         range=range,
                         view=view)

    @property
    def num_conditions(self):
        """
        Gets the number of conditions for the configuration.

        Returns:
            int: The number of conditions.

        """
        return 50


def list_configs() -> set[str]:
    """
    Lists all available NETT configurations.

    Returns:
        set[str]: A set of configuration names.

    """
    is_class_member = lambda member: inspect.isclass(member) and member.__module__ == __name__
    clsmembers = inspect.getmembers(sys.modules[__name__], is_class_member)
    clsmembers = [clsmember[0] for clsmember in clsmembers if clsmember[0] != "NETTConfig"]
    return clsmembers
