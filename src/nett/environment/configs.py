"""This module contains the NETT configurations for different experiments."""

import sys
import inspect
from typing import Any
from abc import ABC, abstractmethod
from itertools import product
from configurations import *

# the naming is confusing since it is used for train or test too.
class NETTConfig(ABC):
    """Abstract base class for NETT configurations.

    Args:
        param_defaults (dict[str, str]): A dictionary of parameter defaults.
        **params: Keyword arguments representing the configuration parameters.

    Raises:
        ValueError: If any parameter value is not a value or subset of the default values.
    """

    def __init__(self, param_defaults: dict[str, str], **params) -> None:
        """Constructor method
        """
        self.param_defaults = param_defaults
        self.params = self._validate_params(params)
        self.conditions = self._create_conditions_from_params(self.params)

    def _create_conditions_from_params(self, params: dict[str, str]) -> set[str]:
        """
        Creates conditions from the configuration parameters.

        Args:
            params (dict[str, str]): The configuration parameters.

        Returns:
            set[str]: A set of conditions.
        """
        combination_params = list(product(*params.values()))
        conditions = {"-".join(combination).lower() for combination in combination_params}
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

    def _validate_params(self, params: dict[str, str]) -> dict[str, str]:
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
    def defaults(self) -> dict[str, Any]:
        """
        Get the default values of the configuration parameters.

        Returns:
            dict[str, Any]: A dictionary of parameter defaults.
        """
        signature = inspect.signature(self.__init__)
        return {param: value.default for param, value in signature.parameters.items()
                if value.default is not inspect.Parameter.empty}

    @property
    @abstractmethod
    def num_conditions(self) -> int:
        """
        Get the number of conditions for the configuration.

        Returns:
            int: The number of conditions.
        """
        pass

    

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
