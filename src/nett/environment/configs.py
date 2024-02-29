"""This module contains the NETT configurations for different experiments."""

import sys
import inspect
from abc import ABC, abstractmethod
from itertools import product

# the naming is confusing since it is used for train or test too.
class NETTConfig(ABC):
    """Abstract base class for NETT configurations.

    :param param_defaults: A dictionary of parameter defaults.
    :type param_defaults: dict[str, str]
    :param **params: Keyword arguments representing the configuration parameters.
    :raises ValueError: If any parameter value is not a value or subset of the default values.
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

        :param params: The configuration parameters.
        :type params: dict[str, str]
        :return: A set of conditions.
        :rtype: set[str]
        """
        combination_params = list(product(*params.values()))
        conditions = {"-".join(combination).lower() for combination in combination_params}
        return conditions

    def _normalize_params(self, params: dict[str, str | int | float]) -> dict[str, str]:
        """
        Normalizes the configuration parameters.

        :param params: The configuration parameters.
        :type params: dict[str, str | int | float]
        :return: The normalized configuration parameters.
        :rtype: dict[str, str]

        """
        params = {param: (value if isinstance(value, list) else [value]) for param, value in params.items()}
        params = {param: [str(item) for item in value] for param, value in params.items()}
        return params

    def _validate_params(self, params: dict[str, str]):
        """
        Validates the configuration parameters.

        :param params: The configuration parameters.
        :type params: dict[str, str]
        :return: The validated configuration parameters.
        :rtype: dict[str, str]
        :raises ValueError: If any parameter value is not a value or subset of the default values.
        """
        params = self._normalize_params(params)
        for (values, default_values) in zip(params.values(), self.param_defaults.values()):
            if not set(values) <= set(default_values):
                raise ValueError(f"{values} should be a value or subset of {default_values}")
        return params

    @property
    def defaults(self):
        """
        Get the default values of the configuration parameters.

        :return: A dictionary of parameter defaults.
        :rtype: dict[str, Any]
        """
        signature = inspect.signature(self.__init__)
        return {param: value.default for param, value in signature.parameters.items()
                if value.default is not inspect.Parameter.empty}

    @property
    @abstractmethod
    def num_conditions(self):
        """
        Get the number of conditions for the configuration.

        :return: The number of conditions.
        :rtype: int
        """
        pass


class IdentityAndView(NETTConfig):
    """
    NETT configuration for Identity and View.

    :param object: The object(s) to be used. Defaults to ["object1", "object2"].
    :type object: str or list[str]
    :param rotation: The rotation(s) to be used. Defaults to ["horizontal", "vertical"].
    :type rotation: str or list[str]
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
    def num_conditions(self):
        """
        Get the number of conditions for the configuration.

        :return: The number of conditions.
        :rtype: int
        """
        return 18


class Binding(NETTConfig):
    """
    NETT configuration for Binding.

    :param object: The object(s) to be used. Defaults to ["object1", "object2"].
    :type object: str or list[str]
    """

    def __init__(self,
                 object: str | list[str] = ["object1", "object2"]) -> None:
        """Constructor method
        """
        super().__init__(param_defaults=self.defaults,
                         object=object)

    @property
    def num_conditions(self):
        """
        Get the number of conditions for the configuration.

        :return: The number of conditions.
        :rtype: int
        """
        return 50


class Parsing(NETTConfig):
    """
    NETT configuration for Parsing.

    :param background: The background(s) to be used. Defaults to ["A", "B", "C"].
    :type background: str or list[str]
    :param object: The object(s) to be used. Defaults to ["ship", "fork"].
    :type object: str or list[str]
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
    def num_conditions(self):
        """
        Get the number of conditions for the configuration.

        :return: The number of conditions.
        :rtype: int
        """
        return 56


class Slowness(NETTConfig):
    """
    NETT configuration for Slowness.

    :param experiment: The experiment(s) to be used. Defaults to [1, 2].
    :type experiment: str or list[int]
    :param object: The object(s) to be used. Defaults to ["obj1", "obj2"].
    :type object: str or list[str]
    :param speed: The speed(s) to be used. Defaults to ["slow", "med", "fast"].
    :type speed: str or list[str]
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
    def num_conditions(self):
        """
        Get the number of conditions for the configuration.

        :return: The number of conditions.
        :rtype: int
        """
        if self.params["experiment"] == "1":
            return 5
        return 13


class Smoothness(NETTConfig):
    """
    NETT configuration for Smoothness.

    :param object: The object(s) to be used. Defaults to ["obj1"].
    :type object: str or list[str]
    :param temporal: The temporal condition(s) to be used. Defaults to ["norm", "scram"].
    :type temporal: str or list[str]
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
    def num_conditions(self):
        """
        Get the number of conditions for the configuration.

        :return: The number of conditions.
        :rtype: int
        """
        return 5


class OneShotViewInvariant(NETTConfig):
    """
    NETT configuration for One-Shot View Invariant.

    :param object: The object(s) to be used. Defaults to ["fork", "ship"].
    :type object: str or list[str]
    :param range: The range(s) to be used. Defaults to ["360", "small", "1"].
    :type range: str or list[str]
    :param view: The view(s) to be used. Defaults to ["front", "side"].
    :type view: str or list[str]
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
    def num_conditions(self):
        """
        Get the number of conditions for the configuration.

        :return: The number of conditions.
        :rtype: int
        """
        return 50


def list_configs() -> set[str]:
    """
    Lists all available NETT configurations.

    :return: A set of configuration names.
    :rtype: set[str]

    """
    is_class_member = lambda member: inspect.isclass(member) and member.__module__ == __name__
    clsmembers = inspect.getmembers(sys.modules[__name__], is_class_member)
    clsmembers = [clsmember[0] for clsmember in clsmembers if clsmember[0] != "NETTConfig"]
    return clsmembers
