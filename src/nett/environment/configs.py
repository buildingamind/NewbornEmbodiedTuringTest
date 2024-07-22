"""This module contains the NETT configurations for different experiments."""

import sys
import inspect
import csv
from typing import Any
from abc import ABC, abstractmethod
from itertools import product

# from configs.biological_motion_recognition import Biomotion

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

# this won't work I'm 1000% certain
# biomotion = Biomotion(NETTConfig)

class Biomotion(NETTConfig):
    # is the Args block right? Not totally sure...
    """
    NETT configuration for Biological Motion Recognition experiment

    Args:
        imprinting_condition (str | list[str], optional): Motion type. Defaults to ["ChickBiologicalMotion", "InvertedBiologicalMotion"].
        test_condition (str | list[str], optional): Test condition. Defaults to ["Rest", "Inverted", "Random", "Rigid", "Cat", "Scrambled", "Color", "Stationary"].
    """

    def __init__(self,
                    imprinting_condition: str | list[str] = [
                        "ChickBiologicalMotion", 
                        "InvertedBiologicalMotion"
                    ],
                    # test_condition: str | list[str] = [
                    #     "Rest",
                    #     "Inverted",
                    #     "Random",
                    #     "Rigid",
                    #     "Cat",
                    #     "Scrambled",
                    #     "Color",
                    #     "Stationary",
                    #     ""
                    # ],
                    # target_video: str | list[str] = ["biomo.webm", "Inverted.webm"],
                    # nontarget_video: str | list[str] = ["Rest", "Inverted", "Random", "Rigid", "Cat", "Scrambled", "Color", "Stationary",  "White"],
                    # left_monitor: str | list[str] = [],
                    # right_monitor: str | list[str] = []
                ) -> None:
        """Constructor method
        """
        super().__init__(
                    param_defaults = self.defaults,
                    imprinting_condition = imprinting_condition,
                    # test_condition = test_condition,
                    # target_video = target_video,
                    # nontarget_video = nontarget_video,
                    # left_monitor = left_monitor,
                    # right_monitor = right_monitor
                )

    @property
    # TODO: move this row-counting method for DesignSheet_<ExperimentName> into more appropriate, shared placement within environment module 
    def num_conditions(self) -> int:
        """
        Get the number of conditions for the configuration.

        Returns:
            int: The number of conditions.
        """
        # # TODO: pull the name of the .csv programmatically from the user-provided sheet 
        # design_file = csv.reader(open("./configs/biomotion.csv")) 
        # # '4' is hard-coded for now to remove 4 lines from DesignSheet w/o data; programmatic implementation should happen
        # row_count = len(list(design_file)) - 4
        # print(row_count)
        # return row_count
        return 72

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


class Binding(NETTConfig):
    """
    NETT configuration for Binding.

    Args:
        object (str | list[str]): The object(s) to be used. Defaults to ["object1", "object2"].

    Raises:
        ValueError: If any parameter value is not a value or subset of the default values.
    """

    def __init__(self,
                 object: str | list[str] = ["object1", "object2"]) -> None:
        """Constructor method
        """
        super().__init__(param_defaults=self.defaults,
                         object=object)

    @property
    def num_conditions(self) -> int:
        """
        Get the number of conditions for the configuration.

        Returns:
            int: The number of conditions.
        """
        return 50




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
        
        


def list_configs() -> list[str]:
    """
    Lists all available NETT configurations.

    Returns:
        list[str]: A list of configuration names.
    """
    #TODO: Are these really strings?
    is_class_member = lambda member: inspect.isclass(member) and member.__module__ == __name__
    clsmembers = inspect.getmembers(sys.modules[__name__], is_class_member)
    clsmembers = [clsmember[0] for clsmember in clsmembers if clsmember[0] != "NETTConfig"]
    return clsmembers
