import sys
import inspect

from abc import ABC, abstractmethod
from itertools import product

# the naming is confusing since it is used for train or test too.
class NETTConfig(ABC):
    def __init__(self, param_defaults: dict[str, str], **params) -> None:
        self.param_defaults = param_defaults
        self.params = self._validate_params(params)
        self.conditions = self._create_conditions_from_params(self.params)

    def _create_conditions_from_params(self, params: dict[str, str]) -> list[str]:
        combination_params = list(product(*params.values()))
        conditions = ['-'.join(combination).lower() for combination in combination_params]
        return conditions

    def _normalize_params(self, params: dict[str, str | int | float]) -> dict[str, str]:
        # add a list if a param has a singleton value (to make sure it works with itertools.product())
        params = {param: (value if isinstance(value, list) else [value]) for param, value in params.items()}
        # typecast values to str
        params = {param: [str(item) for item in value] for param, value in params.items()}
        return params

    def _validate_params(self, params: dict[str, str]):
        params = self._normalize_params(params)
        for (values, default_values) in zip(params.values(), self.param_defaults.values()):
            if not set(values) <= set(default_values):
                raise ValueError(f"{values} should be a value or subset of {default_values}")
        return params

    # don't quite like the round-robin happening here
    # method of the super class is used to pass a value into the super class??
    # move to utils?
    @property
    def defaults(self):
        signature = inspect.signature(self.__init__)
        return {param: value.default for param, value in signature.parameters.items()
                if value.default is not inspect.Parameter.empty}

    @property
    @abstractmethod
    def num_conditions(self):
        pass


class IdentityAndView(NETTConfig):
    """
    NETT description goes here
    """
    def __init__(self,
                 object: str | list[str] = ['object1', 'object2'],
                 rotation: str | list[str] = ['horizontal', 'vertical']) -> None:
        super().__init__(param_defaults=self.defaults,
                         object=object,
                         rotation=rotation)

    @property
    def num_conditions(self):
        return 18


class Binding(NETTConfig):
    """
    NETT description goes here
    """
    def __init__(self,
                 object: str | list[str] = ['object1', 'object2']) -> None:
        super().__init__(param_defaults=self.defaults,
                         object=object)

    @property
    def num_conditions(self):
        return 50

class Parsing(NETTConfig):
    """
    NETT description goes here
    """
    def __init__(self,
                 background: str | list[str] = ['A', 'B', 'C'],
                 object: str | list[str] = ['ship', 'fork']) -> None:
        super().__init__(param_defaults=self.defaults,
                         background=background,
                         object=object)

    @property
    def num_conditions(self):
        return 56

class Slowness(NETTConfig):
    """
    NETT description goes here
    """
    def __init__(self,
                 experiment: str | list[int] = [1, 2],
                 object: str | list[str] = ['obj1', 'obj2'],
                 speed: str | list[str] = ['slow', 'med', 'fast']) -> None:
        super().__init__(param_defaults=self.defaults,
                         experiment=experiment,
                         object=object,
                         speed=speed)

    @property
    def num_conditions(self):
        if self.params['experiment'] == "1":
            return 5
        else:
            return 13

class Smoothness(NETTConfig):
    """
    NETT description goes here
    """
    def __init__(self,
                 object: str | list[str] = ['obj1'],
                 temporal: str | list[str] = ['norm', 'scram']) -> None:
        super().__init__(param_defaults=self.defaults,
                         object=object,
                         temporal=temporal)

    @property
    def num_conditions(self):
        return 5

class OneShotViewInvariant(NETTConfig):
    """
    NETT description goes here
    """
    def __init__(self,
                 object: str | list[str] = ['fork', 'ship'],
                 range: str | list[str] = ['360', 'small', '1'],
                 view: str | list[str] = ['front', 'side']) -> None:
        super().__init__(param_defaults=self.defaults,
                         object=object,
                         range=range,
                         view=view)

    @property
    def num_conditions(self):
        return 50


# get all available NETT configs
def list_configs() -> set[str]:
    is_class_member = lambda member: inspect.isclass(member) and member.__module__ == __name__
    clsmembers = inspect.getmembers(sys.modules[__name__], is_class_member)
    clsmembers = [clsmember[0] for clsmember in clsmembers if clsmember[0] != 'NETTConfig']
    return clsmembers
