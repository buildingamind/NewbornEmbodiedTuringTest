"""NETT-skrl: Isaac Lab + skrl backend for the NETT benchmark."""

import logging
from pathlib import Path

from ._version import __version__

logging.basicConfig(format="[%(name)s] %(levelname)s:  %(message)s", level=logging.INFO)
logger = logging.getLogger("nett")

from .analysis import analyze, merge, test_viz, train_viz
from .body import Body, wrapper_list
from .brain.registry import algorithms_list, encoders_list, rewards_list
from .environment import get_experiment_design
from .nett import NETT


def list_conditions(design_sheet: str | Path) -> list[str]:
    """List imprint conditions present in a NETT design CSV."""
    return list(get_experiment_design(Path(design_sheet)).keys())


def list_wrappers() -> list[str]:
    return wrapper_list


def list_algorithms() -> list[str]:
    return algorithms_list


def list_encoders() -> list[str]:
    return encoders_list


def list_rewards() -> list[str]:
    return rewards_list


__all__ = [
    "NETT",
    "Body",
    "__version__",
    "analyze",
    "list_algorithms",
    "list_conditions",
    "list_encoders",
    "list_rewards",
    "list_wrappers",
    "merge",
    "test_viz",
    "train_viz",
]
