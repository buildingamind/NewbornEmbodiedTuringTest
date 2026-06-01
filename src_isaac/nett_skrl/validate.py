import json
import jsonschema
from pathlib import Path
import yaml


def validate_config(config: Path | str | dict, schema: dict) -> dict:
    if isinstance(config, dict):
        valid_config = config
        jsonschema.validate(valid_config, schema)
    elif isinstance(config, (str, Path)):
        config_str = str(config)
        with open(config, "r") as file:
            if config_str.endswith(".yaml") or config_str.endswith(".yml"):
                valid_config = yaml.safe_load(file)
            elif config_str.endswith(".json"):
                valid_config = json.load(file)
            else:
                raise TypeError("Config should be a yaml or json file.")
        jsonschema.validate(valid_config, schema)
    else:
        raise TypeError("Configs should be type dict, str or Path.")
    _validate_registered_names(valid_config)
    return valid_config


_VALID_REWARD_TYPES = {"closeness", "completeness"}


def _validate_registered_names(config: dict) -> None:
    """Validate dynamic registry-backed names after static JSON schema checks."""
    brain = config.get("brain") or {}
    if "encoder" in brain:
        from nett_skrl.brain.registry import validate_encoder

        validate_encoder(brain["encoder"])
    if "algorithm" in brain:
        from nett_skrl.brain.registry import validate_algorithm

        validate_algorithm(brain["algorithm"])
    if "reward" in brain:
        from nett_skrl.brain.registry import validate_reward

        validate_reward(brain["reward"])

    body = config.get("body") or {}
    wrappers = body.get("wrappers") or []
    if wrappers:
        from nett_skrl.body.wrappers.registry import validate_wrappers

        validate_wrappers(wrappers)

    environment = config.get("environment") or {}
    reward_types = environment.get("reward_types") or []
    for name in reward_types:
        if not isinstance(name, str) or name not in _VALID_REWARD_TYPES:
            raise ValueError(
                f"environment.reward_types entries must be one of "
                f"{sorted(_VALID_REWARD_TYPES)}; got {name!r}."
            )
