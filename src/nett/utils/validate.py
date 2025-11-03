import json
import jsonschema
from pathlib import Path
import yaml


def validate_config(config: Path | str | dict, schema: dict) -> dict:
    if isinstance(config, dict):
        valid_config = config
    elif isinstance(config, (str, Path)):
        with open(config, "r") as file:
            if config.endswith(".yaml"):
                valid_config = yaml.safe_load(file)
            elif config.endswith(".json"):
                valid_config = json.load(file)
            else:
                raise TypeError("Config should be a yaml or json file.")
        jsonschema.validate(valid_config, schema)
    else:
        raise TypeError("Configs should be type dict, str or Path.")
    return valid_config
