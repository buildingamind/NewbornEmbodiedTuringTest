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
    return valid_config
