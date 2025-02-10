import jsonschema
from pathlib import Path
import yaml

def validate_config(config: Path | str | dict, schema: dict) -> dict:
    if isinstance(config, dict):
        valid_config = config
    if isinstance(config, (str, Path)):
        with open(config, "r") as file:
            valid_config = yaml.safe_load(file)
    else:
        raise TypeError("Configs should be type dict, str or Path.")
    jsonschema.validate(valid_config, schema)
    return valid_config