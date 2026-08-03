import json
import jsonschema
from pathlib import Path
import yaml


#: jsonschema's ``"array"`` type accepts ``list`` and NOT ``tuple``, but a dict config
#: is written in Python where a fixed-size pair is naturally a tuple -- ``(256, 160)``
#: for a resolution is the obvious spelling, and ``_ENV_CFG_FIELDS`` even declares
#: ``tuple`` as its transform. Left alone, jsonschema rejects it with "(128, 80) is not
#: of type 'array', 'null'", which reads like the value is malformed rather than merely
#: the wrong sequence type, and it fails at config-load time -- before a single step,
#: after the GPUs are already committed.
_ARRAY_TYPES = (list, tuple)


def _accepts_tuples(schema: dict) -> jsonschema.protocols.Validator:
    """A validator whose ``"array"`` also means ``tuple``.

    Only the type CHECKER is widened; every keyword (``minItems``, ``items``, ...)
    still applies, so a tuple is validated exactly as strictly as the list it
    stands in for.
    """
    base = jsonschema.validators.validator_for(schema)
    checker = base.TYPE_CHECKER.redefine(
        "array", lambda _checker, value: isinstance(value, _ARRAY_TYPES)
    )
    return jsonschema.validators.extend(base, type_checker=checker)(schema)


def validate_config(config: Path | str | dict, schema: dict) -> dict:
    if isinstance(config, dict):
        valid_config = config
        # Dict configs come from Python and may carry tuples; file configs are
        # parsed from JSON/YAML and cannot, so they take the stock validator.
        _accepts_tuples(schema).validate(valid_config)
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
