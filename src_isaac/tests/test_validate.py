"""Tests for the schema + registry validators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import nett_skrl
from nett_skrl import Body
from nett_skrl.brain.registry import (
    algorithms_list,
    register_encoder,
    register_reward,
    validate_algorithm,
    validate_encoder,
    validate_reward,
)
from nett_skrl.brain.encoders import NETTFeatureExtractor
from nett_skrl.body.wrappers.registry import validate_wrappers
from nett_skrl.utils.validate import validate_config


_SCHEMA = json.loads((Path(nett_skrl.__file__).parent / "schema.json").read_text())


def _minimal_cfg() -> dict:
    return {
        "name": "t",
        "environment": {
            "design_sheet": "/tmp/x.csv",
            "media_root": "/tmp/v",
        },
    }


def test_schema_accepts_minimal():
    assert validate_config(_minimal_cfg(), _SCHEMA) == _minimal_cfg()


def test_schema_rejects_missing_required():
    cfg = {"environment": {}}
    with pytest.raises(Exception):
        validate_config(cfg, _SCHEMA)


def test_schema_rejects_unknown_algorithm():
    cfg = _minimal_cfg()
    cfg["brain"] = {"algorithm": "NopeQ"}
    with pytest.raises(Exception):
        validate_config(cfg, _SCHEMA)


def test_schema_rejects_dqn_for_continuous_nett_actions():
    cfg = _minimal_cfg()
    cfg["brain"] = {"algorithm": "DQN"}
    with pytest.raises(Exception):
        validate_config(cfg, _SCHEMA)


@pytest.mark.parametrize("algorithm", ["DDQN", "Q_LEARNING", "SARSA", "AMP"])
def test_schema_rejects_non_nett_continuous_algorithms(algorithm):
    cfg = _minimal_cfg()
    cfg["brain"] = {"algorithm": algorithm}
    with pytest.raises(Exception):
        validate_config(cfg, _SCHEMA)


def test_schema_rejects_recurrent_ppo_until_recurrent_models_exist():
    cfg = _minimal_cfg()
    cfg["brain"] = {"algorithm": "PPO_RNN"}
    with pytest.raises(Exception):
        validate_config(cfg, _SCHEMA)


@pytest.mark.parametrize(
    "policy",
    ["Gaussian", "Deterministic", "Categorical", "CnnPolicy", "MlpPolicy", "MultiInputPolicy"],
)
def test_schema_rejects_policy_config(policy):
    cfg = _minimal_cfg()
    cfg["brain"] = {"policy": policy}
    with pytest.raises(Exception):
        validate_config(cfg, _SCHEMA)


def test_schema_accepts_recording_and_record_mode():
    cfg = _minimal_cfg()
    cfg["environment"].update(
        {
            "record_mode": "spatial",
            "random_first_frame": True,
            "switch_steps": 50,
            "decision_period": 5,
            "enable_neck_flexion": True,
            "enable_lateral_bending": True,
            "recording": {
                "fps": 30,
                "egocentric": {"train": [0, 2], "record": [0, 2]},
                "chamber": {"test": "1:"},
            },
        }
    )
    cfg["episodes"] = {"train": 1, "record": 1}
    cfg["brain"] = {
        "intrinsic_reward_weight": 0.5,
        "train_intrinsic_reward": False,
        "model": {"hidden_sizes": [32, 16], "activation": "relu", "initial_log_std": -0.5},
    }
    cfg["wrappers"] = ["video", "dvs", "retina"]
    cfg["eval_freq"] = 50000
    validate_config(cfg, _SCHEMA)


def test_schema_accepts_body_block():
    cfg = _minimal_cfg()
    cfg["body"] = {
        "wrappers": ["video"],
        "binocular_vision": False,
        "input_resolution": 32,
    }
    validate_config(cfg, _SCHEMA)


def test_schema_rejects_duplicate_wrapper_locations():
    cfg = _minimal_cfg()
    cfg["wrappers"] = ["video"]
    cfg["body"] = {"wrappers": ["dvs"]}
    with pytest.raises(ValueError):
        validate_config(cfg, _SCHEMA)


def test_schema_rejects_eval_strategy():
    cfg = _minimal_cfg()
    cfg["environment"]["eval_strategy"] = "in_process"
    with pytest.raises(Exception):
        validate_config(cfg, _SCHEMA)


@pytest.mark.parametrize("name", ["DinoV1", "DinoV2", "CNNLSTM", "ViT", "SegmentAnything"])
def test_schema_rejects_unregistered_heavy_encoder_names(name):
    cfg = _minimal_cfg()
    cfg["brain"] = {"encoder": name}
    with pytest.raises(Exception):
        validate_config(cfg, _SCHEMA)


def test_validate_algorithm_string_resolves_to_skrl_class():
    cls = validate_algorithm("PPO")
    assert cls.__name__ == "PPO"


def test_validate_algorithm_rejects_unknown():
    with pytest.raises(KeyError):
        validate_algorithm("NopeQ")


def test_validate_algorithm_rejects_dqn():
    with pytest.raises(KeyError):
        validate_algorithm("DQN")


def test_register_encoder_adds_to_registry():
    class Dummy(NETTFeatureExtractor):
        def forward(self, observations):
            return observations

    register_encoder("DummyEnc", Dummy)
    assert validate_encoder("DummyEnc") is Dummy
    cfg = _minimal_cfg()
    cfg["brain"] = {"encoder": "DummyEnc"}
    validate_config(cfg, _SCHEMA)


def test_unregistered_heavy_encoder_name_fails_with_registration_guidance():
    with pytest.raises(KeyError, match="register_encoder"):
        validate_encoder("DinoV1")


def test_register_encoder_rejects_non_nett_feature_extractor():
    class Dummy:
        pass

    with pytest.raises(TypeError, match="NETTFeatureExtractor"):
        register_encoder("BadDummyEnc", Dummy)


def test_register_reward_adds_to_registry():
    class DummyReward:
        pass
    register_reward("DummyReward", DummyReward)
    assert validate_reward("DummyReward") is DummyReward


def test_validate_reward_none_returns_none():
    # Env-side rewards like "closeness" intentionally map to None.
    assert validate_reward("closeness") is None


@pytest.mark.parametrize("name", ["ICM", "E3B", "RIDE", "PseudoCounts", "NGU"])
def test_validate_builtin_intrinsic_rewards(name):
    cls = validate_reward(name)
    assert cls is not None
    assert cls.__name__ in {"ICM", "E3B", "RIDE", "PseudoCounts"}


@pytest.mark.parametrize("name", ["small", "medium", "large", "Resnet10CNN", "Resnet18CNN"])
def test_validate_builtin_native_encoders(name):
    cls = validate_encoder(name)
    assert cls is not None


def test_list_algorithms_matches_continuous_skrl_surface():
    assert set(nett_skrl.list_algorithms()) == {
        "A2C",
        "CEM",
        "DDPG",
        "PPO",
        "RPO",
        "SAC",
        "TD3",
        "TRPO",
    }
    assert nett_skrl.list_algorithms() == algorithms_list


def test_model_type_listing_removed_from_public_api():
    assert not hasattr(nett_skrl, "list_model_types")


@pytest.mark.parametrize("name", ["Disagreement", "Fabric", "RE3", "RND"])
def test_unsupported_intrinsic_rewards_fail_explicitly(name):
    cls = validate_reward(name)
    with pytest.raises(ImportError, match=name):
        cls()


def test_public_wrapper_names_validate_lazily():
    wrappers = validate_wrappers(["video", "dvs", "retina"])
    assert [wrapper.__name__ for wrapper in wrappers] == ["Video", "DVS", "Retina"]


def test_body_is_public_and_validates_wrappers():
    body = Body(wrappers=["video"], binocular_vision=False, input_resolution=32)
    assert [wrapper.__name__ for wrapper in body.wrappers] == ["Video"]
    assert body.binocular_vision is False
    assert body.input_resolution == 32


@pytest.mark.parametrize("name", ["binocular", "multiobs"])
def test_removed_wrapper_names_fail(name):
    with pytest.raises(KeyError):
        validate_wrappers([name])
