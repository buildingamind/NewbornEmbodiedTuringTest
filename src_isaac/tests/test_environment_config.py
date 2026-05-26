"""Tests for NETT-skrl to NETTEnvCfg wiring."""

from __future__ import annotations

from pathlib import Path

from nett_skrl.environment.environment import Environment
from nett_skrl.environment.environment import parse_episode_selector
from nett_skrl.runtime.task import TaskConfig


def _design_sheet(tmp_path: Path) -> Path:
    csv = tmp_path / "design.csv"
    csv.write_text(
        "ImprintCondition,Phase,TestCondition,TargetVideo,NonTargetVideo,LeftMonitor,RightMonitor\n"
        "Object1,Train,,a.mp4,b.mp4,a.mp4,b.mp4\n"
        "Object1,Test,t1,a.mp4,b.mp4,a.mp4,b.mp4\n"
    )
    return csv


class _Nested:
    pass


class _Cfg:
    def __init__(self):
        self.scene = _Nested()
        self.observation = _Nested()
        self.screens = _Nested()
        self.asset_root = None
        self.scene.num_envs = 1
        self.observation.binocular = True
        self.observation.input_resolution = 64
        self.screens.random_first_frame = False
        self.screens.switch_steps = 0
        self.screens.decision_period = 1
        self.seed = 0
        self.post_init_calls = 0

    def __post_init__(self):
        self.post_init_calls += 1
        width = self.observation.input_resolution * (2 if self.observation.binocular else 1)
        self.derived_shape = (self.observation.input_resolution, width, 3)


class _TaskConfig:
    current_mode = "train"
    condition = "Object1"
    episodes = {"train": 3, "test": 1, "record": 1}
    seed = 123
    dry_run = False
    eval_metrics_only = False
    path: Path


def test_configure_cfg_refreshes_derived_fields_and_seed(tmp_path):
    media_root = tmp_path / "media"
    media_root.mkdir()
    env = Environment(
        design_sheet=_design_sheet(tmp_path),
        media_root=media_root,
        binocular_vision=False,
        input_resolution=32,
        episode_steps=77,
        reward_types=("closeness",),
        record_mode="spatial",
        random_first_frame=True,
        switch_steps=9,
        decision_period=3,
        recording={
            "egocentric": {"train": [0, 2]},
            "chamber": {"train": "1:"},
            "fps": 12,
        },
    )
    env.adjust_to_agent(num_brains=4)
    task = _TaskConfig()
    task.path = tmp_path / "run"
    cfg = _Cfg()

    env._configure_cfg(cfg, task, seed=999)

    assert cfg.scene.num_envs == 4
    assert cfg.phase == "train"
    assert cfg.imprint_condition == "Object1"
    assert cfg.episode_steps == 77
    assert cfg.observation.binocular is False
    assert cfg.observation.input_resolution == 32
    assert cfg.reward_types == ("closeness",)
    assert cfg.record_mode == "spatial"
    assert cfg.screens.random_first_frame is True
    assert cfg.screens.switch_steps == 9
    assert cfg.screens.decision_period == 3
    assert cfg.record_path.endswith("recordings/egocentric/train")
    assert cfg.chamber_record_path.endswith("recordings/chamber/train")
    assert cfg.record_episodes == (0, 2)
    assert cfg.chamber_record_episodes == (1, 2)
    assert cfg.seed == 999
    assert cfg.post_init_calls == 1
    assert cfg.derived_shape == (32, 32, 3)


def test_configure_cfg_infers_private_asset_root(tmp_path):
    assets = tmp_path / "assets"
    for rel in ("chick/chick.usd", "chamber/chamber.usd"):
        path = assets / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    design_dir = assets / "design_sheets"
    design_dir.mkdir(parents=True, exist_ok=True)
    design_sheet = _design_sheet(design_dir)
    media_root = assets / "videos"
    media_root.mkdir()

    env = Environment(design_sheet=design_sheet, media_root=media_root)
    task = _TaskConfig()
    task.path = tmp_path / "run"
    cfg = _Cfg()

    env._configure_cfg(cfg, task)

    assert cfg.asset_root == str(assets.resolve())


def test_recording_episode_empty_selector_disables_path(tmp_path):
    media_root = tmp_path / "media"
    media_root.mkdir()
    env = Environment(
        design_sheet=_design_sheet(tmp_path),
        media_root=media_root,
        recording={"egocentric": {"train": "0:0:1"}},
    )
    task = _TaskConfig()
    task.path = tmp_path / "run"
    cfg = _Cfg()

    env._configure_cfg(cfg, task)

    assert not hasattr(cfg, "record_path")


def test_metrics_only_eval_disables_recording_paths(tmp_path):
    media_root = tmp_path / "media"
    media_root.mkdir()
    env = Environment(
        design_sheet=_design_sheet(tmp_path),
        media_root=media_root,
        recording={
            "egocentric": {"test": [0]},
            "chamber": {"test": [0]},
        },
    )
    task = _TaskConfig()
    task.current_mode = "test"
    task.eval_metrics_only = True
    task.path = tmp_path / "run"
    cfg = _Cfg()

    env._configure_cfg(cfg, task)

    assert not hasattr(cfg, "record_path")
    assert not hasattr(cfg, "chamber_record_path")


def test_parse_episode_selector_supports_lists_slices_and_all():
    assert parse_episode_selector(None, 3) == (0, 1, 2)
    assert parse_episode_selector([2, 0, 2], 3) == (2, 0)
    assert parse_episode_selector("1:", 4) == (1, 2, 3)
    assert parse_episode_selector("0:4:2", 5) == (0, 2)
    assert parse_episode_selector(9, 2) == ()


def test_task_config_seed_is_stable(tmp_path):
    a = TaskConfig("Object1", tmp_path / "run", ["train"])
    b = TaskConfig("Object1", tmp_path / "run", ["train"])
    c = TaskConfig("Object2", tmp_path / "run", ["train"])
    assert a.seed == b.seed
    assert a.seed != c.seed


def test_task_config_mode_view_does_not_mutate_base(tmp_path):
    config = TaskConfig("Object1", tmp_path / "run", ["train", "test"])
    train = config.for_mode("train")
    test = config.for_mode("test")
    assert config.current_mode is None
    assert train.current_mode == "train"
    assert test.current_mode == "test"
    assert train.condition == test.condition == "Object1"
