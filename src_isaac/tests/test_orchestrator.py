"""Unit tests for BrainTrainer's per-env-scope dispatch.

Mocks both the env and the skrl agents so this runs without Isaac Sim and
without skrl's optimizer step. The point is to lock the slicing contract:
N agents each see only their contiguous env scope for act/record_transition.
"""

from __future__ import annotations

import json
import csv
import logging
from concurrent.futures import Future
from types import SimpleNamespace

import torch

import nett_skrl.nett as nett_module
from nett_skrl.recording import RunRecorder
from nett_skrl.brain.trainer import BrainTrainer, TrainCfg
from nett_skrl.recording import RecordingCfg
from nett_skrl.nett import NETT
from nett_skrl.runtime.parallel_envs import capped_num_envs, num_env_candidates
from nett_skrl.runtime.task_runner import (
    _is_tolerated_isaac_teardown_exit,
    _training_boundaries,
    _write_eval_metrics,
)
from nett_skrl.runtime.task import TaskConfig


class _FakeEnv:
    """Minimal skrl-wrapper-compatible env yielding random tensors."""

    def __init__(self, num_envs: int, obs_dim: int = 4, act_dim: int = 2,
                 device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        self._t = 0
        self.num_agents = 1

    def _obs(self) -> torch.Tensor:
        return torch.randn(self.num_envs, self.obs_dim, device=self.device)

    def reset(self):
        return self._obs(), {}

    def step(self, actions):
        # Verify action stack shape matches (num_envs, act_dim).
        assert actions.shape == (self.num_envs, self.act_dim), actions.shape
        self._t += 1
        return (
            self._obs(),
            torch.randn(self.num_envs, 1, device=self.device),
            torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            {},
        )

    def state(self):
        return None

    def render(self, *args, **kwargs):
        return None

    def close(self):
        pass


class _FakeAgent:
    """Records every call so the test can assert per-slice indexing."""

    def __init__(self, act_dim: int = 2) -> None:
        self.act_dim = act_dim
        self.act_calls: list = []
        self.record_calls: list = []
        self.post_calls = 0
        self.pre_calls = 0
        self.init_calls = 0
        self.mode = None
        self.saved: list[str] = []

    def init(self, trainer_cfg=None):
        self.init_calls += 1

    def enable_training_mode(self, enabled: bool, *, apply_to_models: bool = False) -> None:
        self.mode = "train" if enabled else "eval"

    def pre_interaction(self, *, timestep, timesteps):
        self.pre_calls += 1

    def act(self, observations, states, *, timestep, timesteps):
        self.act_calls.append(observations.clone())
        # Stub action: zeros, same batch dim as obs.
        return torch.zeros(observations.shape[0], self.act_dim), {}

    def record_transition(self, **kwargs):
        self.record_calls.append({k: v for k, v in kwargs.items() if k != "infos"})

    def post_interaction(self, *, timestep, timesteps):
        self.post_calls += 1

    def track_data(self, *args, **kwargs):
        pass

    def save(self, path: str):
        self.saved.append(path)


def test_trainer_dispatches_each_slice_to_its_agent():
    n = 3
    env = _FakeEnv(num_envs=n)
    agents = [_FakeAgent() for _ in range(n)]
    trainer = BrainTrainer(env, agents, device="cpu")
    trainer.train(TrainCfg(total_timesteps=4))

    # Each agent saw the same number of act + record_transition calls as steps.
    for i, agent in enumerate(agents):
        assert len(agent.act_calls) == 4, f"agent {i}: {len(agent.act_calls)}"
        assert len(agent.record_calls) == 4
        # Each call's observation batch should be (1, obs_dim) — single slice.
        for obs in agent.act_calls:
            assert obs.shape == (1, env.obs_dim)
        assert agent.post_calls == 4


def test_trainer_dispatches_parallel_env_scope_to_single_agent():
    env = _FakeEnv(num_envs=2)
    agent = _FakeAgent()
    trainer = BrainTrainer(env, [agent], device="cpu")
    trainer.train(TrainCfg(total_timesteps=4))

    assert len(agent.act_calls) == 4
    assert len(agent.record_calls) == 4
    for obs in agent.act_calls:
        assert obs.shape == (2, env.obs_dim)


def test_trainer_dispatches_parallel_env_scopes_to_multiple_agents():
    env = _FakeEnv(num_envs=4)
    agents = [_FakeAgent() for _ in range(2)]
    trainer = BrainTrainer(env, agents, device="cpu")
    trainer.train(TrainCfg(total_timesteps=4))

    assert trainer.scopes == [2, 2]
    for agent in agents:
        assert len(agent.act_calls) == 4
        for obs in agent.act_calls:
            assert obs.shape == (2, env.obs_dim)


def test_trainer_eval_switches_mode_to_eval():
    env = _FakeEnv(num_envs=2)
    agents = [_FakeAgent() for _ in range(2)]
    trainer = BrainTrainer(env, agents, device="cpu")
    means = trainer.eval(total_timesteps=3)
    assert set(means.keys()) == {0, 1}
    assert all(a.mode == "eval" for a in agents)


def test_trainer_eval_does_not_finish_wandb_runs():
    class _Run:
        def __init__(self) -> None:
            self.finished = 0

        def finish(self):
            self.finished += 1

    env = _FakeEnv(num_envs=1)
    agent = _FakeAgent()
    run = _Run()
    agent._nett_wandb_run = run

    BrainTrainer(env, [agent], device="cpu").eval(total_timesteps=1)

    assert run.finished == 0
    assert agent._nett_wandb_run is run


def test_trainer_eval_zero_steps_returns_zero_without_reset():
    env = _FakeEnv(num_envs=2)
    agents = [_FakeAgent() for _ in range(2)]
    trainer = BrainTrainer(env, agents, device="cpu")
    assert trainer.eval(total_timesteps=0) == {0: 0.0, 1: 0.0}


def test_trainer_raises_on_env_agent_mismatch():
    env = _FakeEnv(num_envs=3)
    agents = [_FakeAgent() for _ in range(2)]
    try:
        BrainTrainer(env, agents, device="cpu")
    except ValueError as e:
        assert "num_envs" in str(e)
    else:
        raise AssertionError("expected ValueError on size mismatch")


def test_trainer_saves_final_checkpoint_per_agent_experiment_dir(tmp_path):
    """``BrainTrainer.train`` must write ``final_agent.pt`` into each
    agent's own ``experiment_dir`` (the same dir skrl writes ``agent_{N}.pt``
    + ``best_agent.pt`` to). NETT's old ``brain_{i}/models/`` layout is gone;
    checkpoints are owned end-to-end by skrl now."""
    env = _FakeEnv(num_envs=2)
    agents = [_FakeAgent() for _ in range(2)]
    # The trainer reads ``agent.experiment_dir`` to locate the checkpoint dir.
    # In real runs this is populated by ``Agent.init()`` from
    # ``cfg.experiment.directory + experiment_name``; here we set it directly.
    agents[0].experiment_dir = str(tmp_path / "wandb_runs" / "brain_1")
    agents[1].experiment_dir = str(tmp_path / "wandb_runs" / "brain_2")
    trainer = BrainTrainer(env, agents, device="cpu")
    trainer.train(TrainCfg(total_timesteps=2, hparams_dir=tmp_path))
    assert agents[0].saved[-1].endswith("wandb_runs/brain_1/checkpoints/final_agent.pt")
    assert agents[1].saved[-1].endswith("wandb_runs/brain_2/checkpoints/final_agent.pt")
    # Hparams JSON backup still written (skrl auto-captures hparams to wandb,
    # but the local backup keeps offline tooling working).
    assert (tmp_path / "logs" / "hparams.json").exists()
    timing = json.loads((tmp_path / "logs" / "train_timing.json").read_text())
    assert timing["env_timesteps"] == 2
    assert timing["train_steps"] == 4
    assert timing["train_steps_per_second"] >= timing["env_steps_per_second"]


def test_run_recorder_skips_outputs_during_dry_run(tmp_path):
    agent = _FakeAgent()
    agent.experiment_dir = str(tmp_path / "wandb_runs" / "brain_1")
    recorder = RunRecorder([agent], num_envs=1)

    cfg = TrainCfg(total_timesteps=2, hparams_dir=tmp_path)
    recorder.before_train(cfg, dry_run=True)
    recorder.after_train(cfg, elapsed_s=1.0, dry_run=True)

    assert not (tmp_path / "logs" / "hparams.json").exists()
    assert not agent.saved


def test_run_recorder_logs_recordings_to_tensorboard_after_export(monkeypatch, tmp_path):
    calls = []

    def _export(cfg):
        calls.append(("export", cfg.root))

    def _log(agents, cfg):
        calls.append(("tensorboard", cfg.root))

    monkeypatch.setattr("nett_skrl.recording.export_recordings", _export)
    monkeypatch.setattr("nett_skrl.recording.log_recording_videos_to_tensorboard", _log)
    agent = _FakeAgent()
    recorder = RunRecorder([agent], num_envs=1)

    incomplete = TrainCfg(total_timesteps=1, hparams_dir=tmp_path)
    recorder.after_train(incomplete, elapsed_s=1.0)
    assert calls == []

    recorder.after_train(
        TrainCfg(total_timesteps=1, hparams_dir=tmp_path),
        elapsed_s=1.0,
        record_cfg=RecordingCfg(root=tmp_path / "recordings", egocentric_enabled=True),
    )
    assert calls == [
        ("export", tmp_path / "recordings"),
        ("tensorboard", tmp_path / "recordings"),
    ]


def test_run_recorder_rollout_finalizer_exports_logs_then_finishes(monkeypatch, tmp_path):
    calls = []

    class _Run:
        def finish(self):
            calls.append("finish")

    def _export(cfg):
        calls.append("export")

    def _log(agents, cfg):
        calls.append("tensorboard")

    monkeypatch.setattr("nett_skrl.recording.export_recordings", _export)
    monkeypatch.setattr("nett_skrl.recording.log_recording_videos_to_tensorboard", _log)

    agent = _FakeAgent()
    agent._nett_wandb_run = _Run()
    RunRecorder([agent], num_envs=1).after_rollout(
        RecordingCfg(root=tmp_path / "recordings", egocentric_enabled=True)
    )

    assert calls == ["export", "tensorboard", "finish"]
    assert agent._nett_wandb_run is None


def test_run_recorder_video_logging_uses_pytorch_writer_for_recording_videos(monkeypatch, tmp_path):
    class _Writer:
        def __init__(self) -> None:
            self.videos = []
            self.flushed = False
            self.closed = False

        def add_video(self, tag, tensor, fps):
            self.videos.append((tag, tuple(tensor.shape), fps))

        def add_scalar(self, *args, **kwargs):
            raise AssertionError("recording video logger must not write scalars")

        def flush(self):
            self.flushed = True

        def close(self):
            self.closed = True

    writer = _Writer()
    agent = _FakeAgent()
    agent.experiment_dir = str(tmp_path / "wandb_runs" / "brain_1")
    recorder = RunRecorder([agent], num_envs=1)

    rec_dir = tmp_path / "recordings" / "egocentric" / "train" / "env_000000"
    rec_dir.mkdir(parents=True)
    mp4 = rec_dir / "env_000000.mp4"
    mp4.write_bytes(b"fake")

    def _add_video(writer, mp4_path, *, tag, fps):
        writer.add_video(tag, torch.zeros(1, 2, 3, 4, 4), fps)

    monkeypatch.setattr("nett_skrl.recording.tensorboard._add_video", _add_video)
    monkeypatch.setattr(
        "nett_skrl.recording.tensorboard._recording_video_writer_for_dir",
        lambda exp_dir: writer,
    )
    from nett_skrl.recording import log_recording_videos_to_tensorboard

    log_recording_videos_to_tensorboard(
        [agent],
        RecordingCfg(root=tmp_path / "recordings", fps=12, egocentric_enabled=True)
    )

    assert writer.flushed is True
    assert writer.closed is True
    assert writer.videos == [(
        "video/egocentric/env_000000/env_000000",
        (1, 2, 3, 4, 4),
        12,
    )]


def test_run_recorder_video_logging_opens_writer_only_when_videos_exist(monkeypatch, tmp_path):
    calls = []

    def _writer(exp_dir):
        calls.append(exp_dir)
        raise AssertionError("writer should not open when there are no recording videos")

    monkeypatch.setattr(
        "nett_skrl.recording.tensorboard._recording_video_writer_for_dir",
        _writer,
    )
    from nett_skrl.recording import log_recording_videos_to_tensorboard

    agent = _FakeAgent()
    agent.experiment_dir = str(tmp_path / "wandb_runs" / "brain_1")
    log_recording_videos_to_tensorboard(
        [agent],
        RecordingCfg(root=tmp_path / "recordings", egocentric_enabled=True),
    )

    assert calls == []


def test_run_recorder_does_not_require_complete_wandb_context(tmp_path):
    agent = _FakeAgent()
    agent.experiment_dir = str(tmp_path / "wandb_runs" / "brain_1")
    recorder = RunRecorder([agent], num_envs=1)

    complete = TrainCfg(
        total_timesteps=1,
        hparams_dir=tmp_path,
        output_dir=tmp_path,
        condition="Object1",
        phase="train",
        run_name="run",
    )
    recorder.after_train(complete, elapsed_s=1.0)
    assert agent.saved


def test_single_mode_passes_record_cfg_into_brain_record_and_test(monkeypatch, tmp_path):
    from nett_skrl.runtime import task_runner

    calls = []
    record_cfg = object()

    class _Config:
        dry_run = False
        seed = 123
        path = tmp_path
        num_brains = 1
        num_envs = 1
        eval_step = None
        logger = logging.getLogger("test")

        def for_mode(self, mode, **overrides):
            calls.append(("for_mode", mode, overrides))
            return self

    class _Body:
        def adjust_to_agent(self, env, **kwargs):
            calls.append(("adjust", kwargs))

        def embed(self, env, config):
            calls.append(("embed", config))
            return "loaded-env"

    class _Brain:
        def record(self, loaded, config, *, record_cfg=None):
            calls.append(("record", loaded, record_cfg))

        def test(self, loaded, config, *, record_cfg=None):
            calls.append(("test", loaded, record_cfg))
            return {}

    task = SimpleNamespace(
        config=_Config(),
        agent=SimpleNamespace(body=_Body(), brain=_Brain(), env=SimpleNamespace()),
    )

    monkeypatch.setattr(task_runner, "set_seeds", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(task_runner, "_make_record_cfg", lambda env, config: record_cfg)
    monkeypatch.setattr(task_runner, "_exit_worker_cleanly", lambda logger: calls.append(("exit", logger)))

    task_runner._run_single_mode(task, "record")
    task_runner._run_single_mode(task, "test")

    assert ("record", "loaded-env", record_cfg) in calls
    assert ("test", "loaded-env", record_cfg) in calls


def test_parallel_env_planning_caps_to_brain_multiple():
    assert capped_num_envs(
        num_brains=3,
        preferred_envs_per_brain=4,
        max_parallel_envs=None,
    ) == 12
    assert capped_num_envs(
        num_brains=3,
        preferred_envs_per_brain=4,
        max_parallel_envs=10,
    ) == 9
    assert num_env_candidates(9, 3) == [9, 6, 3]


def test_single_run_logs_wandb_instructions_without_argument_mismatch(monkeypatch, tmp_path):
    class _Brain:
        envs_per_brain = 1
        iterations_per_test_episode = {}

        def __init__(self, **kwargs):
            self.wandb_cfg = kwargs.get("wandb", {})

        def env_reward_types(self):
            return ()

        def calc_iterations(self, *args):
            pass

    class _Body:
        def adjust_to_agent(self, env, **kwargs):
            env.num_brains = kwargs["num_brains"]
            env.num_envs = kwargs["num_envs"]

    class _Environment:
        conditions = ["Object1"]
        iterations_per_test_episode = {"Object1": 1}
        num_brains = 1
        num_envs = 1

        def __init__(self, **kwargs):
            pass

    class _Executor:
        def submit(self, *args, **kwargs):
            fut = Future()
            fut.set_result(None)
            return fut

    assigned = []
    monkeypatch.setattr(nett_module, "Brain", _Brain)
    monkeypatch.setattr(nett_module, "Body", _Body)
    monkeypatch.setattr(nett_module, "Environment", _Environment)
    monkeypatch.setattr(nett_module, "build_tasks", lambda *args, **kwargs: ["task"])
    monkeypatch.setattr(nett_module, "future_wait", lambda *args, **kwargs: None)

    nett = object.__new__(NETT)
    nett.logger = logging.getLogger("test")
    nett.output_path = tmp_path
    nett.executor = _Executor()
    nett._assign_task = assigned.append

    nett.single_run(
        name="run",
        environment={"design_sheet": "design.csv", "media_root": "media"},
        brain={"wandb": {"mode": "online", "project": "nett-test"}},
        task_memory=1.0,
    )

    assert assigned == ["task"]


def test_auto_memory_resolution_searches_down_to_safe_env_count(tmp_path):
    nett = object.__new__(NETT)
    nett.logger = logging.getLogger("test")
    brain = SimpleNamespace(envs_per_brain=3)
    env = SimpleNamespace(conditions=["Object1"], num_brains=2, num_envs=6)

    class _Body:
        def __init__(self):
            self.plans = []

        def adjust_to_agent(self, env, **kwargs):
            self.plans.append(kwargs["num_envs"])
            env.num_brains = kwargs["num_brains"]
            env.num_envs = kwargs["num_envs"]

    body = _Body()

    def _estimate(self, brain, body, env, output_dir):
        if env.num_envs == 6:
            raise RuntimeError("too many envs")
        return 123.0

    nett._estimate_task_memory_via_dry_run = _estimate.__get__(nett, NETT)
    memory, num_envs = NETT._resolve_task_memory_and_envs(
        nett,
        "auto",
        brain,
        body,
        env,
        tmp_path,
        num_brains=2,
        num_envs=6,
        steps_per_episode=200,
    )

    assert memory == 123.0
    assert num_envs == 4
    assert brain.envs_per_brain == 2
    assert body.plans == [6, 4]


def test_auto_memory_resolution_falls_back_when_no_env_count_succeeds(tmp_path):
    nett = object.__new__(NETT)
    nett.logger = logging.getLogger("test")
    brain = SimpleNamespace(envs_per_brain=2)
    env = SimpleNamespace(conditions=["Object1"], num_brains=2, num_envs=4)

    class _Body:
        def adjust_to_agent(self, env, **kwargs):
            env.num_brains = kwargs["num_brains"]
            env.num_envs = kwargs["num_envs"]

    def _estimate(self, brain, body, env, output_dir):
        raise RuntimeError("nope")

    nett._estimate_task_memory_via_dry_run = _estimate.__get__(nett, NETT)
    memory, num_envs = NETT._resolve_task_memory_and_envs(
        nett,
        "auto",
        brain,
        _Body(),
        env,
        tmp_path,
        num_brains=2,
        num_envs=4,
        steps_per_episode=200,
    )

    assert memory == 6.0 * (1024**3)
    assert num_envs == 2
    assert brain.envs_per_brain == 1


class _Intrinsic:
    def __init__(self, value: float):
        self.value = value
        self.watch_calls = 0
        self.update_calls = 0

    def watch(self, *args):
        self.watch_calls += 1

    def compute(self, **kwargs):
        return torch.full_like(kwargs["rewards"], self.value)

    def update(self):
        self.update_calls += 1


def test_intrinsic_reward_is_added_before_record_transition():
    class _ConstantRewardEnv(_FakeEnv):
        def step(self, actions):
            obs, reward, terminated, truncated, infos = super().step(actions)
            return obs, torch.ones_like(reward), terminated, truncated, infos

    env = _ConstantRewardEnv(num_envs=1)
    agent = _FakeAgent()
    intrinsic = _Intrinsic(value=0.5)
    trainer = BrainTrainer(env, [agent], device="cpu")
    trainer.train(
        TrainCfg(total_timesteps=1),
        intrinsic_reward_adapters=[intrinsic],
    )

    recorded_reward = agent.record_calls[0]["rewards"]
    assert intrinsic.watch_calls == 1
    assert intrinsic.update_calls == 1
    assert recorded_reward.item() == 1.5


def test_eval_checkpoint_boundaries_use_eval_and_checkpoint_milestones():
    assert _training_boundaries(
        200_000,
        eval_freq=50_000,
        checkpoint_freq=60_000,
    ) == [50_000, 60_000, 100_000, 120_000, 150_000, 180_000, 200_000]


def test_only_known_isaac_teardown_exit_codes_are_tolerated():
    assert _is_tolerated_isaac_teardown_exit(-11)
    assert _is_tolerated_isaac_teardown_exit(139)
    assert not _is_tolerated_isaac_teardown_exit(1)


def test_write_eval_metrics_appends_csv_and_jsonl(tmp_path):
    config = TaskConfig(
        "Object1",
        tmp_path / "run",
        ["train", "test"],
        current_mode="test",
        eval_step=50_000,
    )

    class _Brain:
        steps_per_episode = 200
        test_iterations = {"Object1": 2}

    _write_eval_metrics(config, {0: 0.25, 1: 0.5}, _Brain())
    _write_eval_metrics(config, {0: 0.75}, _Brain())

    csv_path = tmp_path / "run" / "Object1" / "logs" / "eval_metrics.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["eval_step"] for r in rows] == ["50000", "50000", "50000"]
    assert [r["brain_id"] for r in rows] == ["1", "2", "1"]
    assert rows[0]["timesteps"] == "400"

    jsonl = (tmp_path / "run" / "Object1" / "logs" / "eval_metrics.jsonl").read_text().splitlines()
    assert len(jsonl) == 3
    assert json.loads(jsonl[0])["condition"] == "Object1"
