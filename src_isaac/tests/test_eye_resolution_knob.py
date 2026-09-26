"""NETT_EYE_RES reaches the camera; NETT_RES, which never did, is refused off its default.

⛔ Before this, ``NETT_RES`` only set ``observation.input_resolution``. Repo B's
``ObservationCfg.eye_resolution`` defaults to (128, 80) and ``lens.eye_resolution`` returns it
whenever it is set, so a ``NETT_RES=256`` arm rendered 128x80 while recording ``res=256``.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import campaign_train as campaign  # noqa: E402


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("NETT_EYE_RES", raising=False)
    monkeypatch.delenv("NETT_EYE_ASPECT_FREE", raising=False)


def test_default_defers_to_repo_b(monkeypatch):
    assert campaign.eye_resolution_override(128) is None


def test_nett_res_off_default_is_refused(monkeypatch):
    with pytest.raises(SystemExit, match="does not reach the camera"):
        campaign.eye_resolution_override(256)


@pytest.mark.parametrize("raw,want", [("256x160", (256, 160)), ("448X280", (448, 280)), (" 128x80 ", (128, 80))])
def test_parses_16_10(monkeypatch, raw, want):
    monkeypatch.setenv("NETT_EYE_RES", raw)
    assert campaign.eye_resolution_override(128) == want
    # an explicit eye makes NETT_RES irrelevant, so it is not refused alongside it
    assert campaign.eye_resolution_override(256) == want


@pytest.mark.parametrize("raw,match", [("256x256", "16:10"), ("160x256", "16:10"), ("0x0", "16:10"),
                                       ("256", "WIDTHxHEIGHT"), ("axb", "WIDTHxHEIGHT")])
def test_rejects(monkeypatch, raw, match):
    monkeypatch.setenv("NETT_EYE_RES", raw)
    with pytest.raises(SystemExit, match=match):
        campaign.eye_resolution_override(128)


@pytest.mark.parametrize("raw,want", [("448x448", (448, 448)), ("256x256", (256, 256)), ("160x256", (160, 256))])
def test_aspect_free_admits_any_aspect(monkeypatch, raw, want):
    """NETT_EYE_ASPECT_FREE=1 is the only way past the 16:10 guard (Unity's square 150x150 eye)."""
    monkeypatch.setenv("NETT_EYE_RES", raw)
    with pytest.raises(SystemExit, match="16:10"):
        campaign.eye_resolution_override(128)
    monkeypatch.setenv("NETT_EYE_ASPECT_FREE", "1")
    assert campaign.eye_resolution_override(128) == want


@pytest.mark.parametrize("flag", ["0", "false", "off"])
def test_aspect_free_off_still_refuses(monkeypatch, flag):
    monkeypatch.setenv("NETT_EYE_RES", "448x448")
    monkeypatch.setenv("NETT_EYE_ASPECT_FREE", flag)
    with pytest.raises(SystemExit, match="16:10"):
        campaign.eye_resolution_override(128)


def test_aspect_free_typo_fails_loud(monkeypatch):
    monkeypatch.setenv("NETT_EYE_RES", "448x448")
    monkeypatch.setenv("NETT_EYE_ASPECT_FREE", "flase")
    with pytest.raises((SystemExit, ValueError)):
        campaign.eye_resolution_override(128)


def test_aspect_free_does_not_admit_nonpositive_or_garbage(monkeypatch):
    monkeypatch.setenv("NETT_EYE_ASPECT_FREE", "1")
    for raw, match in [("0x448", "positive"), ("448", "WIDTHxHEIGHT")]:
        monkeypatch.setenv("NETT_EYE_RES", raw)
        with pytest.raises(SystemExit, match=match):
            campaign.eye_resolution_override(128)


def test_main_passes_a_square_eye_only_with_the_flag(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="16:10"):
        _capture_config(monkeypatch, tmp_path, NETT_EYE_RES="448x448")
    cfg = _capture_config(monkeypatch, tmp_path, NETT_EYE_RES="448x448", NETT_EYE_ASPECT_FREE="1")
    assert cfg["environment"]["eye_resolution"] == [448, 448]


def test_the_defect_and_the_fix_through_repo_b_lens():
    """The firing test at the level of the helper both repos read the camera size through."""
    lens = pytest.importorskip("nett_isaac.lens")
    # repo B's declared default, read from source (importing nett_env_cfg needs Kit)
    import ast, inspect
    src = Path(inspect.getfile(lens)).with_name("nett_env_cfg.py").read_text()
    default = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "ObservationCfg":
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and getattr(st.target, "id", "") == "eye_resolution":
                    default = ast.literal_eval(st.value)
    assert default == (128, 80), default
    # defect: input_resolution alone does not move the eye
    assert lens.eye_resolution(SimpleNamespace(input_resolution=256, eye_resolution=default)) == (128, 80)
    # fix: eye_resolution does
    assert lens.eye_resolution(SimpleNamespace(input_resolution=128, eye_resolution=(256, 160))) == (256, 160)


class _Captured(Exception):
    pass


def _capture_config(monkeypatch, tmp_path, **env):
    import nett_skrl

    class _Stub:
        def __init__(self, config):
            raise _Captured(config)

    monkeypatch.setattr(nett_skrl, "NETT", _Stub)
    for k, v in {"NETT_MODEL": "CNN2F", "NETT_EXPERIMENT": "parsing", "NETT_IMPRINT": "fork-1",
                 "NETT_OUT_ROOT": str(tmp_path), "NETT_BRAINS": "1", "NETT_MAX_ENVS": "16", **env}.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(_Captured) as got:
        campaign.main()
    return got.value.args[0]


def test_main_passes_the_eye_to_the_environment(monkeypatch, tmp_path):
    cfg = _capture_config(monkeypatch, tmp_path, NETT_EYE_RES="256x160")
    assert cfg["environment"]["eye_resolution"] == [256, 160]


def test_main_default_leaves_the_eye_to_repo_b(monkeypatch, tmp_path):
    cfg = _capture_config(monkeypatch, tmp_path)
    assert "eye_resolution" not in cfg["environment"]
