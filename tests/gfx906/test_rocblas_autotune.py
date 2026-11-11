"""serve/rocblas_autotune.py's gating, with a stand-in for tune_rocblas (no GPU)."""
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve import rocblas_autotune as at  # noqa: E402


def fake_tool(tmp_path, code=0):
    tool = tmp_path / "tune_rocblas"
    tool.write_text(f'#!/bin/sh\necho "$@" > "{tmp_path}/args"\necho "tune: status"\necho noise\nexit {code}\n')
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return {"PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "STRATA_ROCBLAS_TUNING": str(tmp_path / "t")}


def test_runs_by_default_and_prints_only_status(tmp_path, capsys):
    env = fake_tool(tmp_path)
    assert at.ensure_table(env, "hip") == "ran"
    assert (tmp_path / "args").read_text().split() == ["--ensure", str(tmp_path / "t")]
    out = capsys.readouterr().out
    assert "tune: status" in out and "noise" not in out


def test_env_turns_it_off(tmp_path, capsys):
    env = fake_tool(tmp_path) | {"STRATA_ROCBLAS_AUTOTUNE": "0"}
    assert at.ensure_table(env, "hip") == "disabled"
    assert not (tmp_path / "args").exists() and "off" in capsys.readouterr().out


def test_a_table_file_is_left_alone(tmp_path):
    env = fake_tool(tmp_path)
    table = tmp_path / "x.txt"
    table.write_text("x")
    env["STRATA_ROCBLAS_TUNING"] = str(table)
    assert at.ensure_table(env, "hip") == "pinned" and not (tmp_path / "args").exists()


def test_other_backends_and_no_tool(tmp_path):
    env = fake_tool(tmp_path)
    assert at.ensure_table(env, "cuda") == "no-tool"
    assert at.ensure_table({"PATH": str(tmp_path / "none"), "STRATA_ROCBLAS_TUNING": "x"}, "hip") == "no-tool"


def test_failure_does_not_raise(tmp_path):
    assert at.ensure_table(fake_tool(tmp_path, code=7), "hip") == "failed"
