"""Tests for the offline viewer, video path, and CLI output."""

from __future__ import annotations

from pathlib import Path

import pytest

from egopersist.cli.main import main
from egopersist.visualization import build_demo_replay
from egopersist.visualization.video import write_frame_png
from egopersist.visualization.viewer import (
    render_replay_html,
    write_replay_html,
    write_replay_json,
)


def test_viewer_html_is_self_contained() -> None:
    replay = build_demo_replay()
    html = render_replay_html(replay)
    assert "<!doctype html>" in html
    assert "Passing person" in html
    assert "Durable mug move" in html
    assert "Occlusion" in html
    assert "http://" not in html
    assert "https://" not in html
    assert "src=" not in html


def test_viewer_files_are_written(tmp_path: Path) -> None:
    replay = build_demo_replay()
    json_path = write_replay_json(replay, tmp_path / "replay.json")
    html_path = write_replay_html(replay, tmp_path / "index.html")
    assert json_path.read_text(encoding="utf-8").startswith("{")
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_generate_cli_writes_viewer(tmp_path: Path) -> None:
    assert main(["generate", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "replay.json").is_file()
    assert (tmp_path / "index.html").is_file()


def test_query_cli_answers_mug_question(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "query",
                "--scenario",
                "durable_mug_move",
                "--time",
                "9.0",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Mug is at" in output
    assert "0.42" in output
    assert "confidence" in output


def test_frame_render_path(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    scenario = build_demo_replay().scenario("occlusion")
    path = write_frame_png(scenario, tmp_path / "frame.png", frame_index=6)
    assert path.is_file()
    assert path.stat().st_size > 1000
