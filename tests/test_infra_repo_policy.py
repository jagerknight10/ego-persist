"""Tests for repository policy and shell entry-point checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from scripts.scan_repository import SECRET_PATTERNS, scan


def make_git_repo(path: Path, files: dict[str, bytes]) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    return path


def test_scan_rejects_large_file(tmp_path: Path) -> None:
    repo = make_git_repo(tmp_path / "repo", {"large.txt": b"x" * 20})
    errors = scan(repo, max_bytes=10)
    assert any("larger than 10 bytes" in error for error in errors)


def test_scan_rejects_checkpoint_and_manifest(tmp_path: Path) -> None:
    repo = make_git_repo(
        tmp_path / "repo",
        {"model.pt": b"model", "adt_urls_cluster.txt": b"https://example.test"},
    )
    errors = scan(repo, max_bytes=100, extra_denied_globs=("adt_urls*.txt",))
    assert any("model.pt" in error for error in errors)
    assert any("adt_urls_cluster.txt" in error for error in errors)


def test_secret_patterns_detect_common_tokens() -> None:
    assert SECRET_PATTERNS["GitHub token"].search("ghp_" + "a" * 30)
    assert SECRET_PATTERNS["AWS access key"].search("AKIA" + "A" * 16)
    assert SECRET_PATTERNS["private key"].search("-----BEGIN " + "PRIVATE KEY-----")


def test_shell_scripts_are_syntactically_valid() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(project_root / "scripts/check_shell_syntax.sh")],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_workflow_files_are_valid_yaml() -> None:
    project_root = Path(__file__).resolve().parents[1]
    workflows = sorted((project_root / ".github/workflows").glob("*.yml"))
    assert workflows
    for path in workflows:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "jobs" in data
