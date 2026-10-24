"""Tests for run provenance and split helpers."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.create_split_manifest import create_split
from scripts.provenance import build_manifest, sha256_file, write_manifest


def make_repo(path: Path) -> tuple[Path, Path]:
    path.mkdir()
    config = path / "config.yaml"
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
    )
    config.write_text("seed: 0\n", encoding="utf-8")
    subprocess.run(["git", "add", "config.yaml"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)
    return path, config


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"egopersist")
    assert sha256_file(path) == hashlib.sha256(b"egopersist").hexdigest()


def test_manifest_hashes_inputs_and_records_slurm(tmp_path: Path) -> None:
    repo, config = make_repo(tmp_path / "repo")
    split = tmp_path / "split.json"
    split.write_text('{"train": ["a"]}\n', encoding="utf-8")
    label = tmp_path / "labels.yaml"
    label.write_text("version: 1\n", encoding="utf-8")
    started = datetime(2026, 10, 24, tzinfo=UTC)

    manifest = build_manifest(
        project_root=repo,
        run_id="run-1",
        stage="train",
        config_path=config,
        data_split_path=split,
        label_config_path=label,
        seed=2,
        container_image="/images/egopersist.sif",
        container_sha256="a" * 64,
        command=("python", "-m", "train"),
        started_at_utc=started,
        environ={"SLURM_JOB_ID": "42", "SLURM_ARRAY_TASK_ID": "1"},
    )

    assert manifest.git_dirty is False
    assert manifest.seed == 2
    assert manifest.config_hash == sha256_file(config)
    assert manifest.extra["stage"] == "train"
    assert manifest.extra["slurm_job_id"] == "42"


def test_manifest_rejects_unsafe_run_id(tmp_path: Path) -> None:
    repo, config = make_repo(tmp_path / "repo")
    with pytest.raises(ValueError, match="unsafe"):
        build_manifest(
            project_root=repo,
            run_id="../escape",
            stage="eval",
            config_path=config,
            data_split_path=config,
            label_config_path=config,
            seed=0,
            container_image="image.sif",
            container_sha256="a" * 64,
            command=("true",),
        )


def test_write_manifest_is_atomic_and_private(tmp_path: Path) -> None:
    repo, config = make_repo(tmp_path / "repo")
    manifest = build_manifest(
        project_root=repo,
        run_id="run-1",
        stage="train",
        config_path=config,
        data_split_path=config,
        label_config_path=config,
        seed=0,
        container_image="image.sif",
        container_sha256="b" * 64,
        command=("true",),
    )

    output = tmp_path / "run" / "provenance.json"
    write_manifest(manifest, output)

    assert output.is_file()
    assert not (output.parent / ".provenance.json.tmp").exists()
    assert output.stat().st_mode & 0o777 == 0o600
    assert manifest.to_json() in output.read_text(encoding="utf-8")


def test_create_split_is_deterministic_and_has_all_sequences() -> None:
    sequences = tuple(f"seq-{index}" for index in range(10))
    first = create_split(
        sequences=sequences,
        seed=3,
        train_fraction=0.6,
        validation_fraction=0.2,
    )
    second = create_split(
        sequences=tuple(reversed(sequences)),
        seed=3,
        train_fraction=0.6,
        validation_fraction=0.2,
    )
    assert first == second
    all_sequences = first["train"] + first["validation"] + first["test"]
    assert sorted(all_sequences) == list(sequences)
    assert len(first["train"]) == 6
    assert len(first["validation"]) == 2
    assert len(first["test"]) == 2


def test_create_split_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="unique"):
        create_split(
            sequences=("a", "a", "b"),
            seed=0,
            train_fraction=0.6,
            validation_fraction=0.2,
        )
