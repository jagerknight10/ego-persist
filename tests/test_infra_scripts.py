"""Tests for cluster safety checks that do not need a cluster."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_bash(
    command: str,
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_download_wrapper_requires_licence(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "adt_urls.txt"
    manifest.write_text("sequence\t0\thttps://example.test/data\n", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    environment = {
        **os.environ,
        "WISE_EGOMAP_DATA_ROOT": str(tmp_path),
        "WISE_EGOMAP_ADT_LICENCE_APPROVED": "no",
    }

    result = run_bash(
        " ".join(
            (
                f"bash {project_root}/scripts/download_adt_subset.sh",
                f"--manifest {manifest}",
                "--sequence seq-1",
                "--types 0",
                f"--out {output}",
            )
        ),
        cwd=project_root,
        env=environment,
    )
    assert result.returncode != 0
    assert "licence approval" in result.stderr
    assert not (output / "seq-1").exists()


def test_download_wrapper_rejects_untrusted_scheme(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "adt_urls.txt"
    manifest.write_text("seq-1\t0\thttp://example.test/data\n", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    environment = {
        **os.environ,
        "WISE_EGOMAP_DATA_ROOT": str(tmp_path),
        "WISE_EGOMAP_ADT_LICENCE_APPROVED": "yes",
    }

    result = run_bash(
        " ".join(
            (
                f"bash {project_root}/scripts/download_adt_subset.sh",
                f"--manifest {manifest}",
                "--sequence seq-1",
                "--types 0",
                f"--out {output}",
            )
        ),
        cwd=project_root,
        env=environment,
    )
    assert result.returncode != 0
    assert "not HTTPS" in result.stderr
    assert not list((output / "seq-1").iterdir())


def test_cluster_env_fails_on_missing_path(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "WISE_EGOMAP_PROJECT_ROOT": str(project_root),
        "WISE_EGOMAP_DATA_ROOT": str(tmp_path / "absent"),
        "WISE_EGOMAP_RUN_ROOT": str(tmp_path),
        "WISE_EGOMAP_IMAGE": str(tmp_path / "absent.sif"),
        "WISE_EGOMAP_DATA_CONFIG": str(project_root / "configs/data/adt_v1.yaml"),
        "WISE_EGOMAP_LABEL_CONFIG": str(project_root / "configs/labels_v1.yaml"),
        "WISE_EGOMAP_EXPERIMENT_CONFIG": str(
            project_root / "configs/experiment/evidence_gru.yaml"
        ),
        "WISE_EGOMAP_MODEL_CONFIG": str(
            project_root / "configs/model/evidence_gru.yaml"
        ),
        "WISE_EGOMAP_SPLIT_FILE": str(tmp_path / "absent.json"),
    }
    result = run_bash(
        f"bash {project_root}/scripts/check_cluster_env.sh",
        cwd=project_root,
        env=environment,
    )
    assert result.returncode != 0
    assert "WISE_EGOMAP_DATA_ROOT" in result.stderr


def test_slurm_templates_require_paths_early() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for name in ("preprocess", "features", "train", "eval"):
        result = run_bash(
            f"env -u WISE_EGOMAP_PROJECT_ROOT bash {project_root}/jobs/{name}.sbatch",
            cwd=project_root,
            env={**os.environ},
        )
        assert result.returncode != 0
        assert "WISE_EGOMAP_PROJECT_ROOT" in result.stderr


def test_cpu_jobs_do_not_request_gpu() -> None:
    """Feature extraction and evaluation are deterministic CPU work."""
    project_root = Path(__file__).resolve().parents[1]
    for stage in ("features", "eval"):
        template = (project_root / "jobs" / f"{stage}.sbatch").read_text(
            encoding="utf-8"
        )
        assert "#SBATCH --gres=gpu" not in template
        assert f"prepare_job {stage} 0" in template


def test_smoke_script_fails_early_without_env() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = run_bash(
        f"env -u WISE_EGOMAP_PROJECT_ROOT bash {project_root}/scripts/smoke_cluster.sh",
        cwd=project_root,
        env={**os.environ},
    )
    assert result.returncode != 0
    assert "WISE_EGOMAP_PROJECT_ROOT" in result.stderr
