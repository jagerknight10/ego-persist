"""RunManifest builder tests (no data, no git dependency beyond repo root)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from wise_egomap.evaluation.manifest import build_run_manifest


def test_manifest_fields_and_hashes(tmp_path: Path) -> None:
    config = tmp_path / "model.yaml"
    config.write_text("version: 1\ninput_dim: 20\n")
    split = tmp_path / "split.json"
    split.write_text('{"train": ["a"]}\n')
    label = tmp_path / "labels.yaml"
    label.write_text("version: 1\n")

    manifest = build_run_manifest(
        run_id="test-run",
        config_path=config,
        split_file=split,
        label_config_path=label,
        project_root=tmp_path,
        extra={"n_decisions": 3},
    )
    assert manifest.run_id == "test-run"
    assert manifest.config_hash == hashlib.sha256(config.read_bytes()).hexdigest()
    assert manifest.data_split_hash == hashlib.sha256(split.read_bytes()).hexdigest()
    assert manifest.label_config_hash == hashlib.sha256(label.read_bytes()).hexdigest()
    assert manifest.seed == 0
    assert manifest.started_at_utc.tzinfo is not None
    assert manifest.extra == {"n_decisions": 3}
    # round-trips through JSON
    d = manifest.to_dict()
    assert d["run_id"] == "test-run"
