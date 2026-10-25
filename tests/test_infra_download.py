"""Tests for the JSON ADT downloader without network access."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from scripts import download_adt_json as dl


def make_manifest(
    tmp_path: Path,
    *,
    sequence: str = "seqX",
    sha1: str | None = None,
    size: int | None = None,
    url: str = "https://example.test/data.zip",
    asset: str = "main_groundtruth",
) -> Path:
    payload = {"scene_objects.csv": b"x\n"}
    zip_path = tmp_path / f"{asset}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    manifest = {
        "sequences": {
            sequence: {
                asset: {
                    "filename": zip_path.name,
                    "sha1sum": sha1
                    if sha1 is not None
                    else hashlib.sha1(zip_path.read_bytes()).hexdigest(),
                    "file_size_bytes": size
                    if size is not None
                    else zip_path.stat().st_size,
                    "download_url": url,
                }
            }
        }
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def patch_download(tmp_path: Path, zip_path: Path, *, fail: bool = False):
    def fake_download(url: str, target: Path) -> None:
        if not url.startswith("https://"):
            raise SystemExit("URL is not HTTPS")
        if fail:
            return
        target.write_bytes(zip_path.read_bytes())

    dl.download = fake_download
    return fake_download


def test_download_verifies_sha1_and_extracts(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    zip_path = tmp_path / "main_groundtruth.zip"
    patch_download(tmp_path, zip_path)
    count = dl.run_download(manifest, ["seqX"], ("main_groundtruth",), tmp_path / "out")
    assert count == 1
    extracted = tmp_path / "out" / "seqX" / "main_groundtruth" / "scene_objects.csv"
    assert extracted.read_bytes() == b"x\n"
    # Files are read-only after extraction.
    assert not extracted.stat().st_mode & 0o200


def test_download_rejects_sha1_mismatch(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, sha1="0" * 40)
    zip_path = tmp_path / "main_groundtruth.zip"
    patch_download(tmp_path, zip_path)
    try:
        dl.run_download(manifest, ["seqX"], ("main_groundtruth",), tmp_path / "out")
        raise AssertionError("expected SystemExit on SHA-1 mismatch")
    except SystemExit as exc:
        assert "SHA-1 mismatch" in str(exc)
    assert not (tmp_path / "out" / "seqX" / "main_groundtruth").exists()


def test_download_rejects_size_mismatch(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, size=1)
    zip_path = tmp_path / "main_groundtruth.zip"
    patch_download(tmp_path, zip_path)
    try:
        dl.run_download(manifest, ["seqX"], ("main_groundtruth",), tmp_path / "out")
        raise AssertionError("expected SystemExit on size mismatch")
    except SystemExit as exc:
        assert "size mismatch" in str(exc)


def test_download_rejects_http_url(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, url="http://example.test/data.zip")
    try:
        dl.run_download(manifest, ["seqX"], ("main_groundtruth",), tmp_path / "out")
        raise AssertionError("expected SystemExit on non-HTTPS URL")
    except SystemExit as exc:
        assert "not HTTPS" in str(exc)


def test_download_rejects_unknown_sequence(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    try:
        dl.run_download(manifest, ["missing"], ("main_groundtruth",), tmp_path / "out")
        raise AssertionError("expected SystemExit on unknown sequence")
    except SystemExit as exc:
        assert "not in manifest" in str(exc)


def test_download_skips_existing(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    zip_path = tmp_path / "main_groundtruth.zip"
    patch_download(tmp_path, zip_path)
    assert (
        dl.run_download(manifest, ["seqX"], ("main_groundtruth",), tmp_path / "out")
        == 1
    )
    assert (
        dl.run_download(manifest, ["seqX"], ("main_groundtruth",), tmp_path / "out")
        == 0
    )


def test_download_refuses_large_manifest_file_by_default(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, size=101 * 1024 * 1024)
    try:
        dl.run_download(manifest, ["seqX"], ("main_groundtruth",), tmp_path / "out")
        raise AssertionError("expected the 100 MB safety limit")
    except SystemExit as exc:
        assert "exceeds" in str(exc)
