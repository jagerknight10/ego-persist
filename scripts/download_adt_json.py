"""Download ADT subsets from the signed JSON URL manifest.

The manifest is the user-provided file with this layout:

    {"sequences": {"<SEQUENCE_ID>": {"<ASSET_TYPE>": {
        "filename": ..., "sha1sum": ..., "file_size_bytes": ...,
        "download_url": "https://..."
    }, ...}, ...}}

Each selected asset is downloaded to <out>/<SEQUENCE_ID>/<ASSET_TYPE>.tmp,
verified by SHA-1, unzipped into <out>/<SEQUENCE_ID>/<ASSET_TYPE>/, and
made read-only. HTTPS only. The manifest must stay outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import zipfile
from collections.abc import Sequence
from pathlib import Path

DEFAULT_TYPES = ("main_groundtruth", "mps_slam_calibration")


def load_manifest(path: Path) -> dict[str, dict[str, dict]]:
    raw = json.loads(path.read_text())
    sequences = raw.get("sequences")
    if not isinstance(sequences, dict) or not sequences:
        raise SystemExit("manifest has no 'sequences' object")
    return sequences


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    if not url.startswith("https://"):
        raise SystemExit(f"URL is not HTTPS: {url}")
    temporary = target.with_name(target.name + ".part")
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--connect-timeout",
            "30",
            "--retry",
            "3",
            "--max-redirs",
            "5",
            "--output",
            str(temporary),
            url,
        ],
        check=False,
    )
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"download failed for {url}")
    temporary.replace(target)


def make_read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(stat.S_IMODE(mode) & 0o444)


def _cleanup_staging(staging: Path) -> None:
    """Remove staging leftovers when a download fails mid-way."""
    if not staging.exists():
        return
    for child in staging.iterdir():
        if child.is_dir():
            import shutil

            shutil.rmtree(child)
        else:
            child.unlink()
    staging.rmdir()


def extract(asset_path: Path, dest: Path) -> bool:
    """Extract an archive, or retain a raw media file as one asset."""
    if zipfile.is_zipfile(asset_path):
        with zipfile.ZipFile(asset_path) as archive:
            names = archive.namelist()
            if any(name.startswith("/") or ".." in name for name in names):
                raise SystemExit(f"unsafe zip path in {asset_path.name}")
            archive.extractall(dest)
        return True
    return False


def run_download(
    manifest_path: Path,
    sequence_ids: Sequence[str],
    asset_types: Sequence[str],
    out_root: Path,
    verify: bool = True,
    max_file_size_mb: float = 100.0,
    allow_large_files: bool = False,
) -> int:
    sequences = load_manifest(manifest_path)
    out_root.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for sequence_id in sequence_ids:
        if sequence_id not in sequences:
            raise SystemExit(f"sequence not in manifest: {sequence_id}")
        assets = sequences[sequence_id]
        for asset_type in asset_types:
            if asset_type not in assets:
                raise SystemExit(
                    f"asset type not in manifest for {sequence_id}: {asset_type}"
                )
            entry = assets[asset_type]
            declared_size = entry.get("file_size_bytes")
            if (
                isinstance(declared_size, (int, float))
                and declared_size > max_file_size_mb * 1024 * 1024
                and not allow_large_files
            ):
                raise SystemExit(
                    f"refusing {asset_type} for {sequence_id}: "
                    f"declared size {declared_size / 1048576:.2f} MB exceeds "
                    f"the {max_file_size_mb:.2f} MB limit; "
                    "use the cluster-only --allow-large-files option"
                )
            seq_dir = out_root / sequence_id
            seq_dir.mkdir(parents=True, exist_ok=True)
            dest = seq_dir / asset_type
            if dest.exists():
                print(f"skip (exists): {dest}")
                continue
            staging = seq_dir / f".{asset_type}.staging"
            if staging.exists():
                for child in staging.iterdir():
                    child.unlink()
            staging.mkdir(exist_ok=True)
            zip_path = staging / entry["filename"]
            try:
                download(entry["download_url"], zip_path)
                if verify:
                    expected = entry.get("sha1sum", "")
                    if expected:
                        actual = sha1_file(zip_path)
                        if actual != expected:
                            raise SystemExit(
                                f"SHA-1 mismatch for {zip_path.name}: "
                                f"expected {expected}, got {actual}"
                            )
                    expected_size = entry.get("file_size_bytes")
                    if expected_size and zip_path.stat().st_size != expected_size:
                        raise SystemExit(
                            f"size mismatch for {zip_path.name}: "
                            f"expected {expected_size}, got {zip_path.stat().st_size}"
                        )
                is_archive = extract(zip_path, staging)
                if is_archive:
                    zip_path.unlink()
                dest.mkdir()
                for child in sorted(staging.iterdir()):
                    child.rename(dest / child.name)
                staging.rmdir()
                dest.chmod(0o755)
                downloaded += 1
            except BaseException:
                _cleanup_staging(staging)
                raise
            for root_dir, _dirs, files in os.walk(dest):
                for name in files:
                    make_read_only(Path(root_dir) / name)
            print(f"downloaded {sequence_id}/{asset_type}")
    return downloaded


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument(
        "--types",
        default=",".join(DEFAULT_TYPES),
        help="Comma-separated asset types. Default: %(default)s",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip SHA-1 and size checks."
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=100.0,
        help="Refuse larger manifest files unless --allow-large-files is set.",
    )
    parser.add_argument(
        "--allow-large-files",
        action="store_true",
        help="Allow files above the safety limit; use only on cluster storage.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    asset_types = tuple(item for item in arguments.types.split(",") if item)
    count = run_download(
        arguments.manifest,
        arguments.sequence,
        asset_types,
        arguments.out,
        verify=not arguments.no_verify,
        max_file_size_mb=arguments.max_file_size_mb,
        allow_large_files=arguments.allow_large_files,
    )
    print(json.dumps({"downloaded": count, "sequences": len(arguments.sequence)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
