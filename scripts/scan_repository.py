"""Reject tracked large files and common secret patterns."""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

FORBIDDEN_SUFFIXES = (
    ".sif",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".parquet",
    ".vrs",
    ".zip",
    ".tar",
    ".tar.gz",
    ".bag",
    ".mcap",
    ".mp4",
    ".mkv",
    ".mov",
)
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
}


def tracked_files(project_root: Path) -> list[Path]:
    """Return tracked and non-ignored files, or fall back to all files."""
    import subprocess

    result = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        return list(
            dict.fromkeys(
                project_root / item.decode("utf-8")
                for item in result.stdout.split(b"\0")
                if item
            )
        )
    return sorted(
        path
        for path in project_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )


def scan(
    project_root: Path,
    *,
    max_bytes: int,
    extra_denied_globs: Iterable[str] = (),
) -> list[str]:
    """Return a list of policy violations."""
    errors: list[str] = []
    denied = tuple(extra_denied_globs)
    for path in tracked_files(project_root):
        relative = path.relative_to(project_root)
        name = relative.as_posix()
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            errors.append(f"missing tracked file: {name}")
            continue
        if size > max_bytes:
            errors.append(f"file is larger than {max_bytes} bytes: {name}")
        if name.endswith(FORBIDDEN_SUFFIXES):
            errors.append(f"forbidden artifact type: {name}")
        if any(fnmatch.fnmatch(name, pattern) for pattern in denied):
            errors.append(f"forbidden path pattern: {name}")
        if size > 2 * 1024 * 1024 or path.suffix in {".png", ".jpg", ".jpeg", ".gif"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label}: {name}")
    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--deny", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print policy violations and return a suitable status."""
    arguments = parse_args(argv)
    if arguments.max_bytes <= 0:
        raise ValueError("max-bytes must be positive")
    errors = scan(
        arguments.project_root.resolve(),
        max_bytes=arguments.max_bytes,
        extra_denied_globs=arguments.deny,
    )
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print("Repository policy check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
