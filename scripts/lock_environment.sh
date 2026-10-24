#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
version_file="${project_root}/environment/uv-version.txt"
expected_version="$(tr -d '[:space:]' < "$version_file")"

if ! command -v uv >/dev/null 2>&1; then
    printf 'ERROR: uv is not installed. Install uv %s first.\n' \
        "$expected_version" >&2
    exit 1
fi
actual_version="$(uv --version | awk '{print $2}')"
if [[ "$actual_version" != "$expected_version" ]]; then
    printf 'ERROR: uv version is %s; expected %s.\n' \
        "$actual_version" "$expected_version" >&2
    exit 1
fi
if ! command -v python3.11 >/dev/null 2>&1; then
    printf 'ERROR: python3.11 is not installed.\n' >&2
    exit 1
fi
if [[ ! -f "${project_root}/pyproject.toml" ]]; then
    printf 'ERROR: pyproject.toml is missing.\n' >&2
    exit 1
fi

cd "$project_root"
uv lock --python 3.11
uv lock --check
printf 'Wrote %s/uv.lock\n' "$project_root"
