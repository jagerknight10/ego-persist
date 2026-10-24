#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
max_bytes="${EGOPERSIST_MAX_TRACKED_BYTES:-5242880}"
python_bin="${EGOPERSIST_PYTHON:-python3.12}"

if [[ ! "$max_bytes" =~ ^[0-9]+$ ]] || [[ "$max_bytes" -le 0 ]]; then
    printf 'ERROR: EGOPERSIST_MAX_TRACKED_BYTES must be a positive integer.\n' >&2
    exit 1
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'ERROR: Python is not available: %s\n' "$python_bin" >&2
    exit 1
fi

exec "$python_bin" "${script_dir}/scan_repository.py" \
    --project-root "$project_root" \
    --max-bytes "$max_bytes" \
    --deny 'adt_urls*.txt' \
    --deny '*download_manifest*'
