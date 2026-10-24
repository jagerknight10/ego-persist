#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"

files=()
while IFS= read -r file; do
    files+=("$file")
done < <(
    find "${project_root}/jobs" "${project_root}/scripts" \
        -type f \( -name '*.sh' -o -name '*.sbatch' \) -print | sort
)
if [[ ${#files[@]} -eq 0 ]]; then
    printf 'ERROR: No shell files found.\n' >&2
    exit 1
fi

failed=0
for file in "${files[@]}"; do
    printf 'Checking %s\n' "${file#"${project_root}/"}"
    if ! bash -n "$file"; then
        failed=1
    fi
    if [[ "$file" == *.sh && ! -x "$file" ]]; then
        printf 'ERROR: Shell script is not executable: %s\n' \
            "${file#"${project_root}/"}" >&2
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    exit 1
fi
if command -v shellcheck >/dev/null 2>&1; then
    printf 'Running ShellCheck\n'
    if ! shellcheck "${files[@]}"; then
        exit 1
    fi
else
    printf 'ShellCheck is not installed; syntax checks only.\n'
fi
printf 'Shell syntax checks passed.\n'
