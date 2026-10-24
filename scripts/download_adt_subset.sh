#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cluster_common.sh
source "${script_dir}/cluster_common.sh"

usage() {
    cat >&2 <<'EOF'
Usage: download_adt_subset.sh --manifest FILE --sequence ID --types LIST --out DIR

Download selected ADT files on the cluster. The licence must be approved.
The manifest must stay outside Git and outside the output directory.
EOF
    exit 2
}

manifest=""
sequence=""
types=""
out=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)
            [[ $# -ge 2 ]] || usage
            manifest="$2"
            shift 2
            ;;
        --sequence)
            [[ $# -ge 2 ]] || usage
            sequence="$2"
            shift 2
            ;;
        --types)
            [[ $# -ge 2 ]] || usage
            types="$2"
            shift 2
            ;;
        --out)
            [[ $# -ge 2 ]] || usage
            out="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

[[ -n "$manifest" && -n "$sequence" && -n "$types" && -n "$out" ]] || usage
require_env WISE_EGOMAP_DATA_ROOT
require_env WISE_EGOMAP_ADT_LICENCE_APPROVED
if [[ "$WISE_EGOMAP_ADT_LICENCE_APPROVED" != "yes" ]]; then
    cluster_fail "ADT licence approval is necessary. Set WISE_EGOMAP_ADT_LICENCE_APPROVED=yes."
fi
if [[ ! "$sequence" =~ ^[A-Za-z0-9_-]+$ ]]; then
    cluster_fail "Sequence ID has unsafe characters."
fi
if [[ ! "$types" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    cluster_fail "Types must be a comma-separated list of integers."
fi
require_command curl

require_file "$manifest" "ADT manifest"
manifest_real="$(canonical_path "$manifest" "ADT manifest")"
out="$(canonical_path "$out" "output directory")"
data_real="$(canonical_path "$WISE_EGOMAP_DATA_ROOT" "WISE_EGOMAP_DATA_ROOT")"
case "$manifest_real" in
    "${data_real}/"*) ;;
    *)
        cluster_fail "The ADT manifest must be under WISE_EGOMAP_DATA_ROOT."
        ;;
esac
case "$manifest_real" in
    "${out%/}/"*)
        cluster_fail "Output directory must not contain the manifest."
        ;;
esac
if [[ ! -d "$out" ]]; then
    cluster_fail "Output directory does not exist: ${out}"
fi

sequence_dir="${out}/${sequence}"
file_types="${types//,/ }"
for file_type in $file_types; do
    if ! awk -F '\t' -v sequence="$sequence" -v type="$file_type" '
        $1 !~ /^#/ && NF >= 3 && $1 == sequence && $2 == type { found = 1 }
        END { exit !found }
    ' "$manifest_real"; then
        cluster_fail "No manifest entry for ${sequence}_${file_type}."
    fi
done

mkdir -- "$sequence_dir"
chmod 750 "$sequence_dir"

for file_type in $file_types; do
    matches=()
    while IFS=$'\t' read -r listed_sequence listed_type url; do
        [[ -n "$url" ]] || continue
        if [[ "$listed_sequence" == "$sequence" && "$listed_type" == "$file_type" ]]; then
            matches+=("$url")
        fi
    done < <(
        awk -F '\t' -v sequence="$sequence" -v type="$file_type" '
            $1 !~ /^#/ && NF >= 3 && $1 == sequence && $2 == type {
                print $1 "\t" $2 "\t" $3
            }
        ' "$manifest_real"
    )
    index=0
    for url in "${matches[@]}"; do
        if [[ ! "$url" =~ ^https:// ]]; then
            cluster_fail "Manifest URL is not HTTPS for ${sequence}_${file_type}."
        fi
        filename="${url%%\?*}"
        filename="${filename##*/}"
        if [[ -z "$filename" || ! "$filename" =~ ^[A-Za-z0-9._-]+$ ]]; then
            cluster_fail "Manifest URL has an unsafe file name for ${sequence}_${file_type}."
        fi
        if [[ ${#matches[@]} -gt 1 ]]; then
            filename="${index}_${filename}"
        fi
        target="${sequence_dir}/${filename}"
        if [[ -e "$target" ]]; then
            cluster_fail "Download target already exists: ${target}"
        fi
        temporary="${target}.part"
        curl --fail --location --silent --show-error \
            --proto '=https' --proto-redir '=https' \
            --tlsv1.2 --connect-timeout 30 --retry 3 --max-redirs 5 \
            --output "$temporary" "$url"
        mv -- "$temporary" "$target"
        index=$((index + 1))
    done
done

chmod 440 "${sequence_dir}"/*
printf 'Downloaded ADT subset to %s\n' "$sequence_dir"
