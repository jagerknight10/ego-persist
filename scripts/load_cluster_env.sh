#!/usr/bin/env bash
# Source this file on the cluster before submitting a WISE-EgoMap job.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"

export PROJECT_ROOT="${project_root}"
export DATA_ROOT="${HOME}/wise-egomap-data"
export RUN_ROOT="${HOME}/wise-egomap-runs"
export VENV_ROOT="${HOME}/wise-egomap-venv"

export WISE_EGOMAP_RUNTIME="venv"
export WISE_EGOMAP_VENV="${VENV_ROOT}"
export WISE_EGOMAP_PROJECT_ROOT="${PROJECT_ROOT}"
export WISE_EGOMAP_DATA_ROOT="${DATA_ROOT}"
export WISE_EGOMAP_RUN_ROOT="${RUN_ROOT}"
export WISE_EGOMAP_DATA_CONFIG="${PROJECT_ROOT}/configs/data/adt_v1.yaml"
export WISE_EGOMAP_LABEL_CONFIG="${PROJECT_ROOT}/configs/labels_v1.yaml"
export WISE_EGOMAP_EXPERIMENT_CONFIG="${PROJECT_ROOT}/configs/experiment/evidence_gru.yaml"
export WISE_EGOMAP_MODEL_CONFIG="${PROJECT_ROOT}/configs/model/evidence_gru.yaml"
export WISE_EGOMAP_SPLIT_FILE="${DATA_ROOT}/splits/adt_splits_v1.json"
export WISE_EGOMAP_PREPROCESS_MODULE="wise_egomap.data.cli"
export WISE_EGOMAP_FEATURES_MODULE="wise_egomap.evidence.cli"
export WISE_EGOMAP_TRAIN_MODULE="wise_egomap.evidence.cli"
export WISE_EGOMAP_EVAL_MODULE="wise_egomap.evaluation.cli"
