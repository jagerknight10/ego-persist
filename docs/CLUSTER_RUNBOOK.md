# WISE-EgoMap Cluster Runbook

All heavy work runs on the university cluster. The laptop holds code and the
synthetic demo only. Nothing over 100 MB is downloaded off-cluster.

## 0. Layout contract

One experiment uses one base run id (`WISE_EGOMAP_RUN_ID`).

| Stage | Run directory | Reads | Writes |
|---|---|---|---|
| preprocess | `$RUN_ROOT/$RUN_ID/preprocess/` | `$DATA_ROOT/<sequence>/` | observations, labels, audit |
| features | `$RUN_ROOT/$RUN_ID/features/` | preprocess output | 20-dim features, Bayes probs |
| train | `$RUN_ROOT/$RUN_ID_train_seed_$N/` | base run features | `train/seed_$N/model.pt` + `model_probs/` |
| eval | `$RUN_ROOT/$RUN_ID_eval_bayes/` or `..._eval_model/` | base run + optional model run | report, decisions, provenance |

Set `WISE_EGOMAP_MODEL_RUN_ID` to evaluate a trained seed; leave it unset to
evaluate the Bayes baseline (CPU, no torch needed at eval time).

## 1. First-time setup (login node)

```bash
# 1. Code
cd <PROJECT_ROOT>
git clone git@github.com:jagerknight10/wise-egomap.git
cd wise-egomap
git checkout v0.1.0

# 2. Copy the signed ADT URL manifest into the data root (never into Git)
mkdir -p "$DATA_ROOT/manifests"
cp /path/to/Aria\ Digital\ Twin\ Dataset\ URLs.json "$DATA_ROOT/manifests/adt_urls.json"
chmod 600 "$DATA_ROOT/manifests/adt_urls.json"

# 3. Environment file
cp environment/cluster.env.example "$HOME/wise_egomap-cluster.env"
chmod 600 "$HOME/wise_egomap-cluster.env"
$EDITOR "$HOME/wise_egomap-cluster.env"   # fill paths, partition, account, RUN_ID
source "$HOME/wise_egomap-cluster.env"
```

## 2. Lock and image (needs network)

```bash
bash scripts/lock_environment.sh        # needs uv 0.13.5 + python3.11
bash scripts/build_container.sh         # uses $WISE_EGOMAP_IMAGE
apptainer test "$WISE_EGOMAP_IMAGE"
bash scripts/check_cluster_env.sh
```

## 3. Download the data subset (compute node)

Only two asset types are needed (poses, boxes, calibration). Video, depth,
and synthetic assets are not used by the pipeline.

Smoke set (3 sequences, ~60 MB total):

```bash
srun --partition "$WISE_EGOMAP_SLURM_PARTITION" --cpus-per-task=4 --mem=16G --time=02:00:00 bash -c '
  set -euo pipefail
  cd "$WISE_EGOMAP_PROJECT_ROOT"
  for seq in Apartment_release_clean_seq131_M1292 \
             Apartment_release_clean_seq133_M1292 \
             Apartment_release_meal_seq131_M1292; do
    python3 scripts/download_adt_json.py \
      --manifest "$WISE_EGOMAP_DATA_ROOT/manifests/adt_urls.json" \
      --sequence "$seq" \
      --types main_groundtruth,mps_slam_calibration \
      --out "$WISE_EGOMAP_DATA_ROOT/sequences"
  done
'
```

Full training set (12 sequences, ~1 GB total). Activities are separated
so train/val/test never share a scene. Run this from a login shell with
the environment sourced; the wrapper is idempotent, so an interrupted
loop can be re-run:

```bash
for seq in \
  Apartment_release_clean_seq131_M1292 \
  Apartment_release_clean_seq133_M1292 \
  Apartment_release_clean_seq134_M1292 \
  Apartment_release_meal_seq131_M1292 \
  Apartment_release_meal_seq133_M1292 \
  Apartment_release_meal_seq134_M1292 \
  Apartment_release_decoration_seq131_M1292 \
  Apartment_release_decoration_seq133_M1292 \
  Apartment_release_decoration_seq134_M1292 \
  Apartment_release_multiuser_cook_seq111_M1292 \
  Apartment_release_multiuser_cook_seq112_M1292 \
  Apartment_release_multiuser_cook_seq113_M1292; do
  python3 "$WISE_EGOMAP_PROJECT_ROOT/scripts/download_adt_json.py" \
    --manifest "$WISE_EGOMAP_DATA_ROOT/manifests/adt_urls.json" \
    --sequence "$seq" \
    --types main_groundtruth,mps_slam_calibration \
    --out "$WISE_EGOMAP_DATA_ROOT/sequences"
done
```

Each asset is SHA-1 verified. The manifest lists all 236 sequences; the
12 above are a balanced 4-activity subset (~1 GB).

## 4. Split file

Smoke split (3 sequences):

```bash
python3 scripts/create_split_manifest.py \
  --sequence Apartment_release_clean_seq131_M1292 \
  --sequence Apartment_release_clean_seq133_M1292 \
  --sequence Apartment_release_meal_seq131_M1292 \
  --seed 0 \
  --output "$WISE_EGOMAP_DATA_ROOT/splits/adt_splits_v1.json"
```

Full split (12 sequences, 60/20/20): pass all 12 sequence IDs from
section 3 to `--sequence` (one flag per sequence). At least 3 sequences
are required.

## 5. Smoke test (before any stage)

```bash
srun --partition "$WISE_EGOMAP_SLURM_PARTITION" --cpus-per-task=4 --mem=16G --time=00:15:00 \
  bash "$WISE_EGOMAP_PROJECT_ROOT/scripts/smoke_cluster.sh"
```

It must print `Cluster smoke passed.`.

## 6. Run the stages

```bash
bash scripts/submit_stage.sh preprocess
bash scripts/submit_stage.sh features
bash scripts/submit_stage.sh train --array=0-2     # 3 seeds, 1 GPU each
bash scripts/submit_stage.sh eval                   # Bayes baseline, CPU
WISE_EGOMAP_MODEL_RUN_ID="${WISE_EGOMAP_RUN_ID}_train_seed_0" \
  bash scripts/submit_stage.sh eval                 # trained model
```

Jobs are idempotent per run id. A failed stage can be resubmitted after a
fix. Logs: `$WISE_EGOMAP_RUN_ROOT/slurm/<stage>_<job>.{out,err}`.

## 7. Read the results

```bash
# Bayes baseline
cat "$WISE_EGOMAP_RUN_ROOT/$WISE_EGOMAP_RUN_ID_eval_bayes/eval/evaluation_report.json"
# Trained model
cat "$WISE_EGOMAP_RUN_ROOT/$WISE_EGOMAP_RUN_ID_eval_model/eval/evaluation_report.json"
```

Key fields: `transient_false_commit`, `persistent_change_micro`,
`commit_timing`, `final_position_error`, `pareto_point`. Check
`provenance.json` (git commit, seed, container hash, config hashes) before
comparing two runs.

Copy results to the laptop (reports only, never raw data):

```bash
rsync -av <HOST>:"$WISE_EGOMAP_RUN_ROOT/$WISE_EGOMAP_RUN_ID_eval_model/eval/" ./results/
```

## 8. Failure table

| Symptom | Check |
|---|---|
| No GPU | `--gres`, partition, `CUDA_VISIBLE_DEVICES` |
| CUDA error | Container vs host driver CUDA version |
| Missing data | `WISE_EGOMAP_DATA_ROOT/sequences/<seq>/` layout |
| Env error | `bash scripts/check_cluster_env.sh` |
| Download blocked | HTTPS URL, manifest path, disk space |
| OOM | Sequence count, batch size |
| Different scores | Seed, config hash, git commit, container hash |

## Rules

1. No data, checkpoints, manifests, or secrets in Git.
2. Raw data stays in `$WISE_EGOMAP_DATA_ROOT`, read-only for jobs.
3. `$WISE_EGOMAP_RUN_ROOT` must be outside both project and data roots.
4. Do not present results as ADT results until the licence is confirmed.
