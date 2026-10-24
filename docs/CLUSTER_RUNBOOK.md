# EgoPersist Cluster Runbook

Status: Template. Fill each placeholder before use.

## 1. Values to Fill

| Name | Value |
|---|---|
| Cluster host | `<HOST>` |
| Project root | `<PROJECT_ROOT>` |
| Raw data root | `<DATA_ROOT>` |
| Run root | `<RUN_ROOT>` |
| Partition | `<GPU_PARTITION>` |
| Account | `<ACCOUNT>` |
| GPU type | `<GPU_TYPE>` |
| GPU memory | `<GPU_MEMORY>` |
| Container runtime | `<APPTAINER_OR_SINGULARITY>` |
| Module commands | `<MODULES>` |

## 2. Rules

1. Do not download data on the laptop.
2. Do not commit the ADT manifest or any data.
3. Keep raw data read-only.
4. Use node-local scratch for staging.
5. Copy results back to the run root.
6. Record the Git commit in every run manifest.

## 3. First-Time Setup

```bash
ssh <HOST>
cd <PROJECT_ROOT>
git clone <REPO_URL> code
cd code
git checkout <TAG_OR_COMMIT>
```

Create the container:

```bash
apptainer build containers/egopersist.sif environment/apptainer.def
```

Set the paths:

```bash
cat > .env.cluster <<'EOF'
EGOPERSIST_DATA_ROOT=<DATA_ROOT>
EGOPERSIST_RUN_ROOT=<RUN_ROOT>
EOF
```

## 4. ADT Access

1. Accept the ADT licence with the project account.
2. Request the temporary download-link file.
3. Put the file at `<DATA_ROOT>/manifests/adt_urls.txt`.
4. Set owner-only permissions.
5. Do not copy the file to Git.

## 5. Sample Download

```bash
bash scripts/download_adt_subset.sh \
  --manifest <DATA_ROOT>/manifests/adt_urls.txt \
  --sequence <SEQUENCE_ID> \
  --types 0,6 \
  --out <DATA_ROOT>/sample
```

## 6. Smoke Test

```bash
srun apptainer exec --nv \
  --bind <DATA_ROOT>:/data:ro,<RUN_ROOT>:/runs \
  containers/egopersist.sif \
  python -m pytest tests -q
```

## 7. Label Audit

```bash
sbatch jobs/preprocess.sbatch
```

Check these outputs:

```text
<RUN_ROOT>/labels/<SPLIT_HASH>/labels.parquet
<RUN_ROOT>/labels/<SPLIT_HASH>/audit.md
```

## 8. Feature Extraction

```bash
sbatch jobs/features.sbatch
```

Check the feature cache and its manifest.

## 9. Training

```bash
sbatch --array=0-2 jobs/train.sbatch
```

Check these outputs:

```text
<RUN_ROOT>/<RUN_ID>/resolved_config.yaml
<RUN_ROOT>/<RUN_ID>/provenance.json
<RUN_ROOT>/<RUN_ID>/metrics.json
<RUN_ROOT>/<RUN_ID>/checkpoint.pt
```

## 10. Evaluation

```bash
sbatch jobs/eval.sbatch
```

Check the Pareto plot and the error table.

## 11. Copy Results

```bash
rsync -av <HOST>:<RUN_ROOT>/<RUN_ID>/ ./results/<RUN_ID>/
```

Do not copy raw data.

## 12. Failure Checks

| Symptom | Check |
|---|---|
| No GPU | `--gres` value and partition |
| CUDA error | Container CUDA version and driver |
| Missing data | Bind mounts and `DATA_ROOT` |
| Slow reads | Stage the data to scratch |
| Out of memory | Batch size and cache size |
| Different scores | Seed, label hash, and Git commit |
