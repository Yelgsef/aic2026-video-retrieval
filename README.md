# AIC 2026 Offline Keyframe Pipeline

This repository currently tracks the reproducible offline pipeline through
BEiT-3 adaptive keyframe selection:

```text
dataset audit
  -> OmniShotCut shot segmentation
  -> shot-local stride sampling
  -> pluggable visual encoder (BEiT-3 baseline)
  -> adaptive spherical clustering
  -> medoid selection
  -> local/global/adjacent-shot redundancy elimination
```

Dataset files, checkpoints, virtual environments, and generated artifacts are
excluded from Git. Retrieval, ASR, OCR, and multimodal fusion are intentionally
kept outside this milestone while they are being evaluated.

## Runtime environments

```powershell
uv sync --project environments\omnishotcut
uv sync --project environments\beit3-timm
```

OmniShotCut is pinned as a Git submodule under `third_party/OmniShotCut`.

## Pipeline example

```powershell
uv run --project environments\omnishotcut python scripts\audit_dataset.py

uv run --project environments\omnishotcut python scripts\run_omnishotcut.py `
  --video-id L21_V001

uv run --project environments\omnishotcut python scripts\build_sampling_manifest.py `
  --video-id L21_V001 --stride 5 --stride-only

uv run --project environments\beit3-timm python scripts\extract_timm_beit3_embeddings.py `
  --video-id L21_V001 --batch-size 32

uv run --project environments\beit3-timm python scripts\select_keyframes.py `
  --video-id L21_V001 `
  --encoder-id timm_beit3_base_patch16_224_pt `
  --protected-anchor-policy none `
  --single-cluster-max-distance 0.01 `
  --redundancy-cosine 0.95 `
  --temporal-window-frames 15 `
  --global-within-shot-cosine 0.995 `
  --adjacent-shot-cosine 0.995
```

## Pluggable encoders

Clustering consumes a Parquet feature manifest identified by `--encoder-id`.
Another visual model can be added by producing the same metadata and embedding
columns, without changing `aic_pipeline/keyframe_selection.py`.

## Local directory layout

```text
archive/       # input dataset; ignored
models/        # downloaded checkpoints; ignored
workspace/     # shots, samples, embeddings, keyframes, reports; ignored
```

See `PIPELINE_AIC2026.md` for the larger experimental roadmap.

