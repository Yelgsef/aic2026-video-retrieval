# AIC production preprocessing

## Audited dataset

- 873 logical videos / 961 physical MP4 files
- 130.66 hours, 82.98 GB of canonical video
- 177,321 provided keyframes
- 871 videos with complete provided visual artifacts
- Only 3 videos with transcript JSON in the current archive

The manifest is `workspace/manifests/videos.parquet`. Re-run the audit only
after adding or changing archive files:

```powershell
uv run python scripts/audit_dataset.py --progress-every 100
```

## Selected production path

```text
OmniShotCut
  -> shot-local stride 5
  -> SigLIP 2 Base 224 on all samples
  -> adaptive clustering and redundancy removal
  -> SigLIP 2 SO400M 384 on final frames only
  -> Faster-Whisper large-v3, float16, beam 1
```

The SO400M all-sample configuration remains an experiment. On the median
benchmark video it took 148 seconds versus 73 seconds for Base. The two-tier
path retains the stronger SO400M representation on the compact index without
paying its cost on every stride sample.

See `workspace/benchmarks/BENCHMARK_SUMMARY.md` for measurements and ETA.

## Scheduler

The scheduler stores state in `workspace/scheduler/jobs.sqlite3`. SQLite uses
WAL and full synchronization. Jobs left with an expired running lease are
returned to pending, and existing valid artifacts are reconciled as completed.

Initialize four shards and inspect status:

```powershell
uv run python scripts/run_dataset_scheduler.py --init --shard-count 4 --status
```

Run one shard until empty:

```powershell
uv run python scripts/run_dataset_scheduler.py `
  --run --shard-count 4 --shard-index 0 --worker-id gpu0
```

Graceful stop uses `Ctrl+C`. The current subprocess is allowed to finish, then
the worker exits without claiming a new job. After an abrupt shutdown, run the
same command again; reconciliation preserves completed artifacts and reclaims
stale jobs.

For a smoke test, cap the work:

```powershell
uv run python scripts/run_dataset_scheduler.py `
  --run --shard-count 4 --shard-index 0 --max-jobs 1 --stages omnishot
```

Do not run multiple GPU-heavy stages concurrently on the same 8 GB GPU. On a
four-GPU Vast host, run one shard per GPU and set `CUDA_VISIBLE_DEVICES` for
each worker.

## Outputs

- Shot ranges: `workspace/shots/<video_id>.json`
- Samples: `workspace/samples/<video_id>.parquet`
- Base final frames: `workspace/keyframes/google_siglip2_base_patch16_224/`
- Base compact index: `workspace/retrieval/google_siglip2_base_patch16_224/<video_id>_unified/`
- SO400M final index: `workspace/retrieval/google_siglip2_so400m_patch14_384/<video_id>_final_from_google_siglip2_base_patch16_224/`
- ASR: `workspace/asr/faster_whisper_large_v3/<video_id>.json`

Workspace artifacts are intentionally excluded from Git. Back up the scheduler
database, manifests, completed embeddings, ASR JSON, and future vector database
snapshots to storage outside an ephemeral Vast container.
