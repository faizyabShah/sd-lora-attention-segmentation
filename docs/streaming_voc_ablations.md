# Full VOC-Sim streaming ablations

This experiment evaluates all 490 annotated entries in OVAM's 500-row VOC-Sim
manifest. Each array task owns one of the 20 classes. A task generates one
sample, evaluates every enabled configuration, atomically checkpoints only
metric counts, releases the sample's attention tensors, and moves to the next
sample. Generated images, latents, and attention tensors are never written.

## Reference baseline

`baseline/ovam_unoptimized` reproduces OVAM's ordinary prompt baseline:

1. use unconditional-branch cross-attention queries;
2. compute keys from the ordinary, unoptimized prompt embedding;
3. apply softmax across the unpadded prompt tokens;
4. sum all 30 denoising steps, attention heads, and non-middle cross-attention
   blocks after bilinear resizing to 64 by 64;
5. select the target position exactly as OVAM does (`len(prompt.split())`);
6. resize to the annotation size, divide by the maximum, and threshold with
   strict `> 0.4`.

There is no token optimization, morphology, self-attention refinement, or CRF
in this baseline.

## Included families

- the exact unoptimized OVAM reference;
- timestep windows, all nonempty resolution combinations, and their Cartesian
  product;
- individual cross-attention layers, regions, leave-one-out, leave-two-out,
  and every nonempty layer subset;
- threshold and normalization calibration;
- fixed Stage 1 time, head, fusion, and morphology configurations;
- exact QK self-attention screens at 8, 16, 32, and 64;
- the 12 previously tested ascending and descending progressive SA sequences.

The learned layer weights and head rankings are frozen from the earlier
100-image development run and evaluated unchanged. They are never refit on an
image being scored. The old per-class layer lookup exists for only four VOC
classes, so it is not presented as a 20-class method.

The direct timestep, resolution, and layer studies use OVAM maps and keep the
published resize, maximum normalization, and strict 0.4 threshold fixed. The
calibrated Stage 1 and self-attention families are labeled separately and keep
the settings selected by the earlier 100-image development experiments.

## Run and resume

```bash
sbatch jobs/run_streaming_voc_ablations.slurm
```

Tasks request one GPU and 32 GB of system RAM, with no node pinned. At the
12-hour warning, each task stops the current process and requeues itself. The
last fully evaluated image remains in `checkpoint.pt`, so at most the current
image is repeated. A genuine Python error is not automatically hidden by a
requeue.

After all 20 array tasks have a `COMPLETE` marker:

```bash
sbatch jobs/merge_streaming_voc_ablations.slurm
```

The merge produces compact metric outputs under
`outputs/segmentation/voc_sim_full_streaming`. It refuses to merge an incomplete
set of shards.

