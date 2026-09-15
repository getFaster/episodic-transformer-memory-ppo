# Implementation status

This repository contains the implementation and reproducibility scaffolding for
episodic Transformer-XL PPO / MoBA. The Colab workflow in
`notebooks/episodic_moba_ppo_colab.ipynb` is a thin launcher: it checks out an
exact commit, installs the frozen environment, authenticates Drive and W&B,
runs the baseline gate, and exposes resume/train/evaluate/analyze commands.
Training logic remains in repository code.

## Baseline gate

The computational baseline evaluation produced a success rate of **0.98** and
mean normalized return of **0.993**, exceeding both configured 0.95 thresholds.
The gate is therefore considered passed for this checkpoint and protocol.

`results/baseline_reference.json` now contains the passing 100 paired episode
records and pinned provenance. The corrected protocol uses explosion delay 5
and the historical 119-step model capacity.

## Implemented

- Hash-pinned pretrained-checkpoint loading and baseline-gate validation.
- Strict experiment configuration, checkpoint provenance, atomic checkpoint
  markers, and evaluation record schemas.
- TrXL and TrXL+MoBA configuration paths and the parameterized Colab launcher.
- PEFT 0.20.0 adapters over the legacy shared-head projections, with exactly
  73,728 trainable parameters and CPU-resident episode traces.
- Fixed-budget PPO runtime, extension gate, evaluator, routing diagnostics,
  checkpoint recovery, and W&B/Drive logging seams.
- Resume/evaluation command surfaces and W&B/Drive configuration fields.

The local CPU verification suite currently passes **99 tests**. This validates
the implementation contracts; it is not evidence of T4 throughput or learning
performance.

## Remaining experimental completion

- A full **T4** training run has not been completed or claimed here.
- The required **3-seed** training/evaluation campaign has not been completed;
  one seed or a smoke check is not evidence for a campaign-level result.
- Long-run throughput, learning curves, and retrieval-quality claims remain
  experimental deliverables after the baseline gate, with durable checkpoints
  and commit markers preserved for each run.
