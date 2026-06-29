# multi-timescale-learning
Multi-timescale learning engine for continual adaptation with nested memories and continuum memory systems.

This repository implements a research-oriented multi-timescale learning engine
for continual adaptation. It is inspired by the "nested learning" view where
models are treated as collections of coupled optimization problems operating at
different timescales, and by continuum memory systems with short-, medium-, and
long-term memories.

## Key Ideas

- **Multi-timescale updates**: Different parts of the model update at different
  frequencies (per-step, per-mini-batch, and rarely), enabling continual
  adaptation while preserving long-term knowledge.
- **Optimizer-as-memory**: The optimizer and recurrent state are reframed as
  explicit learnable memories. A learned momentum module predicts gradient
  trends and updates based on surprise (prediction error), rather than raw
  accumulation.
- **Continuum memory block**: A stack of feed-forward modules with distinct
  update schedules acts as short-, medium-, and long-term memory. We evaluate
  stability and sample-efficiency on streaming data tasks.

These ideas are structurally related to recent work on nested learning, multi-
timescale memory, and continual adaptation in deep and recurrent models.

## Structure

- `src/mtsl_engine/core`: Timescale scheduler, continuum memory, optimizer
  memory, surprise module.
- `src/mtsl_engine/models`: Recurrent core and feed-forward blocks.
- `src/mtsl_engine/data`: Streaming tasks (e.g., rotating Gaussians).
- `src/mtsl_engine/training`: Training loop and evaluation utilities.
- `src/mtsl_engine/viz`: Basic metric plotting.

## Quick Start

Install in editable mode:

```bash
pip install -e .
```

Train the engine on a streaming task:

```bash
mtsl train --input_dim 64 --hidden_dim 128 --max_steps 5000
```

Evaluate on a held-out streaming run:

```bash
mtsl eval --input_dim 64 --hidden_dim 128 --num_batches 100
```