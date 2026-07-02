"""
mtsl_engine: Multi-timescale learning engine for continual adaptation.

Concepts:
- Nested learning: multiple coupled optimization problems at different timescales.
- Optimizers as memory: recurrent / optimizer states as explicit learnable memories.
- Continuum memory systems: short-, medium-, and long-term memory modules with
  distinct update schedules.

This package provides:
- Timescale schedulers for per-step, per-batch, and rare updates.
- ContinuumMemory blocks that stack feed-forward modules with distinct update rates.
- OptimizerMemory modules where momentum and state are parameterized and learned.
- Training loops and streaming-data evaluations for continual adaptation tasks.
"""

__all__ = ["config", "core", "models", "data", "training", "viz"] 
