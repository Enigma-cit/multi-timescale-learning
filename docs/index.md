# Multi-Timescale Learning Engine

This documentation describes the components of `mtsl_engine`, a Python package
for multi-timescale learning and continual adaptation. 

## Modules

- `core.timescale_scheduler`: Defines update schedules at multiple timescales.
- `core.continuum_memory`: Implements continuum memory blocks.
- `core.optimizer_memory`: Defines optimizer-as-memory modules.
- `models.recurrent_core`: Fast-timescale recurrent memory core.
- `data.streaming_datasets`: Streaming tasks for continual evaluation.
- `training.loop`: End-to-end training loop for streaming data.
- `training.evaluation`: Evaluation utilities.
