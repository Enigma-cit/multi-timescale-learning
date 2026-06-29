from dataclasses import dataclass


@dataclass
class EngineConfig:
    input_dim: int = 64
    hidden_dim: int = 128
    num_memory_blocks: int = 3  # short, medium, long
    short_update_every: int = 1
    medium_update_every: int = 16
    long_update_every: int = 256
    base_lr: float = 1e-3
    surprise_threshold: float = 0.1
    device: str = "cpu"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    max_steps: int = 10_000
    eval_interval: int = 500
    stream_task: str = "rotating_gaussians"