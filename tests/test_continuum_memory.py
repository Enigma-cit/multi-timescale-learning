import torch
from mtsl_engine.core.continuum_memory import ContinuumMemory


def test_continuum_memory_shape_preservation():
    cm = ContinuumMemory(input_dim=8, hidden_dim=16, num_blocks=3)
    x = torch.randn(4, 8)  # [B, D]
    out = cm(x)
    assert out.shape == x.shape


def test_continuum_memory_timescale_indices():
    cm = ContinuumMemory(input_dim=8, hidden_dim=16, num_blocks=3)
    idx = cm.timescale_indices()
    assert set(idx.keys()) == {"short", "medium", "long"}
    assert idx["short"] == 0
    assert idx["medium"] == 1
    assert idx["long"] == 2