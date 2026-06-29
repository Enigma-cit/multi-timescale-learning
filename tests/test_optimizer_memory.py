import torch
from mtsl_engine.core.optimizer_memory import OptimizerMemory


def test_optimizer_memory_step_updates_momentum():
    param_dim = 10
    om = OptimizerMemory(param_dim=param_dim, hidden_dim=16, surprise_threshold=0.1)

    grad = torch.ones(param_dim)
    update = om.step(grad)

    # Update direction should have correct shape
    assert update.shape == grad.shape

    # Momentum should have been updated away from all-zeros
    assert torch.norm(om.momentum) > 0.0


def test_optimizer_memory_high_surprise_changes_momentum_more():
    param_dim = 5
    om = OptimizerMemory(param_dim=param_dim, hidden_dim=8, surprise_threshold=0.01)

    grad_small = torch.zeros(param_dim)
    grad_large = torch.ones(param_dim) * 10.0

    om.step(grad_small)
    m_after_small = om.momentum.clone()

    om.step(grad_large)
    m_after_large = om.momentum.clone()

    # Large surprise should yield larger momentum norm
    assert torch.norm(m_after_large) > torch.norm(m_after_small)