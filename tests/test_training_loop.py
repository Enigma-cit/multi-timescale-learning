from mtsl_engine.config import EngineConfig, TrainingConfig
from mtsl_engine.training.loop import train_engine


def test_train_engine_runs_and_returns_metrics():
    cfg = EngineConfig(input_dim=4, hidden_dim=8, short_update_every=1,
                       medium_update_every=2, long_update_every=4)
    tcfg = TrainingConfig(batch_size=8, max_steps=50, eval_interval=10)

    engine, metrics = train_engine(cfg, tcfg)

    assert engine is not None
    assert "loss" in metrics and "acc" in metrics
    assert len(metrics["loss"]) > 0
    assert len(metrics["acc"]) > 0