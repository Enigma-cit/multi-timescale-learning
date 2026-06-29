import torch
import torch.nn as nn
import torch.nn.functional as F

from mtsl_engine.config import EngineConfig, TrainingConfig
from mtsl_engine.core.timescale_scheduler import TimescaleScheduler
from mtsl_engine.core.continuum_memory import ContinuumMemory
from mtsl_engine.core.optimizer_memory import OptimizerMemory
from mtsl_engine.models.recurrent_core import RecurrentCore
from mtsl_engine.data.streaming_datasets import RotatingGaussiansStream
from mtsl_engine.core.surprise_module import prediction_surprise


class MultiTimescaleEngine(nn.Module):
    """
    Full engine coupling:

    - RecurrentCore (fast-timescale memory).
    - ContinuumMemory (short/medium/long modules).
    - OptimizerMemory (learned momentum / trend prediction).
    """

    def __init__(self, cfg: EngineConfig):
        super().__init__()
        self.cfg = cfg
        self.recurrent = RecurrentCore(cfg.input_dim, cfg.hidden_dim)
        self.continuum = ContinuumMemory(cfg.hidden_dim, cfg.hidden_dim, num_blocks=3)

        # OptimizerMemory over a flattened parameter vector
        param_dim = sum(p.numel() for p in self.parameters())
        self.opt_memory = OptimizerMemory(param_dim, hidden_dim=cfg.hidden_dim,
                                          surprise_threshold=cfg.surprise_threshold)

    def forward(self, x: torch.Tensor, h=None):
        # x: [B, T, input_dim]
        core_out, h_next = self.recurrent(x, h)
        # Take last time step representation
        last = core_out[:, -1, :]
        out = self.continuum(last)
        logits = nn.Linear(self.cfg.hidden_dim, 2).to(x.device)(out)  # simple head
        return logits, h_next

    def parameters_vector(self) -> torch.Tensor:
        # Flatten parameters into a single vector
        return torch.cat([p.view(-1) for p in self.parameters()])


def train_engine(cfg: EngineConfig, tcfg: TrainingConfig, device: str | None = None):
    if device is None:
        device = cfg.device

    engine = MultiTimescaleEngine(cfg).to(device)
    scheduler = TimescaleScheduler(
        short_every=cfg.short_update_every,
        medium_every=cfg.medium_update_every,
        long_every=cfg.long_update_every,
    )
    stream = RotatingGaussiansStream(dim=cfg.input_dim)

    # Standard optimizer just to apply updates; direction comes from OptimizerMemory
    base_opt = torch.optim.SGD(engine.parameters(), lr=cfg.base_lr)

    h = None
    metrics = {"loss": [], "acc": []}

    for step in range(tcfg.max_steps):
        x, y = stream.sample(tcfg.batch_size)
        x = x.to(device)
        y = y.to(device)

        # Treat batch as a sequence of length 1 for simplicity
        x_seq = x.unsqueeze(1)  # [B, 1, input_dim]
        logits, h = engine(x_seq, h)

        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        base_opt.zero_grad()
        loss.backward()

        # Flatten gradients
        grad_vec = torch.cat([p.grad.view(-1) for p in engine.parameters()])
        update_vec = engine.opt_memory.step(grad_vec)

        # Map update_vec back onto parameters
        offset = 0
        for p in engine.parameters():
            size = p.numel()
            p_update = update_vec[offset:offset + size].view_as(p)
            offset += size

            # Multi-timescale masking: only update certain parameter subsets at some steps
            # (sketch: here we apply all updates; you can add masks based on scheduler).
            p.data -= cfg.base_lr * p_update

        scheduler.advance()

        if step % tcfg.eval_interval == 0:
            metrics["loss"].append(float(loss.item()))
            metrics["acc"].append(float(acc.item()))
            print(f"step={step} loss={loss.item():.3f} acc={acc.item():.3f}")

    return engine, metrics