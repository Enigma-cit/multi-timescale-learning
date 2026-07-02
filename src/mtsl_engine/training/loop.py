import torch
import torch.nn as nn
import torch.nn.functional as F

from mtsl_engine.config import EngineConfig, TrainingConfig
from mtsl_engine.core.timescale_scheduler import TimescaleScheduler
from mtsl_engine.core.continuum_memory import ContinuumMemory
from mtsl_engine.core.optimizer_memory import OptimizerMemory
from mtsl_engine.models.recurrent_core import RecurrentCore
from mtsl_engine.data.streaming_datasets import RotatingGaussiansStream


class MultiTimescaleEngine(nn.Module):
    """
    Full engine coupling:

    - RecurrentCore: fast-timescale memory
    - ContinuumMemory: short/medium/long memory stack
    - OptimizerMemory: learned update memory
    - Head: registered classifier head
    """

    def __init__(self, cfg: EngineConfig):
        super().__init__()
        self.cfg = cfg

        self.recurrent = RecurrentCore(cfg.input_dim, cfg.hidden_dim)
        self.continuum = ContinuumMemory(cfg.hidden_dim, cfg.hidden_dim, num_blocks=3)
        self.head = nn.Linear(cfg.hidden_dim, 2)

        param_dim = sum(p.numel() for p in self.parameters())
        self.opt_memory = OptimizerMemory(
            param_dim=param_dim,
            hidden_dim=cfg.hidden_dim,
            surprise_threshold=cfg.surprise_threshold,
        )

    def forward(self, x: torch.Tensor, h=None):
        core_out, h_next = self.recurrent(x, h)
        last = core_out[:, -1, :]
        out = self.continuum(last)
        logits = self.head(out)
        return logits, h_next


def _flatten_grads(module: nn.Module) -> torch.Tensor:
    grads = []
    for p in module.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)


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

    h = None
    metrics = {"loss": [], "acc": []}

    for step in range(tcfg.max_steps):
        x, y = stream.sample(tcfg.batch_size)
        x = x.to(device)
        y = y.to(device)

        x_seq = x.unsqueeze(1)
        logits, h = engine(x_seq, h)

        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        engine.zero_grad(set_to_none=True)
        loss.backward()

        grad_vec = _flatten_grads(engine)
        update_vec = engine.opt_memory.step(grad_vec).detach()

        offset = 0
        with torch.no_grad():
            for p in engine.parameters():
                size = p.numel()
                p_update = update_vec[offset:offset + size].view_as(p)
                offset += size

                # You can later add parameter-group masks here based on scheduler.
                p -= cfg.base_lr * p_update

        if h is not None:
            h = h.detach()

        scheduler.advance()

        if step % tcfg.eval_interval == 0:
            metrics["loss"].append(float(loss.item()))
            metrics["acc"].append(float(acc.item()))

    return engine, metrics
