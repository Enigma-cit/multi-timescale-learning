import torch
import torch.nn as nn
import torch.nn.functional as F

from mtsl_engine.config import EngineConfig, TrainingConfig
from mtsl_engine.core.timescale_scheduler import TimescaleScheduler
from mtsl_engine.core.continuum_memory import ContinuumMemory
from mtsl_engine.core.optimizer_memory import OptimizerMemory
from mtsl_engine.models.recurrent_core import RecurrentCore
from mtsl_engine.data.streaming_datasets import RotatingGaussiansStream


def _detach_hidden(h):
    if h is None:
        return None
    if isinstance(h, tuple):
        return tuple(v.detach() for v in h)
    if isinstance(h, list):
        return [v.detach() for v in h]
    return h.detach()


def _flatten_grads_from_params(params) -> torch.Tensor:
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.reshape(-1))
    return torch.cat(grads)


class MultiTimescaleEngine(nn.Module):
    """
    Coupled recurrent + continuum-memory engine with a registered classifier head.

    Important:
    - base-model parameters are updated by the learned optimizer-memory rule,
    - optimizer-memory parameters are NOT part of the target update vector.
    """

    def __init__(self, cfg: EngineConfig):
        super().__init__()
        self.cfg = cfg

        self.recurrent = RecurrentCore(cfg.input_dim, cfg.hidden_dim)
        self.continuum = ContinuumMemory(cfg.hidden_dim, cfg.hidden_dim, num_blocks=3)
        self.head = nn.Linear(cfg.hidden_dim, 2)

        # Define which parameters belong to the base model only.
        base_params = list(self.recurrent.parameters()) \
                    + list(self.continuum.parameters()) \
                    + list(self.head.parameters())

        self._base_param_shapes = [p.shape for p in base_params]
        self._base_param_numels = [p.numel() for p in base_params]
        self.base_param_dim = sum(self._base_param_numels)

        self.opt_memory = OptimizerMemory(
            param_dim=self.base_param_dim,
            hidden_dim=cfg.hidden_dim,
            surprise_threshold=cfg.surprise_threshold,
        )

    def base_parameters(self):
        yield from self.recurrent.parameters()
        yield from self.continuum.parameters()
        yield from self.head.parameters()

    def forward(self, x: torch.Tensor, h=None):
        core_out, h_next = self.recurrent(x, h)
        last = core_out[:, -1, :]
        out = self.continuum(last)
        logits = self.head(out)
        return logits, h_next


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

        h = _detach_hidden(h)

        x_seq = x.unsqueeze(1)
        logits, h = engine(x_seq, h)

        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        engine.zero_grad(set_to_none=True)
        loss.backward()

        base_params = list(engine.base_parameters())
        grad_vec = _flatten_grads_from_params(base_params)

        if grad_vec.numel() != engine.opt_memory.param_dim:
            raise RuntimeError(
                f"Gradient vector length {grad_vec.numel()} does not match "
                f"OptimizerMemory dimension {engine.opt_memory.param_dim}"
            )

        update_vec = engine.opt_memory.step(grad_vec).detach()

        offset = 0
        with torch.no_grad():
            for p in base_params:
                size = p.numel()
                p_update = update_vec[offset:offset + size].view_as(p)
                offset += size

                # Minimal stable rule for now; later you can use parameter groups
                if scheduler.should_update_short():
                    p -= cfg.base_lr * p_update

        scheduler.advance()

        if step % tcfg.eval_interval == 0:
            metrics["loss"].append(float(loss.item()))
            metrics["acc"].append(float(acc.item()))

    return engine, metrics
