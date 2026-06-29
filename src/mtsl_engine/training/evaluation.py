import torch
from mtsl_engine.data.streaming_datasets import RotatingGaussiansStream


def evaluate_engine(engine, input_dim: int, num_batches: int = 100, batch_size: int = 64, device: str = "cpu"):
    stream = RotatingGaussiansStream(dim=input_dim)
    correct = 0
    total = 0
    h = None

    engine.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = stream.sample(batch_size)
            x = x.to(device)
            y = y.to(device)
            logits, h = engine(x.unsqueeze(1), h)
            preds = logits.argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += batch_size

    return correct / total