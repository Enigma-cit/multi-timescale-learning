from dataclasses import dataclass


@dataclass
class TimescaleScheduler:
    """
    Tracks step index and decides which memories to update.

    Example timescales:
      - step-level (fast)
      - mini-batch-level (medium)
      - rare (slow / structural)
    """
    short_every: int
    medium_every: int
    long_every: int

    step: int = 0

    def advance(self) -> None:
        self.step += 1

    def should_update_short(self) -> bool:
        return self.step % self.short_every == 0

    def should_update_medium(self) -> bool:
        return self.step % self.medium_every == 0

    def should_update_long(self) -> bool:
        return self.step % self.long_every == 0