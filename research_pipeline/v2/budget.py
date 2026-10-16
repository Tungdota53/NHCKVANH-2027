"""Observed finalization reserve; a heuristic, not a hard-deadline proof."""

from pathlib import Path
import time

from research_pipeline.integrity import sha


class FinalizationReserve:
    def __init__(self, cleanup, boundary_seconds, clock=time.monotonic):
        self.cleanup, self.boundary_seconds, self.clock = cleanup, boundary_seconds, clock
        self.attempts = 0
        self.checkpoint_bytes = 0
        self.bulk_rates = []

    def observe(self, result):
        self.attempts += 1
        path = result.get("best_checkpoint")
        expected = result.get("checkpoint_sha256")
        if not path or not expected or not Path(path).is_file():
            return
        path = Path(path)
        before = self.clock()
        if sha(path) != expected:
            raise ValueError("Checkpoint changed before finalization-cost calibration")
        seconds = max(self.clock() - before, 0.001)
        size = path.stat().st_size
        self.checkpoint_bytes += size
        if size:
            # Sparse dataset stat/open overhead is not bulk-checkpoint bandwidth.
            # Measure actual checkpoint reads instead of reserving tens of
            # seconds by applying a tiny-PNG bytes/sec rate to large files.
            self.bulk_rates.append(size / seconds)

    def seconds(self):
        rate = min(self.bulk_rates, default=1024 ** 2)
        rate = max(1024 ** 2, min(128 * 1024 ** 2, rate))
        return max(self.cleanup, 2 * self.boundary_seconds + 1 + 0.08 * self.attempts
                   + 2 * self.checkpoint_bytes / rate)

    def summary(self):
        return {"reserved_seconds": self.seconds(), "checkpoint_bytes": self.checkpoint_bytes,
                "checkpoint_hash_rate_samples": len(self.bulk_rates),
                "basis": "full-boundary verification + observed bulk checkpoint hashing + per-attempt metadata margin",
                "upper_bound_guaranteed": False}
