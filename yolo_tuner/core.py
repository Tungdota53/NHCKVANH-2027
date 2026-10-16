"""Pure search logic. No CUDA, dataset downloads or fabricated accuracy models."""

from dataclasses import asdict, dataclass, replace
import math
import time


@dataclass(frozen=True)
class Config:
    model: str
    imgsz: int = 640
    batch: int = 1
    precision: str = "fp16"
    freeze: int = 0
    nbs: int = 64
    optimizer: str = "AdamW"
    lr0: float = 0.001

    @property
    def accumulation(self):
        # Ultralytics rounds nbs/batch after warmup; nbs is NOT an exact batch promise.
        return max(round(self.nbs / self.batch), 1)

    def payload(self):
        return {**asdict(self), "accumulation_after_warmup": self.accumulation,
                "effective_batch_after_warmup": self.batch * self.accumulation}


class Deadline:
    def __init__(self, seconds, clock=time.monotonic):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Search budget must be finite and positive")
        self.clock = clock
        self.start = clock()
        self.end = self.start + seconds

    @property
    def remaining(self):
        return max(0.0, self.end - self.clock())

    @property
    def elapsed(self):
        return self.clock() - self.start


def memory_batch_estimate(points, available_mb, max_batch=64, reserve_fraction=0.20):
    """Two-point proposal, never a certificate. The proposed batch must be measured."""
    if len(points) < 2:
        return points[-1][0] if points else 1
    (b1, v1), (b2, v2) = sorted(points)[-2:]
    limit = available_mb * (1 - reserve_fraction)
    if b2 == b1 or v2 <= v1:
        # A flat/noisy fit gives no evidence for an unbounded batch.
        return min(max_batch, b2)
    slope = (v2 - v1) / (b2 - b1)
    intercept = max(0.0, v1 - slope * b1)
    proposal = max(1, int((limit - intercept) / slope))
    # At most 2x beyond the last actual measurement.
    return min(max_batch, 2 * b2, proposal)


def accuracy_score(row):
    return row.get("accuracy_score", row["map50_proxy"])


def memory_score(row):
    return row.get("memory_mb",row["peak_reserved_mb"])


def dominates(a, b):
    va = (accuracy_score(a), -a["estimated_full_seconds"], -memory_score(a))
    vb = (accuracy_score(b), -b["estimated_full_seconds"], -memory_score(b))
    return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))


def pareto(rows):
    return [r for r in rows if not any(dominates(other, r) for other in rows)]


def choose_presets(rows, max_time=None):
    """Presets use equal-fidelity measured proxies, not final accuracy guarantees."""
    rows = [r for r in rows if max_time is None or r["estimated_full_seconds"] <= max_time]
    if not rows:
        return {}, []
    front = pareto(rows)
    names = ("accuracy_score", "estimated_full_seconds", "memory_mb")
    def value(row, key):
        return accuracy_score(row) if key == "accuracy_score" else memory_score(row) if key=="memory_mb" else row[key]
    minima = {k: min(value(r, k) for r in front) for k in names}
    maxima = {k: max(value(r, k) for r in front) for k in names}

    def balanced(row):
        costs = []
        for key in names:
            span = maxima[key] - minima[key]
            cost = 0.0 if span == 0 else (value(row, key) - minima[key]) / span
            costs.append(1 - cost if key == "accuracy_score" and span else cost)
        # Normalized Tchebycheff distance to the ideal, augmented to break ties.
        return max(costs) + 0.05 * sum(costs)

    return {
        "Accuracy": max(front, key=lambda r: (accuracy_score(r), -r["estimated_full_seconds"])),
        "Time": min(front, key=lambda r: r["estimated_full_seconds"]),
        "Memory": min(front, key=memory_score),
        "Balanced": min(front, key=balanced),
    }, front


def rank(rows):
    return sorted(rows, key=lambda r: (-r["map50_proxy"], r["estimated_full_seconds"], r["peak_reserved_mb"]))


def successive_halving(configs, rungs, evaluate, deadline, on_result=lambda r: None):
    """Only rank complete, equal-epoch cohorts. Interrupted rungs do not replace them.

    Each rung restarts from identical pretrained weights and the same full scheduler
    horizon. This costs extra compute, which is included in the wall-clock budget.
    """
    active = list(configs)
    last_cohort, history = [], []
    for epochs in rungs:
        cohort = []
        finished = True
        for config in active:
            if deadline.remaining < 1:
                finished = False
                break
            result = evaluate(config, epochs, deadline.remaining)
            result["requested_epochs"] = epochs
            history.append(result)
            on_result(result)
            if result["status"] == "ok" and result.get("epochs_completed") == epochs:
                cohort.append(result)
            elif result["status"] == "timeout":
                finished = False
                break
        if not finished:
            # First rung has no prior comparable cohort: expose completed subset.
            return last_cohort or cohort, history, False
        if not cohort:
            return last_cohort, history, True
        last_cohort = cohort
        keep = max(1, math.ceil(len(cohort) / 2))
        active = [Config(**r["config"]) for r in rank(cohort)[:keep]]
    return last_cohort, history, True


def training_finished(result, target_epochs):
    """An explicit completed early-stop is not the same as a timed-out partial run."""
    return (result.get("status") == "ok" and
            (result.get("epochs_completed") == target_epochs or
             result.get("training_completed") is True and result.get("early_stopped") is True
             and 0 < result.get("epochs_completed", 0) < target_epochs))


