"""Append-only hash-chained event ledger. Integrity, not anti-tamper attestation."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from yolo_tuner.research import digest


def read_events(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError("Ledger has an interrupted final record; preserve it, do not silently resume")
    events, previous = [], None
    for line in data.decode("utf-8").splitlines():
        event = json.loads(line)
        value = {k: v for k, v in event.items() if k != "sha256"}
        if event["sequence"] != len(events) or event["previous_sha256"] != previous or digest(value) != event["sha256"]:
            raise ValueError("Ledger checksum/sequence mismatch")
        if events and event["active_seconds"] < events[-1]["active_seconds"]:
            raise ValueError("Ledger clock went backwards")
        events.append(event)
        previous = event["sha256"]
    return events


class Ledger:
    def __init__(self, path, clock=time.monotonic, origin=None):
        self.path, self.clock = Path(path), clock
        self.events = read_events(self.path)
        self.prior = self.events[-1]["active_seconds"] if self.events else 0.0
        self.origin = clock() if origin is None else origin
        self.last = self.events[-1]["sha256"] if self.events else None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def elapsed(self):
        return self.prior + max(0, self.clock() - self.origin)

    def emit(self, kind, **payload):
        value = {"sequence": len(self.events), "previous_sha256": self.last,
                 "active_seconds": self.elapsed, "utc": datetime.now(timezone.utc).isoformat(),
                 "kind": kind, "payload": payload}
        event = {**value, "sha256": digest(value)}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(event)
        self.last = event["sha256"]
        return event

    @contextmanager
    def span(self, name, **payload):
        start = self.elapsed
        self.emit("span_start", name=name, **payload)
        try:
            yield
        except BaseException as exc:
            self.emit("span_end", name=name, seconds=self.elapsed - start, error=type(exc).__name__)
            raise
        else:
            self.emit("span_end", name=name, seconds=self.elapsed - start)


def worker_cost(events):
    # Parent results remain per-attempt; nested spans are never added to this.
    return sum(e["payload"]["worker_seconds"] for e in events if e["kind"] == "trial_completed")
