"""Reference-only diagnostic analysis. Never called by online acquisition."""

import math

from .ledger import read_events
from .schema import complete


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, start = [0.0] * len(values), 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start + end - 1) / 2
        start = end
    return result


def spearman(a, b):
    if len(a) != len(b) or len(a) < 2:
        return None
    a, b = ranks(a), ranks(b)
    am, bm = sum(a) / len(a), sum(b) / len(b)
    denom = math.sqrt(sum((x - am) ** 2 for x in a) * sum((x - bm) ** 2 for x in b))
    return sum((x - am) * (y - bm) for x, y in zip(a, b)) / denom if denom else None


def first_full_seconds(folder, protocol, rows):
    good = {r["attempt_id"]: r for row in rows for r in row["runs"] if complete(r, protocol["full_epochs"]) and r.get("attempt_id")}
    seen = {}
    for event in read_events(folder / "events.jsonl"):
        if event["kind"] == "trial_completed" and event["payload"]["attempt_id"] in good:
            r = good[event["payload"]["attempt_id"]]
            seen.setdefault(r["index"], set()).add(r["seed"])
            if set(protocol["training_seeds"]) <= seen[r["index"]]:
                return event["active_seconds"]
    return None


def proxy_reference_diagnostics(state, protocol, reference):
    epochs = sorted({r.get("epochs_completed", 0) for r in state["probes"] if r.get("status") == "ok"})
    output = []
    for epoch in epochs:
        proxies = {r["index"]: r for r in state["probes"] if r.get("status") == "ok" and r.get("epochs_completed") == epoch and r.get("quality") is not None}
        paired = []
        for i, proxy in proxies.items():
            full = next((r for r in reversed(reference[i]["runs"]) if r["seed"] == proxy["seed"] and complete(r, protocol["full_epochs"])), None)
            if full:
                paired.append((proxy["quality"], full["quality"]))
        output.append({"subset_epochs": epoch, "paired_configs": len(paired),
                       "spearman_same_seed": spearman([q for q, _ in paired], [q for _, q in paired]),
                       "mae_same_seed": sum(abs(q - a) for q, a in paired) / len(paired) if paired else None,
                       "basis": "independent reference full-data/horizon, same seed; not online selection input"})
    return output
