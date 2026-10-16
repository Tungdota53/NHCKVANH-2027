"""Online cost estimates and explicitly scoped, diagnostic-only learned priors."""

import json
import math
from pathlib import Path
import statistics

from yolo_tuner.research import digest
from .model_features import descriptor_for, family_scale, model_identity

FEATURE_NAMES = ["intercept", "train_count", "val_count", "epochs", "image_area", "batch", "accumulation", "width", "hidden", "fp16", "sgd",
                 "model_family", "model_scale", "parameters", "layers", "gflops", "descriptor_known", "pretrained", "freeze", "bf16", "classes", "weight_decay", "log_lr", "dropout", "label_smoothing", "feature_grid",
                 "class_entropy", "objects_per_image", "small_box_fraction", "pixel_mean", "pixel_sd", "edge_mean"]


def feature_row(protocol, config):
    counts = protocol["example_counts"]
    descriptor = descriptor_for(protocol, config)
    family, scale = family_scale(config)
    data = protocol.get("dataset_descriptor", {})
    return [1.0, math.log1p(counts["train"]), math.log1p(counts["val"]), math.log1p(protocol["full_epochs"]),
            math.log1p(config["imgsz"] ** 2), math.log1p(config["batch"]), math.log1p(config["accumulation"]),
            math.log1p(config.get("width", 16)), math.log1p(config.get("hidden", 256)),
            float(config["precision"] == "fp16"), float(config["optimizer"] == "SGD"),
            float(family), float(scale), math.log1p(descriptor.get("parameters") or 0), math.log1p(descriptor.get("layers") or 0),
            math.log1p((descriptor.get("gflops_reference_32") or 0) * (config["imgsz"] / 32) ** 2), float(descriptor.get("parameters") is not None),
            float(descriptor.get("pretrained_checkpoint", str(config["model"]).endswith(".pt"))), float(config.get("freeze", 0)),
            float(config["precision"] == "bf16"), math.log1p(protocol.get("num_classes", 0)), math.log1p(config["weight_decay"]),
            math.log(config["lr0"]), config.get("dropout", 0), config.get("label_smoothing", 0), config.get("feature_grid", 0),
            data.get("class_entropy", 0), math.log1p(data.get("objects_per_image", 0)), data.get("small_box_fraction_area_lt_1pct_not_coco", 0),
            data.get("pixel_mean", 0), data.get("pixel_sd", 0), data.get("edge_mean", 0)]


def training_signature(protocol):
    # Controller/CLI/report edits do not change full-worker targets. Any worker,
    # runner, telemetry, model/metric or legacy dependency edit invalidates it.
    exact = {"research_pipeline/v2/worker.py", "research_pipeline/v2/runner.py", "research_pipeline/v2/pooling.py",
             "research_pipeline/v2/telemetry.py", "research_pipeline/v2/terminal.py", "research_pipeline/v2/model_features.py", "research_pipeline/v2/data_features.py", "research_pipeline/v2/predictors.py",
             "research_pipeline/v2/vision_worker.py", "research_pipeline/v2/caption_worker.py", "research_pipeline/v2/loaders.py",
             "research_pipeline/worker.py", "research_pipeline/runner.py", "research_pipeline/integrity.py"}
    sources = {k: v for k, v in protocol["source_files"].items() if k in exact or k.startswith("yolo_tuner/")}
    # Loader/cache policy changes full worker cost and RSS targets even with the
    # same source. Old cards/signatures without this field retain old semantics.
    if "execution_policy" in protocol:
        return digest({"sources": sources, "execution_policy": protocol["execution_policy"]})
    return digest(sources)


def compatibility(protocol, card):
    reasons = []
    if card.get("schema_version") != 2 or card.get("feature_names") != FEATURE_NAMES:
        reasons.append("legacy/incompatible predictor feature schema; refit from audited outcomes")
    if protocol["task"] != card["task"]:
        reasons.append("task unseen")
    hw = protocol["hardware"]
    hw_id = hw.get("gpu_uuid") if hw["device"] not in {"cpu", "mps"} else hw.get("host_label")
    if hw_id not in card["hardware_ids"]:
        reasons.append("physical hardware unseen")
    if protocol["runtime_versions"] not in card["runtimes"]:
        reasons.append("runtime unseen")
    compatible_training = card.get("training_source_signatures") and training_signature(protocol) in card["training_source_signatures"]
    if not compatible_training and protocol["implementation_version"] not in card["implementations"]:
        reasons.append("implementation unseen")
    if not compatible_training and card.get("source_signatures") and digest(protocol["source_files"]) not in card["source_signatures"]:
        reasons.append("source version unseen")
    return reasons


def point_prediction(protocol, config, card):
    reasons = compatibility(protocol, card)
    features = feature_row(protocol, config)
    if model_identity(config) not in card.get("model_identities", []):
        reasons.append("model architecture/checkpoint unseen")
    ranges = card.get("feature_ranges", [])
    if len(ranges) != len(features) or any(not lo <= x <= hi for x, (lo, hi) in zip(features, ranges)):
        reasons.append("input feature range/schema unseen")
    if reasons:
        return {"available": False, "reasons": reasons}
    values = {}
    for key in ("time", "memory", "quality"):
        coefficients = card.get(key + "_coefficients", [])
        if len(coefficients) != len(features) or not all(math.isfinite(v) for v in coefficients):
            return {"available": False, "reasons": ["invalid predictor coefficients"]}
        value = max(-30, min(20, sum(x * w for x, w in zip(features, coefficients))))
        values[key] = 1 / (1 + math.exp(-value)) if key == "quality" else math.exp(value)
    return {"available": True, **values, "reasons": [], "quality_measured": False, "upper_bound_guaranteed": False}


def observed_cross_cost(protocol, fulls, index):
    """Separate observed non-loop overhead from heuristic cross-config compute.

    Non-loop time also contains checkpoints/teardown, not just cold imports.
    Its future-width stability is not established by this point estimate.
    """
    configs = protocol["configs"]
    config = configs[index]

    def setting(c, ignored):
        return {k: v for k, v in c.items() if k not in ignored}

    def components(rows):
        valid = [(r["worker_wall_seconds"], r.get("measured_loop_seconds")) for r in rows]
        return [(wall, loop) for wall, loop in valid
                if isinstance(loop, (int, float)) and math.isfinite(loop) and 0 < loop <= wall]

    observed = {i: components(rows) for i, rows in fulls.items()}
    observed = {i: rows for i, rows in observed.items() if rows}
    if not observed:
        return None
    target_setting = setting(config, {"width", "imgsz", "lr0"})
    def distance(i):
        other = configs[i]
        return (setting(other, {"width", "imgsz", "lr0"}) != target_setting,
                abs(math.log(config["imgsz"] / other["imgsz"])) +
                abs(math.log(config.get("width", 16) / other.get("width", 16))), i)
    nearest = min(observed, key=distance)
    other = configs[nearest]
    reference_setting = setting(other, {"width", "lr0"})
    widths = {}
    for i, values in observed.items():
        candidate = configs[i]
        if setting(candidate, {"width", "lr0"}) == reference_setting:
            widths.setdefault(candidate.get("width", 16), []).extend(loop for _, loop in values)
    points = [(math.log(width), math.log(statistics.median(values))) for width, values in widths.items()]
    raw_exponent = None
    exponent = 2.0
    if len(points) >= 2:
        mean_x = statistics.mean(x for x, _ in points)
        mean_y = statistics.mean(y for _, y in points)
        variance = sum((x - mean_x) ** 2 for x, _ in points)
        if variance > 1e-12:
            raw_exponent = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
            exponent = max(0.0, min(2.0, raw_exponent))
    reference_loop = statistics.median(loop for _, loop in observed[nearest])
    overhead = max(wall - loop for wall, loop in observed[nearest])
    model_factor = 1 if config["model"] == other["model"] else 2
    scale = (config["imgsz"] / other["imgsz"]) ** 2 * (config.get("width", 16) / other.get("width", 16)) ** exponent * model_factor
    scaled_loop = reference_loop * max(0.5, scale)
    return {"point": overhead + scaled_loop,
            "basis": "observed non-loop overhead + cross-config loop scaling; heuristic point, no calibrated bounds",
            "cost_components": {"reference_index": nearest, "observed_non_loop_overhead_seconds": overhead,
                                "reference_loop_seconds": reference_loop, "scaled_loop_seconds": scaled_loop,
                                "future_non_loop_overhead_guaranteed": False},
            "width_scaling": {"exponent": exponent, "unclamped_exponent": raw_exponent,
                              "distinct_observed_widths": len(widths), "reference_imgsz": other["imgsz"],
                              "basis": "same-setting median-loop log slope" if raw_exponent is not None else "quadratic fallback; insufficient distinct widths",
                              "future_cost_guaranteed": False}}


class CostModel:
    def __init__(self, protocol):
        self.protocol, self.fulls, self.probes = protocol, {}, {}
        self.card = protocol.get("predictor")
        self.ood = compatibility(protocol, self.card) if self.card else ["no learned prior"]

    def update(self, index, result):
        if result.get("status") != "ok":
            return
        if result["mode"] == "train":
            self.fulls.setdefault(index, []).append(result)
        elif result.get("estimated_full_seconds", 0) > 0:
            self.probes.setdefault(index, []).append(result)

    def estimate(self, index):
        config = self.protocol["configs"][index]
        margin = self.protocol["scheduler"]["cost_margin"]
        memory, basis, predicted_quality = None, "unavailable", None
        candidate_ood = list(self.ood)
        cross_details = {}
        point = None
        if index in self.fulls:
            point = max(r["worker_wall_seconds"] for r in self.fulls[index])
            memory = max(r["memory_mb"] for r in self.fulls[index])
            basis = "same-config observed full workers; future-seed heuristic margin"
        elif index in self.probes:
            probes = self.probes[index]
            point = max(r["estimated_full_seconds"] + max(0, r["worker_wall_seconds"] - r.get("measured_loop_seconds", 0)) for r in probes)
            memory = max(r["memory_mb"] for r in probes)
            basis = "subset loop extrapolation + startup; not full memory evidence"
        elif self.card and not self.ood:
            prediction = point_prediction(self.protocol, config, self.card)
            if not prediction["available"]:
                candidate_ood.extend(prediction["reasons"])
                basis = "; ".join(prediction["reasons"]) + "; no usable prior"
            else:
                point, memory, predicted_quality = max(0.1, prediction["time"]), prediction["memory"], prediction["quality"]
                basis = "learned point prior; residual quantile diagnostic only, not a safety guarantee"
        if point is None and self.fulls:
            # Cross-config scaling is deliberately marked heuristic, including
            # architectures that cannot be represented by resolution alone.
            modern = self.protocol["scheduler"].get("policy_version", 1) >= 2
            observed = observed_cross_cost(self.protocol, self.fulls, index) if modern else None
            if observed:
                point, basis = observed["point"], observed["basis"]
                cross_details = {k: observed[k] for k in ("cost_components", "width_scaling")}
            else:
                rows = [(i, max(r["worker_wall_seconds"] for r in rs)) for i, rs in self.fulls.items()]
                nearest, time_s = min(rows, key=lambda p: abs(math.log(config["imgsz"] / self.protocol["configs"][p[0]]["imgsz"])))
                other = self.protocol["configs"][nearest]
                scale = (config["imgsz"] / other["imgsz"]) ** 2
                width = (config.get("width", 16) / other.get("width", 16)) ** 2
                model_factor = 1 if config["model"] == other["model"] else 2
                point = time_s * max(0.5, scale * width * model_factor)
                basis = "cross-config resolution/width heuristic; no calibrated bounds"
                if modern:
                    basis += "; loop telemetry unavailable, whole-worker fallback cannot separate non-loop overhead"
            if candidate_ood:
                basis += "; learned prior unavailable: " + "; ".join(candidate_ood)
        return {"point_seconds": point, "planning_seconds": point * margin if point else None,
                "predicted_memory_mb": memory, "basis": basis, "calibration": "uncalibrated_scheduling",
                "ood_reasons": list(dict.fromkeys(candidate_ood)), "upper_bound_guaranteed": False, "predicted_quality": predicted_quality,
                **cross_details}


def grouped_residuals(rows, predict, alpha=0.05):
    """One score per domain block; insufficient blocks => no finite quantile.

    This is a candidate conformal construction under exchangeable domain blocks,
    NOT an established adaptive/unseen-hardware coverage claim.
    """
    grouped = {}
    for row in rows:
        error = max(0.0, row["target"] - predict(row))
        grouped[row["domain"]] = max(grouped.get(row["domain"], 0.0), error)
    ordered = sorted(grouped.values())
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    q = ordered[rank - 1] if ordered and rank <= len(ordered) else None
    return {"domain_count": len(ordered), "alpha": alpha, "finite_sample_rank": rank,
            "one_sided_quantile": q, "status": "candidate_requires_exchangeable_blocks_and_validation" if q is not None else "insufficient_independent_calibration_domains",
            "coverage_guarantee": False}


def fit_card(manifest, output):
    """Fits only explicit fit domains; calibration/evaluation remain disjoint.

    Residual quantiles are diagnostics, NOT conformal intervals: trial rows in
    one dataset/GPU are dependent and acquisition is adaptive.
    """
    import numpy as np
    from .workflow import audit_run
    spec = json.loads(Path(manifest).read_text(encoding="utf-8"))
    partitions, identities = {}, {}
    for role in ("fit", "calibration", "evaluation"):
        partitions[role] = []
        for path in spec.get(role, []):
            folder, protocol, rows, _ = audit_run(path)
            hw = protocol["hardware"]
            hw_id = hw.get("gpu_uuid") if hw["device"] not in {"cpu", "mps"} else hw.get("host_label")
            domain = (protocol["dataset_content_id"], hw_id)
            if hw_id is None:
                raise ValueError("Cannot train a transferable card with unverified hardware identity")
            if domain in identities and identities[domain] != role:
                raise ValueError("Predictor domain leakage between fit/calibration/evaluation")
            identities[domain] = role
            for row in rows:
                if not row["complete"]:
                    continue
                from .schema import complete
                valid = {r["seed"]: r for r in row["runs"] if r["seed"] in protocol["training_seeds"] and complete(r, protocol["full_epochs"])}
                for r in valid.values():
                    if r["mode"] == "train" and r.get("status") == "ok":
                        partitions[role].append({"protocol": protocol, "features": feature_row(protocol, row["config"]),
                                                 "config": row["config"], "time": r["worker_wall_seconds"], "memory": r["memory_mb"], "quality": r["quality"], "domain": domain})
    fit = partitions["fit"]
    if len(fit) < 12:
        raise ValueError("Need at least 12 audited full outcomes in fit partition; do not fit proxy labels")
    task = fit[0]["protocol"]["task"]
    kind = fit[0]["protocol"]["memory_kind"]
    if any(r["protocol"]["task"] != task or r["protocol"]["memory_kind"] != kind for rs in partitions.values() for r in rs):
        raise ValueError("One predictor card per task and memory counter/backend; no pooled RSS/VRAM")
    x = np.asarray([r["features"] for r in fit])
    regularization = np.eye(x.shape[1]) * 0.1
    regularization[0, 0] = 0
    coefficients = {}
    for key in ("time", "memory", "quality"):
        values = np.asarray([r[key] for r in fit])
        y = np.log(np.clip(values, 1e-5, 1 - 1e-5) / (1 - np.clip(values, 1e-5, 1 - 1e-5))) if key == "quality" else np.log(np.maximum(1e-3, values))
        coefficients[key] = np.linalg.solve(x.T @ x + regularization, x.T @ y).tolist()
    def predict(r, key):
        z = max(-30, min(20, sum(x * w for x, w in zip(r["features"], coefficients[key]))))
        return 1 / (1 + math.exp(-z)) if key == "quality" else math.exp(z)
    def diagnostic(rs, key):
        errors = [r[key] - predict(r, key) for r in rs]
        positive = sorted(max(0, e) for e in errors)
        return {"n": len(rs), "domain_count": len({r["domain"] for r in rs}),
                "mae": statistics.mean(abs(e) for e in errors) if errors else None,
                "underprediction_rate": statistics.mean(e > 0 for e in errors) if errors else None,
                "p90_positive_residual": positive[min(len(positive) - 1, math.ceil(len(positive) * 0.9) - 1)] if positive else None}
    protocol = fit[0]["protocol"]
    calibration = {}
    heldout = {}
    for key in ("time", "memory", "quality"):
        def score(r, k=key):
            delta = r[k] - predict(r, k)
            return abs(delta) if k == "quality" else max(0.0, delta)
        calibration[key] = grouped_residuals([{**r, "target": score(r)} for r in partitions["calibration"]], lambda r: 0)
        calibration[key]["score_type"] = "absolute_quality_error" if key == "quality" else "positive_cost_underprediction"
        by_domain = {}
        for row in partitions["evaluation"]:
            by_domain[row["domain"]] = max(by_domain.get(row["domain"], 0), score(row))
        q = calibration[key]["one_sided_quantile"]
        heldout[key] = {"domain_count": len(by_domain), "observed_group_coverage": sum(e <= q for e in by_domain.values()) / len(by_domain) if q is not None and by_domain else None,
                        "coverage_established": False, "reason": "finite sample heldout diagnostic, not conditional/adaptive/unseen-hardware guarantee"}
    card = {"schema_version": 2, "feature_names": FEATURE_NAMES, "task": task, "memory_kind": kind, "time_coefficients": coefficients["time"],
            "memory_coefficients": coefficients["memory"], "partition_manifest": spec,
            "quality_coefficients": coefficients["quality"],
            "model_identities": sorted({model_identity(r["config"]) for r in fit}),
            "hardware_ids": sorted({r["domain"][1] for r in fit}),
            "runtimes": list({digest(r["protocol"]["runtime_versions"]): r["protocol"]["runtime_versions"] for r in fit}.values()),
            "implementations": sorted({r["protocol"]["implementation_version"] for r in fit}),
            "source_signatures": sorted({digest(r["protocol"]["source_files"]) for r in fit}),
            "training_source_signatures": sorted({training_signature(r["protocol"]) for r in fit}),
            "feature_ranges": [[float(lo), float(hi)] for lo, hi in zip(x.min(axis=0), x.max(axis=0))],
            "diagnostics": {role: {key: diagnostic(rs, key) for key in ("time", "memory", "quality")} for role, rs in partitions.items()},
            "grouped_calibration": calibration, "interval_evaluation": heldout,
            "calibration_status": "empirical_residual_diagnostic_not_validated_safety_interval",
            "coverage_guarantee": False, "unseen_gpu_validated": False,
            "limitations": ["No conditional/adaptive coverage guarantee.", "Fit is success-only; censored outcomes are not exact full-time labels.",
                            "Metadata does not establish unseen-dataset exchangeability; true out-domain validation is still required.",
                            "Quality point priors are ridge-logit baselines, not GP/qNEHVI or a substitute for full confirmation."]}
    card["sha256"] = digest(card)
    from yolo_tuner.pipeline import write_json
    write_json(output, card)
    return card
