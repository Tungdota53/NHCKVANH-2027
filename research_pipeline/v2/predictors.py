"""Online cost estimates and explicitly scoped, diagnostic-only learned priors."""

import json
import math
from pathlib import Path
import statistics

from yolo_tuner.research import digest
from .model_features import descriptor_for, family_scale, model_identity, trainable_parameters_for

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
    # Controller/CLI/display/report/Web UI edits (cli.py, wizard.py, terminal.py,
    # report.py, paper_export.py, diagnostics.py, web_ui/) do not change full-worker
    # targets. Any worker, runner, telemetry, model/metric or legacy edit invalidates it.
    exact = {"research_pipeline/v2/worker.py", "research_pipeline/v2/runner.py", "research_pipeline/v2/pooling.py",
             "research_pipeline/v2/telemetry.py", "research_pipeline/v2/model_features.py", "research_pipeline/v2/data_features.py", "research_pipeline/v2/predictors.py",
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
    if ("detection_metric" in protocol or "detection_metric" in card) and protocol.get("detection_metric") != card.get("detection_metric"):
        reasons.append("detection_metric mismatch")
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


def _static_model_state_mb(protocol, config, observed_runs=None):
    total_p, train_p, param_basis = trainable_parameters_for(protocol, config, observed_runs)
    precision = config.get("precision", "fp32")
    optimizer = config.get("optimizer", "AdamW")
    is_amp = precision in {"fp16", "bf16"}
    weight_bytes = (6.0 if is_amp else 4.0) * total_p
    grad_bytes = 4.0 * train_p
    opt_bytes = (8.0 if optimizer == "AdamW" else 4.0) * train_p
    static_mb = (weight_bytes + grad_bytes + opt_bytes) / (1024.0 ** 2)
    return float(static_mb), int(total_p), int(train_p), param_basis


def _activation_proxy(protocol, config):
    """Activation proxy scaling with batch * imgsz^2 * model/layers, independent of accumulation."""
    batch = max(1, int(config.get("batch", 1) or 1))
    imgsz = max(1, int(config.get("imgsz", 32) or 32))
    precision = config.get("precision", "fp32")
    prec_factor = 2.0 if precision == "fp32" else 1.0
    descriptor = descriptor_for(protocol, config)
    task = protocol.get("task", "detection")
    if task == "detection":
        layers = max(1, int(descriptor.get("layers") or 24))
        freeze = max(0, int(config.get("freeze", 0) or 0))
        freeze_act_factor = max(0.25, 1.0 - 1.5 * (min(freeze, layers) / layers))
        flops = descriptor.get("gflops_reference_32")
        params = descriptor.get("parameters")
        _, scale = family_scale(config)
        if isinstance(flops, (int, float)) and not isinstance(flops, bool) and math.isfinite(flops) and flops > 0:
            model_act_factor = (float(flops) / 0.01653) ** 0.75 * (layers / 24.0) ** 0.25
        elif isinstance(params, (int, float)) and not isinstance(params, bool) and math.isfinite(params) and params > 0:
            model_act_factor = (float(params) / 2_600_000.0) ** 0.65 * (layers / 24.0) ** 0.25
        else:
            model_act_factor = max(1, scale or 1) ** 1.2
    else:
        width = max(4, int(config.get("width", 16) or 16))
        hidden = max(4, int(config.get("hidden", 256) or 256)) if task == "captioning" else 0
        model_act_factor = (width / 16.0) * (1.0 + 0.15 * hidden / 256.0)
        freeze_act_factor = 1.0
    return float(batch) * (float(imgsz) / 320.0) ** 2 * prec_factor * model_act_factor * freeze_act_factor


def observed_cross_memory(protocol, fulls, index):
    """Decompose observed memory into allocator base + static model state + activation memory."""
    configs = protocol["configs"]
    config = configs[index]
    obs_data = {}
    for i, rows in fulls.items():
        valid = [r for r in rows
                 if isinstance(r.get("memory_mb"), (int, float))
                 and not isinstance(r.get("memory_mb"), bool)
                 and math.isfinite(r["memory_mb"]) and r["memory_mb"] >= 0]
        if not valid:
            continue
        obs_mem_i = max(float(r["memory_mb"]) for r in valid)
        static_i, _, _, _ = _static_model_state_mb(protocol, configs[i], valid)
        act_raw_i = _activation_proxy(protocol, configs[i])
        rem_i = max(0.0, obs_mem_i - static_i)
        obs_data[i] = {"obs_mem": obs_mem_i, "static": static_i, "act_raw": act_raw_i, "rem": rem_i}
    if not obs_data:
        return None
    static_target, total_target, train_target, param_basis = _static_model_state_mb(protocol, config)
    act_raw_target = _activation_proxy(protocol, config)

    def mem_distance(i):
        other = configs[i]
        return (model_identity(other) != model_identity(config),
                other.get("precision") != config.get("precision"),
                abs(math.log(max(act_raw_target, 1e-12) / max(obs_data[i]["act_raw"], 1e-12))),
                i)

    nearest = min(obs_data, key=mem_distance)
    fit_mode = None
    base_mb = None
    act_mb = None
    if len(obs_data) >= 2:
        xs = [d["act_raw"] for d in obs_data.values()]
        ys = [d["rem"] for d in obs_data.values()]
        mean_x = statistics.mean(xs)
        mean_y = statistics.mean(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x > 1e-12:
            cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            k = cov_xy / var_x
            base = mean_y - k * mean_x
            if base < 0.0:
                base = 0.0
                denom = sum(x * x for x in xs)
                k = (sum(x * y for x, y in zip(xs, ys)) / denom) if denom > 1e-12 else 0.0
            if k > 0.0 and math.isfinite(k) and math.isfinite(base):
                base_mb = base
                act_mb = k * act_raw_target
                fit_mode = "2-param OLS allocator+activation decomposition"
    if fit_mode is None:
        rem_ref = obs_data[nearest]["rem"]
        act_raw_ref = obs_data[nearest]["act_raw"]
        is_cpu_or_mps = (protocol.get("hardware", {}).get("device") in {"cpu", "mps"}
                         or protocol.get("memory_kind") == "process_tree_rss_mib_not_vram")
        base_mb = (0.50 * rem_ref) if is_cpu_or_mps else min(0.40 * rem_ref, 200.0)
        act_ref = max(0.0, rem_ref - base_mb)
        act_mb = act_ref * (act_raw_target / max(act_raw_ref, 1e-12))
        fit_mode = "single-anchor allocator/activation prior decomposition"
    base_mb = max(0.0, float(base_mb))
    act_mb = max(0.0, float(act_mb))
    point_mb = max(0.0, base_mb + static_target + act_mb)
    return {
        "point_mb": float(point_mb),
        "base_allocator_mb": float(base_mb),
        "static_model_state_mb": float(static_target),
        "activation_mb": float(act_mb),
        "basis": f"cross-config physical memory ({fit_mode}; {param_basis}; activation independent of accumulation); heuristic point, no calibrated bounds",
        "upper_bound_guaranteed": False,
        "reference_index": nearest,
        "trainable_parameters": int(train_target),
        "total_parameters": int(total_target),
        "trainable_ratio": float(train_target / max(1, total_target)),
    }


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
                if not isinstance(wall, bool) and isinstance(wall, (int, float)) and math.isfinite(wall)
                and not isinstance(loop, bool) and isinstance(loop, (int, float)) and math.isfinite(loop)
                and 0 < loop <= wall]

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
    is_gpu = protocol.get("hardware", {}).get("device") not in {"cpu", "mps"}
    target_desc = descriptor_for(protocol, config)
    other_desc = descriptor_for(protocol, other)
    model_factor = 1.0
    if config["model"] != other["model"]:
        target_flops = target_desc.get("gflops_reference_32")
        other_flops = other_desc.get("gflops_reference_32")
        target_params = target_desc.get("parameters")
        other_params = other_desc.get("parameters")
        exp_model = 0.55 if is_gpu else 0.85
        if all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (target_flops, other_flops)):
            model_factor = max(0.35, min(6.0, (float(target_flops) / float(other_flops)) ** exp_model))
        elif all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (target_params, other_params)):
            model_factor = max(0.35, min(6.0, (float(target_params) / float(other_params)) ** exp_model))
        else:
            model_factor = 2.0
    batch_t = max(1, int(config.get("batch", 1) or 1))
    batch_o = max(1, int(other.get("batch", 1) or 1))
    batch_factor = 1.0 if batch_t == batch_o else max(0.75, min(1.35, (batch_o / batch_t) ** (0.25 if is_gpu else 0.10)))
    layers = max(1, int(target_desc.get("layers") or other_desc.get("layers") or 24))
    frz_t = int(config.get("freeze", 0) or 0)
    frz_o = int(other.get("freeze", 0) or 0)
    freeze_factor = 1.0 if frz_t == frz_o else max(0.65, min(1.40, (1.0 - 0.5 * min(frz_t, layers) / layers) / max(0.5, 1.0 - 0.5 * min(frz_o, layers) / layers)))
    prec_t = config.get("precision", "fp32")
    prec_o = other.get("precision", "fp32")
    if prec_t == prec_o:
        prec_factor = 1.0
    elif is_gpu and prec_t == "fp32" and prec_o in {"fp16", "bf16"}:
        prec_factor = 1.35
    elif is_gpu and prec_o == "fp32" and prec_t in {"fp16", "bf16"}:
        prec_factor = 0.75
    else:
        prec_factor = 1.0
    scale = (config["imgsz"] / other["imgsz"]) ** 2 * (config.get("width", 16) / other.get("width", 16)) ** exponent * model_factor * batch_factor * freeze_factor * prec_factor
    scaled_loop = reference_loop * max(0.35, scale)
    return {"point": overhead + scaled_loop,
            "basis": "observed non-loop overhead + cross-config loop scaling; heuristic point, no calibrated bounds",
            "cost_components": {"reference_index": nearest, "observed_non_loop_overhead_seconds": overhead,
                                "reference_loop_seconds": reference_loop, "scaled_loop_seconds": scaled_loop,
                                "batch_factor": batch_factor, "freeze_factor": freeze_factor, "precision_factor": prec_factor,
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
        self.online_loop_errors = []
        self.online_abs_errors = []
        self.online_memory_errors = []

    def update(self, index, result):
        if result.get("status") != "ok":
            return
        if result["mode"] == "train":
            prior = None
            if self.fulls or self.probes or (self.card and not self.ood):
                prior = self.estimate(index)
            elif isinstance(result.get("scheduling_prediction"), dict):
                prior = result["scheduling_prediction"]
            if prior:
                loop_pred = prior.get("loop_seconds")
                loop_actual = result.get("measured_loop_seconds")
                point_pred = prior.get("point_seconds")
                wall_actual = result.get("worker_wall_seconds")
                err = None
                if (isinstance(loop_pred, (int, float)) and math.isfinite(loop_pred) and loop_pred > 0
                        and isinstance(loop_actual, (int, float)) and math.isfinite(loop_actual) and loop_actual > 0):
                    err = (float(loop_actual) - float(loop_pred)) / float(loop_pred)
                elif (isinstance(point_pred, (int, float)) and math.isfinite(point_pred) and point_pred > 0
                        and isinstance(wall_actual, (int, float)) and math.isfinite(wall_actual) and wall_actual > 0):
                    err = (float(wall_actual) - float(point_pred)) / float(point_pred)
                if err is not None:
                    clamped = max(-0.5, min(1.0, err))
                    self.online_loop_errors.append(clamped)
                    self.online_abs_errors.append(abs(clamped))
                mem_pred = prior.get("predicted_memory_mb")
                mem_actual = result.get("memory_mb")
                if (isinstance(mem_pred, (int, float)) and not isinstance(mem_pred, bool) and math.isfinite(mem_pred) and mem_pred > 0
                        and isinstance(mem_actual, (int, float)) and not isinstance(mem_actual, bool) and math.isfinite(mem_actual) and mem_actual >= 0):
                    m_err = (float(mem_actual) - float(mem_pred)) / float(mem_pred)
                    self.online_memory_errors.append(max(-0.5, min(1.0, m_err)))
            self.fulls.setdefault(index, []).append(result)
        elif result.get("estimated_full_seconds", 0) > 0:
            self.probes.setdefault(index, []).append(result)

    def _margin_stats(self):
        base_margin = float(self.protocol["scheduler"]["cost_margin"])
        if not self.online_loop_errors:
            return base_margin, None
        n = len(self.online_loop_errors)
        online_error_ratio = statistics.mean(self.online_abs_errors)
        mean_pos_err = statistics.mean(max(0.0, e) for e in self.online_loop_errors)
        empirical_margin = max(min(base_margin, 1.03), min(max(base_margin, 1.35), 1.0 + mean_pos_err + 0.5 * online_error_ratio))
        weight = n / (n + 2.0)
        effective_margin = (1.0 - weight) * base_margin + weight * empirical_margin
        return float(effective_margin), float(online_error_ratio)

    def estimate(self, index):
        config = self.protocol["configs"][index]
        margin = float(self.protocol["scheduler"]["cost_margin"])
        policy_version = self.protocol["scheduler"].get("policy_version", 1)
        memory, memory_basis, memory_components = None, None, None
        basis, predicted_quality = "unavailable", None
        candidate_ood = list(self.ood)
        cross_details = {}
        point = None
        non_loop_s, loop_s = 0.0, 0.0
        if index in self.fulls:
            worst = max(self.fulls[index], key=lambda r: r["worker_wall_seconds"])
            point = float(worst["worker_wall_seconds"])
            measured_loop = worst.get("measured_loop_seconds")
            if isinstance(measured_loop, (int, float)) and not isinstance(measured_loop, bool) and math.isfinite(measured_loop) and 0 < measured_loop <= point:
                loop_s = float(measured_loop)
                non_loop_s = max(0.0, point - loop_s)
            else:
                non_loop_s = 0.0
                loop_s = point
            memory = max(r["memory_mb"] for r in self.fulls[index])
            memory_basis = "same-config observed full workers"
            basis = "same-config observed full workers; future-seed heuristic margin"
        elif index in self.probes:
            probes = self.probes[index]
            point = max(r["estimated_full_seconds"] + max(0, r["worker_wall_seconds"] - r.get("measured_loop_seconds", 0)) for r in probes)
            non_loop_s = min(point, max(max(0.0, r["worker_wall_seconds"] - r.get("measured_loop_seconds", r["worker_wall_seconds"])) for r in probes))
            loop_s = max(0.0, point - non_loop_s)
            memory = max(r["memory_mb"] for r in probes)
            memory_basis = "subset probe peak"
            basis = "subset loop extrapolation + startup; not full memory evidence"
        elif self.card and not self.ood:
            prediction = point_prediction(self.protocol, config, self.card)
            if not prediction["available"]:
                candidate_ood.extend(prediction["reasons"])
                basis = "; ".join(prediction["reasons"]) + "; no usable prior"
            else:
                point, memory, predicted_quality = max(0.1, prediction["time"]), prediction["memory"], prediction["quality"]
                non_loop_s, loop_s = 0.0, float(point)
                memory_basis = "learned point prior"
                basis = "learned point prior; residual quantile diagnostic only, not a safety guarantee"
        if point is None and self.fulls:
            # Cross-config scaling is deliberately marked heuristic, including
            # architectures that cannot be represented by resolution alone.
            modern = policy_version >= 2
            observed = observed_cross_cost(self.protocol, self.fulls, index) if modern else None
            if observed:
                point, basis = observed["point"], observed["basis"]
                non_loop_s = float(observed["cost_components"]["observed_non_loop_overhead_seconds"])
                loop_s = float(observed["cost_components"]["scaled_loop_seconds"])
                cross_details = {k: observed[k] for k in ("cost_components", "width_scaling")}
            else:
                rows = [(i, max(r["worker_wall_seconds"] for r in rs)) for i, rs in self.fulls.items()]
                nearest, time_s = min(rows, key=lambda p: abs(math.log(config["imgsz"] / self.protocol["configs"][p[0]]["imgsz"])))
                other = self.protocol["configs"][nearest]
                scale = (config["imgsz"] / other["imgsz"]) ** 2
                width = (config.get("width", 16) / other.get("width", 16)) ** 2
                model_factor = 1 if config["model"] == other["model"] else 2
                point = time_s * max(0.5, scale * width * model_factor)
                non_loop_s, loop_s = 0.0, float(point)
                basis = "cross-config resolution/width heuristic; no calibrated bounds"
                if modern:
                    basis += "; loop telemetry unavailable, whole-worker fallback cannot separate non-loop overhead"
            if candidate_ood:
                basis += "; learned prior unavailable: " + "; ".join(candidate_ood)
        if memory is None and self.fulls and policy_version >= 2:
            mem_obs = observed_cross_memory(self.protocol, self.fulls, index)
            if mem_obs:
                memory = mem_obs["point_mb"]
                memory_basis = mem_obs["basis"]
                memory_components = {k: mem_obs[k] for k in ("base_allocator_mb", "static_model_state_mb", "activation_mb", "reference_index", "trainable_parameters", "total_parameters", "trainable_ratio")}
        effective_margin, online_error_ratio = self._margin_stats()
        if point is None:
            planning_seconds = None
        elif policy_version >= 3:
            planning_seconds = non_loop_s + loop_s * effective_margin
        else:
            planning_seconds = point * margin
            effective_margin = margin
        return {"point_seconds": point, "planning_seconds": planning_seconds,
                "non_loop_seconds": non_loop_s if point is not None else None,
                "loop_seconds": loop_s if point is not None else None,
                "effective_margin": effective_margin, "online_error_ratio": online_error_ratio,
                "predicted_memory_mb": memory, "memory_basis": memory_basis,
                "memory_components": memory_components,
                "basis": basis, "calibration": "uncalibrated_scheduling",
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
    det_metric = fit[0]["protocol"].get("detection_metric")
    if any(r["protocol"]["task"] != task or r["protocol"]["memory_kind"] != kind for rs in partitions.values() for r in rs):
        raise ValueError("One predictor card per task and memory counter/backend; no pooled RSS/VRAM")
    if any(r["protocol"].get("detection_metric") != det_metric for rs in partitions.values() for r in rs):
        raise ValueError("Cannot pool different detection primary metrics in one predictor card")
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
    if det_metric is not None:
        card["detection_metric"] = det_metric
    card["sha256"] = digest(card)
    from yolo_tuner.pipeline import write_json
    write_json(output, card)
    return card
