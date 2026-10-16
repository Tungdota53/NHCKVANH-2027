"""Pre-registered finite-domain HPO benchmark with independently measured oracle.

Search decisions are locked before full-reference outcomes exist. The exhaustive
reference is offline evaluation cost, never a hidden part of online search budget.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace

from . import cli_ui
from .core import Deadline, training_finished


@dataclass(frozen=True)
class Objective:
    name: str = "balanced"
    min_quality: float = 0.0
    max_time: float | None = None
    max_memory: float | None = None
    time_scale: float = 600.0
    memory_scale: float = 1024.0
    weights: tuple = (0.6, 0.25, 0.15)

    def __post_init__(self):
        if self.name not in {"quality", "time", "memory", "balanced"}:
            raise ValueError("Unknown optimization objective")
        if not 0 <= self.min_quality <= 1:
            raise ValueError("min-quality must be in [0,1]")
        for value in (self.time_scale, self.memory_scale, self.max_time, self.max_memory):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Scales and resource limits must be finite positive values")
        if len(self.weights)!=3 or any(not math.isfinite(w) or w < 0 for w in self.weights) or sum(self.weights)<=0:
            raise ValueError("Three nonnegative weights with positive sum are required")

    def feasible(self, row):
        return (row.get("complete", True) and row["quality_min"] >= self.min_quality
                and (self.max_time is None or row["time_max"] <= self.max_time)
                and (self.max_memory is None or row["memory_max"] <= self.max_memory))

    def value(self, row):
        if self.name == "quality":
            return row["quality_mean"]
        if self.name == "time":
            return -row["time_mean"]
        if self.name == "memory":
            return -row["memory_max"]
        w = [x/sum(self.weights) for x in self.weights]
        return w[0]*row["quality_mean"] - w[1]*row["time_mean"]/self.time_scale - w[2]*row["memory_max"]/self.memory_scale


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode("utf-8")).hexdigest()


def measured_quality(result):
    # Caption target is best validation checkpoint within a fixed horizon.
    # Other compact tasks retain their explicitly defined final-epoch metric.
    return result.get("best_quality",result.get("quality_score"))


def aggregate(config_id, runs, expected_seeds):
    valid=[r for r in runs if r.get("full_completed")]
    if not valid:
        return {"config_id":config_id,"complete":False,"runs":runs,"quality_mean":0.0,"quality_min":0.0,
                "time_mean":None,"time_max":None,"memory_max":None}
    q=[measured_quality(r) for r in valid]
    t=[r["worker_wall_seconds"] for r in valid]
    m=[r["memory_mb"] for r in valid]
    complete=len(valid)==len(expected_seeds) and {r["training_seed"] for r in valid}==set(expected_seeds)
    row={"config_id":config_id,"complete":complete,"runs":runs,"quality_mean":statistics.mean(q),
            "quality_sd":statistics.stdev(q) if len(q)>1 else 0.0,"quality_min":min(q),"quality_max":max(q),
            "time_mean":statistics.mean(t),"time_sd":statistics.stdev(t) if len(t)>1 else 0.0,
            "time_max":max(t),"memory_max":max(m),"n_seeds":len(valid)}
    best_metrics=[]
    for r in valid:
        epoch=next((point for point in r.get("learning_curve",[]) if point["epoch"]==r.get("best_epoch")),{})
        if "bleu4" in epoch and "chrf" in epoch:
            best_metrics.append(epoch)
    if len(best_metrics)==len(valid):
        for key in ("bleu4","chrf"):
            values=[point[key] for point in best_metrics]
            row[key+"_mean"]=statistics.mean(values)
            row[key+"_sd"]=statistics.stdev(values) if len(values)>1 else 0.0
    return row


def nondominated(rows):
    def dominates(a,b):
        av=(a["quality_mean"],-a["time_mean"],-a["memory_max"])
        bv=(b["quality_mean"],-b["time_mean"],-b["memory_max"])
        return all(x>=y for x,y in zip(av,bv)) and any(x>y for x,y in zip(av,bv))
    return [r for r in rows if not any(dominates(o,r) for o in rows)]


def hypervolume(rows, objective):
    """Exact union volume of 3D origin boxes with fixed, pre-registered anchors."""
    boxes=[(max(0.0,min(1.0,r["quality_mean"])),max(0.0,1-r["time_mean"]/objective.time_scale),
            max(0.0,1-r["memory_max"]/objective.memory_scale)) for r in rows]
    volume=0.0; previous_x=0.0
    for x in sorted({b[0] for b in boxes}):
        active=[b for b in boxes if b[0]>=x]
        area=0.0; previous_y=0.0
        for y in sorted({b[1] for b in active}):
            z=max((b[2] for b in active if b[1]>=y),default=0.0)
            area+=(y-previous_y)*z; previous_y=y
        volume+=(x-previous_x)*area; previous_x=x
    return volume


def proxy_row(observation):
    return {"quality_mean":observation["quality"],"quality_min":observation["quality"],
            "time_mean":observation["estimated_time"],"time_max":observation["estimated_time"],
            "memory_max":observation["memory"],"config_id":observation["config_id"],"complete":True}


def run_search(method, configs, evaluate, objective, seconds, rungs, full_epochs, seed, clock=time.monotonic):
    """evaluate(id,epochs,remaining,full_data) cannot access reference outcomes."""
    deadline=Deadline(seconds,clock)
    history=[]; cohort=[]; confirmations=[]
    rng=random.Random(seed)
    order=list(range(len(configs))); rng.shuffle(order)

    def observe(index,epochs,full=False):
        if deadline.remaining < 0.1:
            return None
        result=evaluate(index,epochs,deadline.remaining,full)
        quality=measured_quality(result)
        row={"config_id":f"c{index:02d}","epochs":epochs,"full_data":full,"status":result.get("status"),
             "quality":quality,"estimated_time":result.get("estimated_full_seconds"),
             "memory":result.get("memory_mb"),"worker_seconds":result.get("worker_wall_seconds"),
             "complete":result.get("status")=="ok" and result.get("epochs_completed")==epochs,
             "result":result}
        if full and row["complete"]:
            row["estimated_time"]=result["worker_wall_seconds"]
        history.append(row)
        return row

    def successful(rows):
        # Do not reject early quality as a certificate of final-quality failure.
        return [r for r in rows if r and r["complete"] and r["quality"] is not None
                and (objective.max_memory is None or r["memory"]<=objective.max_memory)]

    def score(row):
        return objective.value(proxy_row(row))

    if method=="default":
        return {"method":method,"seed":seed,"selected":"c00","finalists":["c00"],"history":[],
                "search_seconds":0.0,"budget_seconds":seconds,"selection_basis":"predeclared default; no reference consulted"}
    if method=="adaptive_verified":
        # Profile the SAME highest proxy fidelity for every candidate. Early
        # quality is diagnostic only; never eliminate slow starters by it.
        probes=[]
        for index in order:
            row=observe(index,max(rungs))
            if row and row["complete"]:
                probes.append(row)
            if row is None or row["status"]=="timeout":
                break
        estimates={r["config_id"]:r["estimated_time"] for r in successful(probes)}
        projected=sum(estimates.values())*1.25+2*len(configs)
        affordable=len(estimates)==len(configs) and projected<=deadline.remaining
        # Cover every LR stratum before revisiting it. This guard is fixed before
        # the experiment; it does not use any previous dataset's best config.
        groups={}
        for index in order:
            groups.setdefault(configs[index].get("lr0"),[]).append(index)
        schedule=[]
        while any(groups.values()):
            for group in groups.values():
                if group:
                    schedule.append(group.pop(0))
        if affordable:
            schedule=order
        for index in schedule:
            estimated=estimates.get(f"c{index:02d}")
            if estimated is not None and deadline.remaining < estimated*1.25+2:
                # Not a proof of infeasibility: retain it as unmeasured.
                continue
            row=observe(index,full_epochs,True)
            if row and row["complete"]:
                confirmations.append(row)
            if row is None or row["status"]=="timeout":
                break
        cohort=probes
    elif method in {"random","random_proxy"}:
        # Separate matched-fidelity control from the full-fidelity random baseline.
        for index in order:
            row=observe(index,full_epochs if method=="random" else max(rungs),method=="random")
            if row and row["complete"]:
                cohort.append(row)
            elif row and row["status"]=="timeout":
                break
    elif method in {"successive_halving","pareto_racing"}:
        active=order
        for rung in rungs:
            wave=[]; complete=True
            for index in active:
                row=observe(index,rung)
                if row is None or row["status"]=="timeout":
                    complete=False; break
                wave.append(row)
            good=successful(wave)
            if not complete:
                cohort=cohort or good
                break
            if not good:
                break
            cohort=good
            if method=="successive_halving":
                promoted=sorted(good,key=lambda r:r["quality"],reverse=True)[:max(1,math.ceil(len(good)/2))]
            else:
                front=nondominated([proxy_row(r) for r in good])
                ids={r["config_id"] for r in front}
                promoted=sorted([r for r in good if r["config_id"] in ids],key=score,reverse=True)
                # Keep a runner-up to reduce premature elimination; all extra work
                # is charged to the same wall-clock cap, not an unreported budget.
                keep=min(len(good),max(2,math.ceil(len(good)/2)))
                promoted=(promoted+[r for r in sorted(good,key=score,reverse=True) if r not in promoted])[:keep]
            active=[int(r["config_id"][1:]) for r in promoted]
        if method=="pareto_racing":
            for row in sorted(cohort,key=score,reverse=True):
                # Extrapolation is only a scheduling proposal. Leave a startup margin.
                if deadline.remaining < row["estimated_time"]*1.1+2:
                    continue
                confirmation=observe(int(row["config_id"][1:]),full_epochs,True)
                if confirmation and confirmation["complete"]:
                    confirmations.append(confirmation)
                if confirmation and confirmation["status"]=="timeout":
                    break
    else:
        raise ValueError("Unknown search baseline")
    candidates=successful(confirmations if method=="adaptive_verified" else confirmations or cohort)
    predicted_feasible=[r for r in candidates if objective.feasible(proxy_row(r))]
    selected=max(predicted_feasible,key=score)["config_id"] if predicted_feasible else None
    front=nondominated([proxy_row(r) for r in candidates])
    decision={"method":method,"seed":seed,"selected":selected,"finalists":[r["config_id"] for r in front],
            "history":history,"search_seconds":deadline.elapsed,"budget_seconds":seconds,
            "budget_overrun_seconds":max(0,deadline.elapsed-seconds),
            "selection_basis":("full-data confirmation" if candidates else "no completed full-data trial") if confirmations or method in {"random","adaptive_verified"} else "equal-fidelity subset proxy; final feasibility unverified"}
    if method=="adaptive_verified":
        decision.update(exhaustive_full_data=len({r["config_id"] for r in confirmations})==len(configs),
                        projected_sweep_seconds=projected,projection_is_bound=False,
                        strategy="measured_affordable_full_sweep" if affordable else "lr_stratified_full_confirmation",
                        unverified_configs=[f"c{i:02d}" for i in range(len(configs)) if not any(r["config_id"]==f"c{i:02d}" for r in confirmations)])
    return decision


def score_decisions(decisions, reference, objective):
    feasible=[r for r in reference if objective.feasible(r)]
    lookup={r["config_id"]:r for r in reference}
    best=max(feasible,key=objective.value) if feasible else None
    oracle_hv=hypervolume(feasible,objective)
    scores=[]
    for decision in decisions:
        chosen=lookup.get(decision["selected"])
        valid=chosen is not None and objective.feasible(chosen)
        recovered=[lookup[c] for c in decision["finalists"] if c in lookup and objective.feasible(lookup[c])]
        quality_opt=max((r["quality_mean"] for r in feasible),default=None)
        scores.append({"method":decision["method"],"search_seed":decision["seed"],"selected":decision["selected"],
            "feasible":valid,"objective_regret":max(0.0,objective.value(best)-objective.value(chosen)) if best and valid else None,
            "quality_regret":max(0.0,quality_opt-chosen["quality_mean"]) if valid and quality_opt is not None else None,
            "quality_mean":chosen["quality_mean"] if valid else None,"time_mean":chosen["time_mean"] if valid else None,
            "memory_max":chosen["memory_max"] if valid else None,"search_seconds":decision["search_seconds"],
            "search_over_full_time":decision["search_seconds"]/chosen["time_mean"] if valid else None,
            "recovered_hypervolume_ratio":hypervolume(recovered,objective)/oracle_hv if oracle_hv else None,
            "oom_count":sum(r["status"]=="oom" for r in decision["history"]),
            "trials":len(decision["history"]),"budget_overrun_seconds":decision.get("budget_overrun_seconds",0.0)})
        errors=[]
        for observation in decision["history"]:
            truth=lookup.get(observation["config_id"])
            matched=next((r for r in truth.get("runs",[]) if r.get("training_seed")==decision["seed"] and r.get("full_completed")),None) if truth else None
            if observation.get("complete") and not observation.get("full_data") and matched:
                errors.append({"q":abs(observation["quality"]-measured_quality(matched)),
                    "t":abs(observation["estimated_time"]-matched["worker_wall_seconds"])/matched["worker_wall_seconds"],
                    "m":abs(observation["memory"]-matched["memory_mb"])/matched["memory_mb"]})
        scores[-1].update(proxy_quality_mae=statistics.mean(e["q"] for e in errors) if errors else None,
                          time_mape=statistics.mean(e["t"] for e in errors) if errors else None,
                          memory_relative_error=statistics.mean(e["m"] for e in errors) if errors else None)
    summaries=[]
    for method in dict.fromkeys(d["method"] for d in decisions):
        rows=[r for r in scores if r["method"]==method]
        summary={"method":method,"repeats":len(rows),"feasibility_rate":sum(r["feasible"] for r in rows)/len(rows),
                 "selected_configs":dict((c,sum(r["selected"]==c for r in rows)) for c in {r["selected"] for r in rows}),
                 "oom_count":sum(r["oom_count"] for r in rows)}
        for key in ("objective_regret","quality_regret","search_seconds","search_over_full_time","recovered_hypervolume_ratio",
                    "proxy_quality_mae","time_mape","memory_relative_error"):
            values=[r[key] for r in rows if r[key] is not None]
            summary[key+"_mean"]=statistics.mean(values) if values else None
            summary[key+"_sd"]=statistics.stdev(values) if len(values)>1 else 0.0 if values else None
        summaries.append(summary)
    return best, scores, summaries


def show(args):
    folder=Path(args.run).resolve()
    protocol=json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
    status=json.loads((folder/"status.json").read_text(encoding="utf-8"))
    cli_ui.heading("NGHIÊN CỨU CẤU HÌNH TỐI ƯU", "Trạng thái và dữ liệu đã đo; không khởi động training")
    print(f"  Phase: {status['phase']} • {len(protocol['configs'])} cấu hình × {len(protocol['training_seeds'])} seed × {protocol['full_epochs']} epoch")
    print(f"  Objective: {protocol['objective']['name']} • Search budget cap: {protocol['search_budget_seconds']} giây/lần")
    print(f"  Q: {protocol['quality_definition']}")
    cli_ui._table("KHÔNG GIAN ĐÃ KHÓA",["ID","Model","Ảnh","Batch","Width","Hidden","LR"],
        [[f"c{i:02d}",c["model"],c["imgsz"],c["batch"],c["width"],c["hidden"],c["lr0"]] for i,c in enumerate(protocol["configs"])])
    reference_file=folder/"reference.partial.json"
    if reference_file.exists():
        data=json.loads(reference_file.read_text(encoding="utf-8"))
        attempted=sum(len(rows) for rows in data.values())
        completed=sum(r.get("full_completed",False) for rows in data.values() for r in rows)
        print(f"  Full-reference hoàn tất: {completed}/{len(protocol['configs'])*len(protocol['training_seeds'])}; đã thử: {attempted}")
    decisions_file=folder/"decisions.locked.json"
    print(f"  Quyết định tìm kiếm đã khóa: {'Có' if decisions_file.exists() else 'Chưa'}")
    partial=folder/"decisions.partial.json"
    if partial.exists() and not decisions_file.exists():
        done=json.loads(partial.read_text(encoding="utf-8"))
        print(f"  Quyết định tìm kiếm đã lưu: {len(done)}/{len(protocol['methods'])*len(protocol['training_seeds'])}")
    if status["phase"] not in {"complete","aborted_pilot"}:
        import psutil
        live=[]
        for process_file in folder.rglob("process.json"):
            process=json.loads(process_file.read_text(encoding="utf-8"))
            if process.get("status")!="running" or not process.get("create_time"):
                continue
            try:
                active=psutil.Process(process["pid"])
                if abs(active.create_time()-process["create_time"])<0.01:
                    live.append((process_file,process))
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                pass
        if live:
            current,process=live[-1]
            request=json.loads((current.parent/"request.json").read_text(encoding="utf-8"))
            result_path=current.parent/"result.json"
            progress=json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            print(f"  Worker đang chạy (PID/create-time đã khớp): seed {request['seed']}, epoch {progress.get('epochs_completed',0)}/{request['trial_epochs']}")
            print(f"  Log: {current.parent/'worker.log'}")
        else:
            cli_ui.warning("Không thấy worker còn sống; phase đã lưu không xác nhận process cha còn chạy (có thể đang chuyển trial).")
    report_path=folder/"study.audited.json"
    if not report_path.exists():
        report_path=folder/"study.json"
    if report_path.exists():
        report=json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("mode")=="online_verification":
            print(f"  Verification status: {report['verification_status']} • Offline oracle cost: 0; mọi full confirmation tính vào online")
        cli_ui._table("SO SÁNH CÙNG NGÂN SÁCH",["Method","Feasible","Regret mean","Search s"],
            [[r["method"],f"{r['feasibility_rate']:.0%}",f"{r['objective_regret_mean']:.6f}" if r["objective_regret_mean"] is not None else "—",f"{r['search_seconds_mean']:.1f}"] for r in report["summaries"]])
        print(f"  Reference: {report['reference_status']} • Empirical best: {report['empirical_best']}")
        cli_ui.artifact("Báo cáo",folder/("RESEARCH_AUDITED.md" if report_path.name=="study.audited.json" else "RESEARCH.md"))
    else:
        cli_ui.warning("Chưa có full-reference đầy đủ; chưa kết luận cấu hình tối ưu.")
    cli_ui.artifact("Protocol",folder/"protocol.json")
    return 0


def verify_protocol(protocol):
    """Check the frozen artifact; hashes detect edits, not trusted timestamps."""
    unsigned={k:v for k,v in protocol.items() if k!="protocol_sha256"}
    if digest(unsigned)!=protocol["protocol_sha256"]:
        raise ValueError("Protocol hash mismatch; do not interpret this run as preregistered")


def verify_source(protocol):
    for name,checksum in protocol.get("source_sha256",{}).items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=checksum:
            raise ValueError(f"Recorded source changed during study: {name}; do not mix implementations")


def verify_dataset_files(manifest):
    from .adapters import file_hash
    for records in manifest["splits"].values():
        for record in records:
            if record.get("sha256") and file_hash(Path(record["image"]))!=record["sha256"]:
                raise ValueError(f"Dataset image changed after discovery: {record['image']}; do not certify this protocol")


def audit(args):
    """Recompute diagnostics from saved measurements without new training/search."""
    from .pipeline import write_json
    folder=Path(args.run).resolve()
    original=json.loads((folder/"study.json").read_text(encoding="utf-8"))
    protocol=json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
    verify_protocol(protocol)
    verify_dataset_files(json.loads((folder/"dataset.json").read_text(encoding="utf-8")))
    decisions=json.loads((folder/"decisions.locked.json").read_text(encoding="utf-8"))
    if digest(decisions)!=original["decisions_sha256"] or original["protocol"]!=protocol or original["decisions"]!=decisions:
        raise ValueError("Saved report does not match locked decisions/protocol")
    results=json.loads((folder/"reference.partial.json").read_text(encoding="utf-8"))
    checked=0
    for index,config in enumerate(protocol["configs"]):
        seen=set()
        for row in results[f"c{index:02d}"]:
            seed=row["training_seed"]
            if seed not in protocol["training_seeds"] or seed in seen:
                raise ValueError("Unexpected/duplicate reference seed")
            seen.add(seed)
            trial=Path(row["log"]).parent
            request=json.loads((trial/"request.json").read_text(encoding="utf-8"))
            raw=json.loads((trial/"result.json").read_text(encoding="utf-8"))
            if (request["config"]!=config or request["seed"]!=seed or request["trial_epochs"]!=protocol["full_epochs"]
                    or Path(request["manifest"]).resolve()!=folder/"dataset.json" or request["mode"]!="train"):
                raise ValueError("Reference request differs from frozen protocol")
            if row.get("full_completed") and (raw["status"]!="ok" or raw.get("epochs_completed")!=protocol["full_epochs"] or raw["config"]!=config
                    or measured_quality(raw)!=measured_quality(row) or raw.get("memory_mb")!=row.get("memory_mb")):
                raise ValueError("Full-reference measurement differs from raw worker result")
            checked+=1
    reference=[aggregate(key,results[key],protocol["training_seeds"]) for key in results]
    objective=Objective(**protocol["objective"])
    best,scores,summaries=score_decisions(decisions,reference,objective)
    complete=all(r["complete"] for r in reference)
    online=original.get("mode")=="online_verification"
    if not complete and not online:
        raise ValueError("Audit requires a complete reference; no partial optimum is certified")
    if online:
        scores=[]; summaries=[]
    report={**original,"reference":reference,"scores":scores,"summaries":summaries,
            "empirical_best":best["config_id"] if best else None,
            "audit":{"verified_reference_requests":checked,"protocol_sha256":protocol["protocol_sha256"],
                     "decisions_sha256":digest(decisions),"new_training_performed":False}}
    report["limitations"]=list(original["limitations"])
    note="Full-fidelity random can be censored by a tight budget; no universal superiority follows without budget/fidelity ablations."
    if note not in report["limitations"]:
        report["limitations"].append(note)
    write_json(folder/"study.audited.json",report)
    (folder/"RESEARCH_AUDITED.md").write_text(markdown(report),encoding="utf-8")
    cli_ui.success(f"Đã đối chiếu {checked} lượt full-reference với request/result gốc; không chạy training mới.")
    cli_ui.artifact("Báo cáo kiểm chứng",folder/"RESEARCH_AUDITED.md")
    return 0


def deployment_scope_changes(selection, config, manifest, hardware):
    """An empirical optimum is conditional; never carry it to another protocol."""
    if not selection.get("research_provenance"):
        return []
    report=json.loads(Path(selection["research_provenance"]).read_text(encoding="utf-8"))
    protocol=report["protocol"]
    verify_protocol(protocol)
    verified=report.get("mode")=="online_verification"
    if (not verified and not report.get("reference_complete")) or not report.get("empirical_best"):
        raise ValueError("Research selection has no complete feasible reference; no optimality claim can be deployed")
    if verified:
        chosen=next((r for r in report["reference"] if r["config_id"]==report["empirical_best"]),None)
        if not chosen or not chosen["complete"]:
            raise ValueError("Online selection has not completed every required full-training seed")
    changes=[]
    if selection["full_epochs"]!=protocol["full_epochs"]:
        changes.append("epoch horizon")
    if selection["seed"] not in protocol["training_seeds"]:
        changes.append("training seed")
    if digest(manifest)!=protocol["dataset_sha256"]:
        changes.append("dataset/split manifest")
    expected=protocol["configs"][int(report["empirical_best"][1:])]
    # Relocating an identical feature cache is not a training hyperparameter.
    if {k:v for k,v in config.items() if k!="feature_cache"}!={k:v for k,v in expected.items() if k!="feature_cache"}:
        changes.append("training configuration")
    for key in ("device","gpu_uuid","gpu","cpu","torch_version","torch_cuda_version","driver_version"):
        if hardware.get(key)!=protocol["hardware"].get(key):
            changes.append(f"hardware/runtime: {key}")
    from importlib.metadata import version
    import platform
    for name,expected_version in protocol.get("runtime_versions",{}).items():
        current=platform.python_version() if name=="python" else platform.platform() if name=="os" else version(name)
        if current!=expected_version:
            changes.append(f"runtime: {name}")
    for name,checksum in protocol.get("source_sha256",{}).items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=checksum:
            changes.append(f"source: {name}")
    return changes


def default_space(task, hw, manifest, epochs, preset="compact"):
    from .pipeline import candidate_configs
    base=candidate_configs(task,hw,manifest,1)[0]
    if task=="captioning":
        base["batch"]=8 if hw["device"]=="cpu" else 32
        variants=[{}, {"hidden":384}, {"lr0":0.0003}, {"hidden":128,"batch":16}]
    elif task=="classification":
        base.update(imgsz=32,batch=16,effective_batch=32)
        variants=[{}, {"width":32}, {"lr0":0.0003}, {"width":32,"lr0":0.0003}]
    else:
        variants=[{}, {"width":32}, {"lr0":0.0003}, {"width":32,"lr0":0.0003}]
    if preset=="expanded":
        if task=="captioning":
            # Same declared dimensions on CPU/CUDA, not a 3050-specific batch.
            base.update(batch=8,width=32,imgsz=160,effective_batch=64)
            variants=[{"hidden":h,"lr0":lr} for lr in (0.001,0.0003) for h in (128,256,384)]
        else:
            base.update(imgsz=32,batch=16,effective_batch=32,width=16,hidden=128)
            variants=[{"width":w,"lr0":lr} for lr in (0.001,0.0003,0.003) for w in (16,32)]
    elif preset!="compact":
        raise ValueError("Unknown space preset")
    return [{**base,**v,"epochs":epochs,"patience":epochs+1,"min_delta":0.0,"deterministic":True} for v in variants]


def markdown(report):
    protocol=report["protocol"]; lines=["# Thực nghiệm xác định cấu hình tối ưu", "",
        f"Trạng thái đối chứng: **{report['reference_status']}**.",
        f"Task: {protocol['task']}. Objective: {protocol['objective']['name']}. Hardware: {protocol['hardware'].get('gpu') or 'CPU'}.",
        f"Không gian: {len(protocol['configs'])} cấu hình; seeds {protocol['training_seeds']}; full horizon {protocol['full_epochs']} epoch.",
        f"Search budget mỗi phương pháp/lần: {protocol['search_budget_seconds']} giây. "+
        ("Full verification nằm trong online budget, không có oracle độc lập." if report.get("mode")=="online_verification" else "Tất cả quyết định khóa trước full-reference."), "",
        "## Full-reference: toàn bộ train/val, không dùng proxy", "",
        "| Config | Q mean ± SD | Time mean ± SD (s) | Peak memory max (MiB) | Seeds đủ |", "|---|---:|---:|---:|---|"]
    for row in report["reference"]:
        if row["time_mean"] is None:
            lines.append(f"| {row['config_id']} | chưa đo đủ | — | — | False |")
        else:
            lines.append(f"| {row['config_id']} | {row['quality_mean']:.6f} ± {row['quality_sd']:.6f} | {row['time_mean']:.2f} ± {row['time_sd']:.2f} | {row['memory_max']:.1f} | {row['complete']} |")
    lines += ["", "## So sánh phương pháp: cùng budget cap, báo cả thời gian thực dùng", "",
        "| Method | Feasibility | Regret mean | Search seconds mean | Search/full | Recovered HV |", "|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.6f}"
    for row in report["summaries"]:
        lines.append(f"| {row['method']} | {row['feasibility_rate']:.2%} | {fmt(row['objective_regret_mean'])} | {fmt(row['search_seconds_mean'])} | {fmt(row['search_over_full_time_mean'])} | {fmt(row['recovered_hypervolume_ratio_mean'])} |")
    label="Cấu hình tối ưu thực nghiệm trong tập đã khai báo" if report["reference_complete"] else "Cấu hình tốt nhất đã xác nhận; chưa chứng nhận toàn không gian"
    lines += ["", f"{label}: {report.get('empirical_best') or 'chưa xác định'}.", "",
        "Objective balanced = wQ*Q - wT*T/time_scale - wM*M/memory_scale; weights và anchors được khóa trước, không fit theo oracle.",
        "Feasibility: min Q trên seeds đạt ngưỡng; max time/memory trên seeds không vượt giới hạn. Đây không phải khoảng tin cậy cho seed/máy tương lai.",
        "Hypervolume dùng cùng điểm tham chiếu cố định và full outcomes của tập finalist mà phương pháp trả về; không phải HV proxy.", "",
        "## Sai số proxy so với full-reference cùng seed", "",
        "| Method | Quality MAE | Time MAPE | Memory relative error |", "|---|---:|---:|---:|"]
    for row in report["summaries"]:
        lines.append(f"| {row['method']} | {fmt(row.get('proxy_quality_mae_mean'))} | {fmt(row.get('time_mape_mean'))} | {fmt(row.get('memory_relative_error_mean'))} |")
    lines += ["", "Sai số chỉ tính cho trial proxy hoàn tất có reference cùng seed; MAPE là tỷ lệ (0.1 = 10%), không phải khoảng tin cậy.",
        f"Chi phí chuẩn bị chung: {protocol['common_setup_seconds']:.2f}s. Chi phí full-reference offline: {report['offline_reference_seconds']:.2f}s.",
        ("selected.json dùng full confirmations đo mới tại máy/dataset này; mọi chi phí đã tính vào online. Không có đối chứng độc lập để kết luận superiority." if report.get("mode")=="online_verification" else
         "selected.json là nghiệm từ full-reference (tốn offline exhaustive cost), không phải kết quả online được tìm thấy miễn phí trong search budget."), "",
        "## Giới hạn", ""] + [f"- {x}" for x in report["limitations"]]
    return "\n".join(lines)+"\n"


def study(args):
    from .adapters import discover, subset
    from .pipeline import new_session, write_json, runtime_info, language_policy, VisionRunner, prepare_caption_backend
    if len(set(args.seeds))!=len(args.seeds) or not args.seeds:
        raise ValueError("Training/search seeds must be distinct")
    methods=getattr(args,"methods",["default","random","successive_halving","pareto_racing"])
    if not methods or len(set(methods))!=len(methods):
        raise ValueError("Study methods must be nonempty and distinct")
    if getattr(args,"command",None)=="verify-config" and methods!=["adaptive_verified"]:
        raise ValueError("verify-config requires only adaptive_verified; use research for comparing methods")
    if args.full_epochs<max(args.rungs) or args.rungs!=sorted(set(args.rungs)) or min(args.rungs)<1:
        raise ValueError("Rungs must increase and fit full-epochs")
    if any(not math.isfinite(v) or v<=0 for v in (args.train_subset,args.val_subset,args.trial_timeout,args.search_budget,args.timeout_hours)):
        raise ValueError("Sizes, budgets and timeout must be positive")
    Objective(args.objective,args.min_quality,args.max_time,args.max_memory,args.time_scale,args.memory_scale,tuple(args.weights))
    folder=new_session(args.output); overall=Deadline(args.timeout_hours*3600)
    import os
    import psutil
    write_json(folder/"study_process.json",{"pid":os.getpid(),"create_time":psutil.Process().create_time()})
    cli_ui.heading("NGHIÊN CỨU CẤU HÌNH TỐI ƯU", "Khóa protocol • so sánh cùng budget • full-reference độc lập • regret/Pareto")
    manifest=discover(args.data,args.split_seed,args.task)
    if manifest["task"] not in {"captioning","classification","reconstruction"}:
        raise ValueError("Research runner currently supports compact caption/classification/reconstruction; native detection requires its own full-reference adapter")
    if not manifest["validation_ready"]:
        raise ValueError("Research needs independent train/validation")
    hw=runtime_info(args.device)
    if not hw["ml_runtime_ready"]:
        raise ValueError("Research needs an actual ML runtime; no synthetic reference is produced")
    cli_ui.hardware_summary(hw); cli_ui.dataset_summary(manifest)
    write_json(folder/"dataset.json",manifest)
    sampled=subset(manifest,args.train_subset,args.val_subset)
    write_json(folder/"subset.json",sampled)
    vocab,maximum=language_policy(manifest)
    configs=default_space(manifest["task"],hw,manifest,args.full_epochs,getattr(args,"space","compact"))
    if args.grid:
        custom=json.loads(Path(args.grid).read_text(encoding="utf-8"))
        if not isinstance(custom,list) or not custom or any(not isinstance(c,dict) for c in custom):
            raise ValueError("grid must be a nonempty JSON list of configuration overrides")
        allowed={"batch","effective_batch","width","lr0","precision"}
        if manifest["task"]=="captioning":
            allowed.update({"hidden","dropout","label_smoothing","weight_decay"})
        else:
            allowed.add("imgsz")
        if any(set(c)-allowed for c in custom):
            raise ValueError("Grid changes model/encoder/protocol or unknown keys; only numeric training hyperparameters and precision are supported")
        configs=[{**configs[0],**c,"epochs":args.full_epochs,"patience":args.full_epochs+1,"min_delta":0.0,"deterministic":True} for c in custom]
    if manifest["task"]=="captioning":
        prep=SimpleNamespace(caption_encoder=args.caption_encoder,offline=args.offline,plan_only=False)
        backend=prepare_caption_backend(prep,folder,hw,overall,folder/"dataset.json")
        if args.grid and backend.get("encoder")=="resnet18" and any("width" in c for c in custom):
            raise ValueError("width is inactive with the frozen ResNet18 encoder; vary hidden/lr/batch instead")
        configs=[{**c,**backend} for c in configs]
    if getattr(args,"precision",None):
        configs=[{**c,"precision":args.precision} for c in configs]
    for c in configs:
        if any(type(c[k]) is not int or c[k]<1 for k in ("batch","effective_batch","width","hidden","imgsz")):
            raise ValueError("Training dimensions and batch sizes must be positive integers")
        if c["optimizer"]!="AdamW" or not math.isfinite(c["lr0"]) or c["lr0"]<=0:
            raise ValueError("Research space currently requires valid AdamW compact configurations")
        if c["effective_batch"]%c["batch"] or c["width"]%4 or c["hidden"]%4:
            raise ValueError("Effective batch must be divisible by batch; widths must be multiples of 4")
        if c["precision"] not in {"fp32","fp16","bf16"} or (c["precision"]!="fp32" and not hw.get(c["precision"],False)):
            raise ValueError("Requested research precision is not supported by the measured hardware")
        for key in ("dropout","label_smoothing"):
            if key in c and (not math.isfinite(c[key]) or not 0<=c[key]<1):
                raise ValueError(f"{key} must be in [0,1)")
        if "weight_decay" in c and (not math.isfinite(c["weight_decay"]) or c["weight_decay"]<0):
            raise ValueError("weight_decay must be finite and nonnegative")
        if hw["device"] in {"cpu","mps"} and c["precision"]!="fp32":
            raise ValueError("CPU/MPS research must use fp32")
    memory_cap=args.max_memory if args.max_memory is not None else (hw.get("vram_free_mb") or hw["ram_available_mb"])*0.8
    objective=Objective(args.objective,args.min_quality,args.max_time,memory_cap,args.time_scale,args.memory_scale,tuple(args.weights))
    setup=overall.elapsed
    protocol={"schema_version":1,"task":manifest["task"],"dataset_source":manifest["source"],
        "dataset_sha256":digest(manifest),"hardware":hw,"configs":configs,"training_seeds":args.seeds,
        "split_seed":args.split_seed,"full_epochs":args.full_epochs,"rungs":args.rungs,
        "search_budget_seconds":args.search_budget*60,"train_subset_images":sampled["counts"]["train"],
        "val_subset_images":sampled["counts"]["val"],"objective":asdict(objective),
        "quality_definition":"best validation (BLEU4+chrF)/200 checkpoint across fixed horizon" if manifest["task"]=="captioning" else "final epoch validation quality_score",
        "methods":methods,"common_setup_seconds":setup,
        "oracle_not_available_to_search":True,"test_history":"Not certified untouched: prior engineering evaluations may exist; this study does not use test for search/reference selection.",
        "reference_training_policy":"all full train/val, identical fixed epoch horizon; early stopping disabled; LR scheduler unchanged",
        "memory_definition":"peak PyTorch reserved allocator MiB on CUDA; process RSS MiB on CPU/MPS; not total machine/GPU usage",
        "dataset_hash_definition":"canonical discovered manifest including per-image SHA256 recorded at discovery; input bytes verified before final publication",
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
        "scope":"empirical optimum over exactly these configurations and training seeds on this recorded hardware/protocol"}
    protocol["mode"]="online_verification" if getattr(args,"command",None)=="verify-config" else "independent_research"
    protocol["space_preset"]=getattr(args,"space","compact")
    protocol["method_development_history"]="Previous ViECAP/CIFAR results informed the late-starter guard; this is not a blind new claim on those datasets."
    from . import __version__
    from importlib.metadata import version
    import platform
    protocol["implementation_version"]=__version__
    protocol["runtime_versions"]={name:version(name) for name in ("torch","torchvision","sacrebleu","Pillow","psutil","PyYAML")}
    protocol["runtime_versions"].update(python=platform.python_version(),os=platform.platform())
    snapshot=folder/"source_snapshot"; snapshot.mkdir()
    for name in protocol["source_sha256"]:
        (snapshot/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    protocol["protocol_sha256"]=digest(protocol)
    write_json(folder/"protocol.json",protocol)
    write_json(folder/"status.json",{"phase":"search","protocol":protocol["protocol_sha256"]})
    decisions=[]
    # Rotate method order between repeats to reduce systematic warmup/thermal bias.
    methods=[m for m in protocol["methods"] if m!="default"]
    for repeat,seed in enumerate(args.seeds):
        if "default" in protocol["methods"]:
            decisions.append(run_search("default",configs,None,objective,args.search_budget*60,args.rungs,args.full_epochs,seed))
        offset=repeat%len(methods) if methods else 0
        ordered=methods[offset:]+methods[:offset]
        for method in ordered:
            if overall.remaining<1:
                raise ValueError("Study timeout during searches; partial artifacts preserved")
            print(f"\nSEARCH {method} • seed {seed} • budget {args.search_budget*60:.0f}s",flush=True)
            def evaluate(index,epochs,remaining,full):
                verify_source(protocol)
                trial_folder=folder/"search"/method/f"seed_{seed}"
                data=manifest if full else sampled
                runner=VisionRunner(trial_folder,folder/("dataset.json" if full else "subset.json"),data,manifest,hw,args.full_epochs,vocab,maximum,seed)
                # Each call has a unique folder while keeping the seed/protocol fixed.
                runner.index=len(list(trial_folder.glob("trial_*")))
                result=runner.run(configs[index],epochs,min(remaining,overall.remaining,args.trial_timeout),"train" if full else "search")
                cli_ui.trial(result)
                return result
            ordering_seed=args.split_seed if protocol["mode"]=="online_verification" else seed
            decision=run_search(method,configs,evaluate,objective,min(args.search_budget*60,overall.remaining),args.rungs,args.full_epochs,ordering_seed)
            decision.update(seed=seed,ordering_seed=ordering_seed)
            decisions.append(decision)
            write_json(folder/"decisions.partial.json",decisions)
    # This immutable artifact is the boundary between online decisions and oracle.
    write_json(folder/"decisions.locked.json",decisions)
    lock_hash=digest(decisions)
    write_json(folder/"status.json",{"phase":"full_reference","decisions_sha256":lock_hash})
    if protocol["mode"]=="online_verification":
        from .verification import publish_online
        return publish_online(folder,protocol,decisions,vocab,maximum,manifest,overall.elapsed)
    results={f"c{i:02d}":[] for i in range(len(configs))}
    jobs=[(i,s) for i in range(len(configs)) for s in args.seeds]
    random.Random(args.split_seed+1000).shuffle(jobs)
    for index,seed in jobs:
        if overall.remaining<1:
            break
        verify_source(protocol)
        print(f"\nFULL REFERENCE c{index:02d} • seed {seed} • {args.full_epochs} epoch • full dataset",flush=True)
        runner=VisionRunner(folder/"reference"/f"c{index:02d}"/f"seed_{seed}",folder/"dataset.json",manifest,manifest,hw,args.full_epochs,vocab,maximum,seed)
        result=runner.run(configs[index],args.full_epochs,min(args.trial_timeout,overall.remaining),"train")
        result.update(training_seed=seed,full_completed=training_finished(result,args.full_epochs) and result.get("epochs_completed")==args.full_epochs)
        results[f"c{index:02d}"].append(result); cli_ui.trial(result)
        write_json(folder/"reference.partial.json",results)
    return publish(folder,protocol,decisions,results,vocab,maximum,manifest,overall.elapsed,lock_hash)


def publish(folder,protocol,decisions,results,vocab,maximum,manifest,elapsed,lock_hash):
    from .pipeline import write_json
    seeds=protocol["training_seeds"]; epochs=protocol["full_epochs"]
    configs=protocol["configs"]; hw=protocol["hardware"]
    objective=Objective(**protocol["objective"])
    verify_source(protocol)
    verify_dataset_files(manifest)
    reference=[aggregate(key,rows,seeds) for key,rows in results.items()]
    complete=all(r["complete"] for r in reference)
    if digest(json.loads((folder/"decisions.locked.json").read_text(encoding="utf-8")))!=lock_hash:
        raise ValueError("Locked search decisions changed; study integrity violated")
    best,scores,summaries=score_decisions(decisions,reference,objective)
    if not complete:
        # No fake zero regret or oracle hypervolume when reference is incomplete.
        for row in scores:
            row["objective_regret"]=row["quality_regret"]=row["recovered_hypervolume_ratio"]=None
        for row in summaries:
            for key in ("objective_regret","quality_regret","recovered_hypervolume_ratio"):
                row[key+"_mean"]=row[key+"_sd"]=None
    report={"schema_version":1,"protocol":protocol,"decisions":decisions,"decisions_sha256":lock_hash,
        "reference":reference,"reference_complete":complete,
        "reference_status":"complete_finite_empirical_reference" if complete else "partial_reference_not_certified",
        "empirical_best":best["config_id"] if best and complete else None,"scores":scores,"summaries":summaries,
        "pareto":[r["config_id"] for r in nondominated([r for r in reference if objective.feasible(r)])],
        "elapsed_seconds":elapsed,"offline_reference_seconds":sum(reference_cost(r) for rows in results.values() for r in rows),
        "limitations":["Finite configurations and seeds, not global optimum or an optimality guarantee for unseen datasets/hardware.",
            "Repeated seeds quantify observed variation, not calibrated future accuracy/time/memory confidence intervals.",
            "Equal budget caps do not imply identical used compute; actual search cost and common preparation are reported separately.",
            "Search decisions were locked before reference labels; prior engineering test exposure is disclosed, so a fresh outer test is needed for publication-level generalization.",
            "No claim of new state-of-the-art or superiority is made when measured regret/cost does not support it."]}
    report["limitations"].append("Full-fidelity random search may complete no trial under a tight cap. This compares operational feasibility at this cap, not universal algorithm superiority; vary budgets and add matched-fidelity ablations before stronger claims.")
    write_json(folder/"study.json",report)
    (folder/"RESEARCH.md").write_text(markdown(report),encoding="utf-8")
    if best and complete:
        index=int(best["config_id"][1:])
        selection={"schema_version":2,"backend":"compact_vision","task":manifest["task"],
            "status":"empirical_optimal_in_declared_space","config":configs[index],"device":hw["device"],
            "manifest":str(folder/"dataset.json"),"full_epochs":epochs,"seed":seeds[0],
            "vocabulary":vocab,"max_tokens":maximum,"names":manifest["names"],"trainable":True,
            "selection_path":str(folder/"selected.json"),"research_provenance":str(folder/"study.json"),
            "protocol_sha256":protocol["protocol_sha256"],"optimality_scope":protocol["scope"],
            "full_reference_measurement":{k:v for k,v in best.items() if k!="runs"},"final_accuracy_verified":False}
        write_json(folder/"selected.json",selection)
    elif (folder/"selected.json").exists():
        # Preserve the old selection for recovery, but do not leave it certified
        # after a resumed/edited reference becomes incomplete or infeasible.
        previous=json.loads((folder/"selected.json").read_text(encoding="utf-8"))
        write_json(folder/"selected.previous.json",previous)
        write_json(folder/"selected.json",{**previous,"status":"reference_not_certified","trainable":False})
    write_json(folder/"status.json",{"phase":"complete","reference_complete":complete,"empirical_best":report["empirical_best"]})
    cli_ui._table("SO SÁNH PHƯƠNG PHÁP",["Method","Feasible","Regret mean","Search s"],
        [[r["method"],f"{r['feasibility_rate']:.0%}",f"{r['objective_regret_mean']:.6f}" if r["objective_regret_mean"] is not None else "—",f"{r['search_seconds_mean']:.1f}"] for r in summaries])
    cli_ui.artifact("Báo cáo nghiên cứu",folder/"RESEARCH.md")
    cli_ui.artifact("Dữ liệu đầy đủ",folder/"study.json")
    if best and complete:
        cli_ui.success(f"Tối ưu thực nghiệm trong tập đã khai báo: {best['config_id']}. Không phải optimal toàn cục.")
        cli_ui.artifact("Cấu hình",folder/"selected.json")
    else:
        cli_ui.warning("Chưa có full-reference đầy đủ/khả thi; không chứng nhận cấu hình tối ưu.")
    return 0 if best and complete else 3


def reference_cost(row):
    return row.get("worker_wall_seconds",0)+sum(reference_cost(r) for r in row.get("previous_attempts",[]))


def resume(args):
    """Resume only missing full-reference jobs, never replay/adapt locked search."""
    from .pipeline import write_json, runtime_info, VisionRunner, language_policy
    import psutil
    folder=Path(args.run).resolve()
    protocol=json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
    verify_protocol(protocol)
    status=json.loads((folder/"status.json").read_text(encoding="utf-8"))
    locked=folder/"decisions.locked.json"
    if not locked.exists():
        raise ValueError("Search decisions are not locked. Start a new study; partial budgeted searches cannot be replayed as if uninterrupted")
    decisions=json.loads(locked.read_text(encoding="utf-8")); lock_hash=digest(decisions)
    saved_report=folder/"study.json"
    expected=json.loads(saved_report.read_text(encoding="utf-8"))["decisions_sha256"] if saved_report.exists() else status.get("decisions_sha256")
    if expected!=lock_hash:
        raise ValueError("Locked decision checksum differs from saved status/report")
    parent_record=folder/"study_process.json"
    if parent_record.exists():
        parent=json.loads(parent_record.read_text(encoding="utf-8"))
        try:
            live=psutil.Process(parent["pid"])
            if abs(live.create_time()-parent["create_time"])<0.01:
                raise ValueError("Study parent is still alive; do not resume concurrently")
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as exc:
            raise ValueError("Cannot verify study parent is stopped; refuse concurrent resume") from exc
    for process_file in folder.rglob("process.json"):
        process=json.loads(process_file.read_text(encoding="utf-8"))
        if process.get("status")!="running" or not process.get("create_time"):
            continue
        try:
            live=psutil.Process(process["pid"])
            if abs(live.create_time()-process["create_time"])<0.01:
                raise ValueError("A worker is still alive. Do not resume concurrently with the original study")
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as exc:
            raise ValueError("Cannot verify worker is stopped; refuse concurrent resume") from exc
    hw=runtime_info(protocol["hardware"]["device"])
    for key in ("device","gpu_uuid","gpu","cpu","torch_version","torch_cuda_version","driver_version"):
        if hw.get(key)!=protocol["hardware"].get(key):
            raise ValueError(f"Reference resume requires the same hardware/runtime: {key} differs")
    from importlib.metadata import version
    import platform
    for name,expected_version in protocol.get("runtime_versions",{}).items():
        current=platform.python_version() if name=="python" else platform.platform() if name=="os" else version(name)
        if current!=expected_version:
            raise ValueError(f"Reference runtime changed: {name}")
    verify_source(protocol)
    manifest=json.loads((folder/"dataset.json").read_text(encoding="utf-8"))
    if digest(manifest)!=protocol["dataset_sha256"]:
        raise ValueError("Dataset manifest changed")
    verify_dataset_files(manifest)
    vocab,maximum=language_policy(manifest)
    results=json.loads((folder/"reference.partial.json").read_text(encoding="utf-8")) if (folder/"reference.partial.json").exists() else {f"c{i:02d}":[] for i in range(len(protocol["configs"]))}
    if any(not math.isfinite(v) or v<=0 for v in (args.timeout_hours,args.trial_timeout)):
        raise ValueError("Reference timeout limits must be finite positive values")
    deadline=Deadline(args.timeout_hours*3600)
    import os
    write_json(parent_record,{"pid":os.getpid(),"create_time":psutil.Process().create_time()})
    prior_elapsed=json.loads(saved_report.read_text(encoding="utf-8")).get("elapsed_seconds",0) if saved_report.exists() else None
    write_json(folder/"status.json",{"phase":"full_reference","decisions_sha256":lock_hash,"resumed":True})
    jobs=[(i,s) for i in range(len(protocol["configs"])) for s in protocol["training_seeds"]]
    random.Random(protocol["split_seed"]+1000).shuffle(jobs)
    for index,seed in jobs:
        rows=results[f"c{index:02d}"]
        if any(r["training_seed"]==seed and r.get("full_completed") for r in rows):
            continue
        if deadline.remaining<1:
            break
        verify_source(protocol)
        old=next((r for r in rows if r["training_seed"]==seed),None)
        print(f"FULL REFERENCE RESUME c{index:02d} • seed {seed} • restart fixed {protocol['full_epochs']} epochs",flush=True)
        trial_folder=folder/"reference"/f"c{index:02d}"/f"seed_{seed}"
        runner=VisionRunner(trial_folder,folder/"dataset.json",manifest,manifest,hw,protocol["full_epochs"],vocab,maximum,seed)
        runner.index=max((int(p.name.split("_")[-1]) for p in trial_folder.glob("trial_*")),default=0)
        result=runner.run(protocol["configs"][index],protocol["full_epochs"],min(deadline.remaining,args.trial_timeout),"train")
        result.update(training_seed=seed,full_completed=training_finished(result,protocol["full_epochs"]) and result.get("epochs_completed")==protocol["full_epochs"])
        if old:
            result["previous_attempts"]=old.get("previous_attempts",[])+[{k:v for k,v in old.items() if k!="previous_attempts"}]
        results[f"c{index:02d}"]=[r for r in rows if r["training_seed"]!=seed]+[result]
        write_json(folder/"reference.partial.json",results); cli_ui.trial(result)
    # Interrupted process downtime cannot be reconstructed as active elapsed cost.
    write_json(folder/"resume.json",{"prior_elapsed_seconds":prior_elapsed,"resume_active_seconds":deadline.elapsed,
                                     "downtime_included":False,"search_replayed":False})
    elapsed=prior_elapsed+deadline.elapsed if prior_elapsed is not None else None
    if protocol.get("mode")=="online_verification":
        from .verification import publish_online
        return publish_online(folder,protocol,decisions,vocab,maximum,manifest,elapsed,results)
    return publish(folder,protocol,decisions,results,vocab,maximum,manifest,elapsed,lock_hash)
