"""Local full-training verification and honest cross-environment evidence."""

import json
from pathlib import Path

from . import cli_ui
from .research import (Objective, aggregate, digest, nondominated, verify_protocol,
                       verify_source, verify_dataset_files, reference_cost)


def content_identity(manifest):
    """Relocation-invariant dataset identity; includes labels and split membership."""
    splits={}
    for split,records in manifest["splits"].items():
        rows=[{k:v for k,v in r.items() if k in {"sha256","captions","class_id","boxes","labels"}} for r in records]
        splits[split]=sorted(rows,key=digest)
    return digest({"task":manifest["task"],"names":manifest.get("names",[]),"splits":splits})


def hardware_identity(hw):
    # One CPU and one CUDA backend on a single host are NOT two physical GPUs.
    if hw["device"] not in {"cpu","mps"}:
        return {"kind":"cuda","physical_id":hw.get("gpu_uuid"),"name":hw.get("gpu"),
                "identity_verified":bool(hw.get("gpu_identity_verified",bool(hw.get("gpu_uuid"))))}
    return {"kind":hw["device"],"physical_id":None,"name":hw.get("cpu"),"identity_verified":False}


def collect_online(protocol,decisions):
    results={f"c{i:02d}":[] for i in range(len(protocol["configs"]))}
    for decision in decisions:
        for observation in decision["history"]:
            if not observation.get("full_data"):
                continue
            result=dict(observation["result"])
            result.update(training_seed=decision["seed"],
                          full_completed=observation["complete"] and observation["epochs"]==protocol["full_epochs"])
            results[observation["config_id"]].append(result)
    return results


def verification_outcome(reference,objective):
    complete=all(r["complete"] for r in reference)
    feasible=[r for r in reference if objective.feasible(r)]
    best=max(feasible,key=objective.value) if feasible else None
    status=("empirical_optimal_in_declared_space" if complete and best else
            "fully_trained_best_observed" if best else "no_verified_feasible_selection")
    return complete,best,status


def publish_online(folder,protocol,decisions,vocab,maximum,manifest,elapsed,results=None):
    from .pipeline import write_json
    verify_protocol(protocol); verify_source(protocol); verify_dataset_files(manifest)
    if digest(decisions)!=digest(json.loads((folder/"decisions.locked.json").read_text(encoding="utf-8"))):
        raise ValueError("Locked online decisions changed")
    results=collect_online(protocol,decisions) if results is None else results
    write_json(folder/"reference.partial.json",results)
    reference=[aggregate(key,rows,protocol["training_seeds"]) for key,rows in results.items()]
    objective=Objective(**protocol["objective"])
    complete,best,status=verification_outcome(reference,objective)
    report={"schema_version":2,"mode":"online_verification","protocol":protocol,"decisions":decisions,
            "decisions_sha256":digest(decisions),"reference":reference,"reference_complete":complete,
            "reference_status":"complete_finite_empirical_online" if complete else "partial_full_space_not_certified",
            "empirical_best":best["config_id"] if best else None,"verification_status":status,
            "pareto":[r["config_id"] for r in nondominated([r for r in reference if objective.feasible(r)])],
            "scores":[],"summaries":[],"elapsed_seconds":elapsed,"offline_reference_seconds":0,
            "online_verification_seconds":sum(reference_cost(r) for rows in results.values() for r in rows),
            "independent_method_regret_available":False,
            "limitations":["Only complete full-data/full-horizon runs across every required seed are eligible for selection.",
                "A partial space yields best observed, NOT a certified optimum. Untested/OOM/timeout candidates remain unresolved.",
                "All confirmation compute is online cost. No free offline oracle or independent method superiority is claimed.",
                "Finite empirical optimum is conditional on this dataset/hardware/space/seeds/objective, not all future machines or datasets."]}
    write_json(folder/"study.json",report)
    from .research import markdown
    (folder/"RESEARCH.md").write_text(markdown(report),encoding="utf-8")
    if best:
        selection={"schema_version":2,"backend":"compact_vision","task":manifest["task"],"status":status,
                   "config":protocol["configs"][int(best["config_id"][1:])],"device":protocol["hardware"]["device"],
                   "manifest":str(folder/"dataset.json"),"full_epochs":protocol["full_epochs"],"seed":protocol["training_seeds"][0],
                   "vocabulary":vocab,"max_tokens":maximum,"names":manifest["names"],"trainable":True,
                   "selection_path":str(folder/"selected.json"),"research_provenance":str(folder/"study.json"),
                   "protocol_sha256":protocol["protocol_sha256"],"optimality_scope":protocol["scope"] if complete else "best observed among fully confirmed configurations; full space unresolved",
                   "full_reference_measurement":{k:v for k,v in best.items() if k!="runs"},"final_accuracy_verified":False}
        write_json(folder/"selected.json",selection)
    elif (folder/"selected.json").exists():
        previous=json.loads((folder/"selected.json").read_text(encoding="utf-8"))
        write_json(folder/"selected.previous.json",previous)
        write_json(folder/"selected.json",{**previous,"status":status,"trainable":False})
    write_json(folder/"status.json",{"phase":"complete","reference_complete":complete,"empirical_best":report["empirical_best"],"verification_status":status})
    cli_ui._table("FULL-TRAINING ĐÃ KIỂM CHỨNG",["Config","Seed đủ","Q mean","Time s","Memory MiB"],
        [[r["config_id"],r["complete"],f"{r['quality_mean']:.6f}" if r["time_mean"] is not None else "—",
          f"{r['time_mean']:.1f}" if r["time_mean"] is not None else "—",r["memory_max"]] for r in reference])
    print(f"  Status: {status} • Selected: {report['empirical_best']} • Full-space verified: {complete}")
    if not complete:
        cli_ui.warning("Chưa đo đủ toàn bộ không gian. Không chứng nhận cấu hình tối ưu; tăng budget hoặc research-resume để đo các lượt còn thiếu.")
    cli_ui.artifact("Báo cáo",folder/"RESEARCH.md")
    if best:
        cli_ui.artifact("Cấu hình",folder/"selected.json")
    return 0 if best else 3


def evidence_row(folder):
    """Read evidence, verify every successful training request; never use a title as proof."""
    folder=Path(folder).resolve()
    report=json.loads((folder/"study.json").read_text(encoding="utf-8"))
    protocol=json.loads((folder/"protocol.json").read_text(encoding="utf-8")); verify_protocol(protocol)
    decisions=json.loads((folder/"decisions.locked.json").read_text(encoding="utf-8"))
    if report["protocol"]!=protocol or report["decisions_sha256"]!=digest(decisions) or report["decisions"]!=decisions:
        raise ValueError("Evidence does not match locked protocol/decisions")
    manifest=json.loads((folder/"dataset.json").read_text(encoding="utf-8"))
    if digest(manifest)!=protocol["dataset_sha256"]:
        raise ValueError("Evidence dataset manifest changed")
    reference=report["reference"]
    if len(reference)!=len(protocol["configs"]) or {r["config_id"] for r in reference}!={f"c{i:02d}" for i in range(len(protocol["configs"]))}:
        raise ValueError("Evidence omits/duplicates a declared configuration")
    checked=0
    for index,row in enumerate(reference):
        for result in row["runs"]:
            if not result.get("full_completed"):
                continue
            trial=Path(result["log"]).parent
            req=json.loads((trial/"request.json").read_text(encoding="utf-8"))
            raw=json.loads((trial/"result.json").read_text(encoding="utf-8"))
            if (req["config"]!=protocol["configs"][int(row["config_id"][1:])] or req["seed"]!=result["training_seed"]
                    or req["trial_epochs"]!=protocol["full_epochs"] or req["mode"]!="train"
                    or Path(req["manifest"]).resolve()!=folder/"dataset.json" or raw.get("status")!="ok"
                    or raw.get("epochs_completed")!=protocol["full_epochs"]
                    or raw.get("config")!=req["config"] or req.get("resume") is not None):
                raise ValueError("Evidence full-training request/result mismatches protocol")
            from .research import measured_quality
            if (measured_quality(raw)!=measured_quality(result) or raw.get("memory_mb")!=result.get("memory_mb")
                    or result["worker_wall_seconds"] < raw.get("measured_loop_seconds",0)):
                raise ValueError("Evidence metrics differ from raw worker result")
            checked+=1
    aggregate_again=[aggregate(r["config_id"],r["runs"],protocol["training_seeds"]) for r in reference]
    complete=all(r["complete"] for r in aggregate_again)
    objective=Objective(**protocol["objective"])
    feasible=[r for r in aggregate_again if objective.feasible(r)]
    best=max(feasible,key=objective.value) if feasible else None
    if complete!=report["reference_complete"] or (report["empirical_best"] is not None and (best is None or best["config_id"]!=report["empirical_best"])):
        raise ValueError("Evidence optimum differs from recomputed measurements")
    portable=[{k:v for k,v in c.items() if k not in {"feature_cache"}} for c in protocol["configs"]]
    return {"run":str(folder),"task":protocol["task"],"dataset_content_id":content_identity(manifest),
            "hardware":hardware_identity(protocol["hardware"]),"runtime":protocol.get("runtime_versions",{k:protocol["hardware"].get(k) for k in ("torch_version","torch_cuda_version")}),
            "space_id":digest(portable),"full_epochs":protocol["full_epochs"],"seeds":protocol["training_seeds"],
            "objective":protocol["objective"],"quality_definition":protocol["quality_definition"],
            "protocol_sha256":protocol["protocol_sha256"],"reference_complete":complete,"checked_full_runs":checked,
            "selected":report["empirical_best"],"mode":report.get("mode","independent_research"),
            "method_summaries":report["summaries"],"generalization_certified":False}


def summarize_evidence(rows):
    unique={r["protocol_sha256"]:r for r in rows}
    rows=list(unique.values())
    completed=[r for r in rows if r["reference_complete"]]
    gpu_ids={r["hardware"]["physical_id"] for r in completed if r["hardware"]["kind"]=="cuda" and r["hardware"]["identity_verified"]}
    datasets={r["dataset_content_id"] for r in completed}
    backends={r["hardware"]["kind"] for r in completed}
    runtime_ids={digest(r["runtime"]) for r in completed}
    paired=[]
    for i,a in enumerate(completed):
        for b in completed[i+1:]:
            keys=("dataset_content_id","space_id","full_epochs","seeds","quality_definition","objective")
            if all(a.get(k)==b.get(k) for k in keys) and a["hardware"]["kind"]!=b["hardware"]["kind"]:
                paired.append({"runs":[a["run"],b["run"]],"selected":[a["selected"],b["selected"]],
                               "same_runtime":a["runtime"]==b["runtime"],"physical_gpu_generalization_proven":False})
    return {"runs":rows,"complete_runs":len(completed),"distinct_dataset_splits":len(datasets),
            "backends":sorted(backends),"distinct_physical_cuda_gpus":len(gpu_ids),"distinct_runtimes":len(runtime_ids),
            "unseen_gpu_validation":"not_established" if len(gpu_ids)<2 else "multiple_gpus_measured_not_a_universal_guarantee",
            "paired_backend_evidence":paired,"unseen_dataset_guarantee":False,"universal_optimality_guarantee":False,
            "limitations":["Dataset identities include split membership; multiple subsets of one source do not prove independent source-domain coverage.",
                "CPU identity is not uniquely verified; CPU/CUDA runtime diversity must not be counted as multiple physical GPUs/hosts.",
                "Historical and post-development benchmarks are not blind held-out evaluations by default.",
                "No pooled quality/memory average across tasks/backends with different metrics, data sizes or protocols."]}


def transfer_report(args):
    from .pipeline import new_session, write_json
    # Independent read/audit for each supplied run; a missing GPU stays missing.
    report=summarize_evidence([evidence_row(p) for p in args.runs])
    folder=new_session(args.output); write_json(folder/"transfer.json",report)
    lines=["# Kiểm chứng chuyển dataset/phần cứng", "",
           f"Complete runs: {report['complete_runs']}; distinct dataset splits: {report['distinct_dataset_splits']}; backends: {report['backends']}.",
           f"Physical CUDA GPUs có UUID đã đo: {report['distinct_physical_cuda_gpus']}. Unseen GPU: {report['unseen_gpu_validation']}.",
           "Không có bảo đảm tối ưu cho mọi dataset/GPU. Các lượt đo lại là kiểm chứng local, không phải chứng minh predictor chuyển miền.","",
           "| Run | Task | Backend | Full runs checked | Full-space complete | Selected |",
           "|---|---|---|---:|---|---|"]
    for row in report["runs"]:
        lines.append(f"| {row['run']} | {row['task']} | {row['hardware']['kind']} | {row['checked_full_runs']} | {row['reference_complete']} | {row['selected']} |")
    if report["paired_backend_evidence"]:
        lines += ["", "## Paired backend evidence", ""]
        for pair in report["paired_backend_evidence"]:
            lines.append(f"- Same content/grid/fidelity/seeds/objective; selected {pair['selected']}; runtime identical: {pair['same_runtime']}. CPU/CUDA pair, not two independent GPUs.")
    lines += ["", "## Giới hạn", ""]+[f"- {x}" for x in report["limitations"]]
    (folder/"TRANSFER.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    cli_ui.heading("KIỂM CHỨNG CHUYỂN MÔI TRƯỜNG", "Đối chiếu raw measurements; không tạo bằng chứng GPU giả")
    print(f"  Complete domains: {report['complete_runs']} • dataset splits: {report['distinct_dataset_splits']} • CUDA GPU UUIDs: {report['distinct_physical_cuda_gpus']}")
    cli_ui.warning(f"GPU chưa thấy: {report['unseen_gpu_validation']}. Không chứng nhận tối ưu tổng quát.")
    cli_ui.artifact("Báo cáo",folder/"TRANSFER.md")
    return 0
