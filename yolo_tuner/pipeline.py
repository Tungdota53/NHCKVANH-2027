"""Automatic task routing, portable hardware and bounded measured configuration search."""

from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from .adapters import discover, export_yolo, subset
from .core import Deadline, training_finished
from .runner import hardware, resolve_device, stop_process_tree
from .vision_worker import tokens
from . import cli_ui


def write_json(path,value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    temp.replace(path)


def new_session(output):
    path = Path(output).resolve()/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True)
    return path


def runtime_info(device):
    try:
        from importlib.metadata import PackageNotFoundError
        info = hardware(device)
        info["ml_runtime_ready"] = True
        return info
    except ImportError as exc:
        import psutil
        return {"device":"cpu","gpu":None,"fp16":False,"bf16":False,"ml_runtime_ready":False,
                "ram_available_mb":psutil.virtual_memory().available/2**20,
                "logical_cpus":psutil.cpu_count(),"vram_free_mb":None,"reason":str(exc)}


def language_policy(manifest):
    # Vocabulary and token length are fit on train only and shared by all trials.
    counts=Counter()
    lengths=[]
    for record in manifest["splits"]["train"]:
        for caption in record.get("captions",[]):
            words=tokens(caption); counts.update(words); lengths.append(len(words)+2)
    words=sorted(counts,key=lambda word:(-counts[word],word))[:5996]
    vocab={"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3}
    for word in words:
        if word not in vocab:
            vocab[word]=len(vocab)
    max_tokens=min(96,max(16,sorted(lengths)[int((len(lengths)-1)*0.95)] if lengths else 32))
    return vocab,max_tokens


class VisionRunner:
    def __init__(self,folder,manifest_path,manifest,full,hw,epochs,vocab,max_tokens,seed):
        self.folder=Path(folder); self.folder.mkdir(parents=True,exist_ok=True)
        self.manifest_path=manifest_path; self.manifest=manifest; self.full=full
        self.hw=hw; self.epochs=epochs; self.vocab=vocab; self.max_tokens=max_tokens; self.seed=seed
        self.index=0

    def run(self,config,epochs,seconds,mode="search",resume=None):
        self.index+=1
        folder=self.folder/f"trial_{self.index:04d}"
        folder.mkdir()
        task=self.manifest["task"]
        full_counts=self.full["captions"] if task=="captioning" else self.full["counts"]
        by_source={split:dict(Counter({})) for split in ("train","val")}
        for split in by_source:
            counter=Counter()
            for record in self.full["splits"][split]:
                counter[record["source"]]+=len(record["captions"]) if task=="captioning" else 1
            by_source[split]=dict(counter)
        device=self.hw["device"]
        request={"task":task,"config":config,"manifest":str(self.manifest_path),
                 "device":device if device in {"cpu","mps"} else f"cuda:{device}",
                 "vocabulary":self.vocab,"max_tokens":self.max_tokens,"full_counts":full_counts,
                 "full_counts_by_source":by_source,
                 "full_epochs":self.epochs,"trial_epochs":epochs,"mode":mode,"resume":resume,
                 "seed":self.seed,"threads":min(4,self.hw.get("logical_cpus") or 1),
                 # Identical cache policy for short and full runs. Otherwise a
                 # tiny subset fits entirely in RAM and badly understates full I/O.
                 "cache_mb":0,"project":str(folder)}
        if task=="captioning":
            common=Counter(" ".join(tokens(cap)) for row in self.full["splits"]["train"] for cap in row["captions"])
            request["caption_baseline"]=common.most_common(1)[0][0] if common else ""
        request_path,result_path=folder/"request.json",folder/"result.json"
        write_json(request_path,request)
        started=time.monotonic()
        timeout=False
        with (folder/"worker.log").open("w",encoding="utf-8") as log:
            import os
            env=None
            if config.get("deterministic"):
                env={**os.environ,"CUBLAS_WORKSPACE_CONFIG":":4096:8"}
            process=subprocess.Popen([sys.executable,"-m","yolo_tuner.vision_worker",str(request_path),str(result_path)],
                                     stdout=log,stderr=subprocess.STDOUT,env=env)
            import psutil
            try:
                create_time=psutil.Process(process.pid).create_time()
            except psutil.NoSuchProcess:
                create_time=None
            process_record={"pid":process.pid,"create_time":create_time,"request":str(request_path),"status":"running"}
            write_json(folder/"process.json",process_record)
            end_time=time.monotonic()+max(0.01,seconds)
            announced_epoch=0
            try:
                while True:
                    remaining=end_time-time.monotonic()
                    if remaining<=0:
                        raise subprocess.TimeoutExpired(process.args,seconds)
                    try:
                        process.wait(timeout=min(15,remaining))
                        break
                    except subprocess.TimeoutExpired:
                        if mode=="train" and result_path.exists():
                            progress=json.loads(result_path.read_text(encoding="utf-8"))
                            if progress.get("epochs_completed",0)>announced_epoch:
                                cli_ui.training_epoch(progress,self.epochs)
                                announced_epoch=progress["epochs_completed"]
                        if process.poll() is not None:
                            break
            except subprocess.TimeoutExpired:
                timeout=True; stop_process_tree(process)
            except BaseException:
                stop_process_tree(process); raise
            finally:
                process_record.update(status="terminated" if timeout else "exited",return_code=process.poll())
                write_json(folder/"process.json",process_record)
        result=json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
            "status":"error","config":config,"error":f"Worker exited {process.returncode} before writing a result"}
        if timeout:
            result.update(status="timeout",error="Worker reached its wall-clock limit")
        elif process.returncode!=0:
            result.update(status="error",error=f"Worker exited {process.returncode}")
        result.update(worker_wall_seconds=time.monotonic()-started,log=str(folder/"worker.log"),requested_epochs=epochs,mode=mode)
        if result["status"]=="ok":
            setup=max(0,result["worker_wall_seconds"]-result.get("measured_loop_seconds",0))
            result["estimated_full_seconds"]=result["estimated_full_seconds"]+setup if mode=="search" else result["worker_wall_seconds"]
            result["memory_limit_mb"]=(self.hw.get("vram_free_mb") or self.hw["ram_available_mb"])*0.8
            result["resource_feasible"]=result.get("memory_mb",math.inf)<=result["memory_limit_mb"]
        return result


def candidate_configs(task,hw,manifest,max_candidates=6):
    constrained=hw["device"]=="cpu" or hw["ram_available_mb"]<1500
    image_size=128 if constrained else 160
    precision="fp16" if hw["device"] not in {"cpu","mps"} and hw.get("fp16") else "fp32"
    baseline={"model":{"captioning":"cnn_gru_captioner","classification":"cnn_classifier",
                       "reconstruction":"cnn_autoencoder"}[task],
              "imgsz":image_size,"batch":2 if constrained else 4,"precision":precision,
              "width":16,"hidden":128,"effective_batch":32,"optimizer":"AdamW","lr0":0.001,
              "workers":0,"image_cache_mb":0,"pretrained":False,"epochs":20}
    if task=="captioning":
        baseline.update(model="attn_gru_captioner",encoder="cnn",hidden=256,width=32,
                        batch=8 if constrained else 32,effective_batch=64,feature_grid=7,
                        dropout=0.3,label_smoothing=0.05,weight_decay=0.01,patience=5,min_delta=0.001)
        changes=[{}, {"hidden":384}, {"lr0":0.0003}, {"dropout":0.5},
                 {"hidden":384,"lr0":0.0003}, {"hidden":128}]
        return [{**baseline,**change,"batch":min(baseline["batch"],manifest["counts"]["train"])}
                for change in changes[:max_candidates]]
    changes=[{}, {"width":32,"hidden":256}, {"lr0":0.0003},
             {"imgsz":min(224,image_size+64)}, {"width":32,"hidden":256,"lr0":0.0003},
             {"width":32,"hidden":256,"imgsz":min(224,image_size+64)}]
    return [{**baseline,**change,"batch":min(baseline["batch"],manifest["counts"]["train"])}
            for change in changes[:max_candidates]]


def select_presets(rows):
    rows=[r for r in rows if r["status"]=="ok" and r.get("resource_feasible") and r.get("quality_eligible",True)]
    def dominates(a,b):
        values_a=(a["quality_score"],-a["estimated_full_seconds"],-a["memory_mb"])
        values_b=(b["quality_score"],-b["estimated_full_seconds"],-b["memory_mb"])
        return all(x>=y for x,y in zip(values_a,values_b)) and any(x>y for x,y in zip(values_a,values_b))
    front=[r for r in rows if not any(dominates(other,r) for other in rows)]
    if not front:
        return {},[]
    keys=("quality_score","estimated_full_seconds","memory_mb")
    low={k:min(r[k] for r in front) for k in keys}; high={k:max(r[k] for r in front) for k in keys}
    def balance(row):
        scores=[]
        for key in keys:
            span=high[key]-low[key]
            value=(row[key]-low[key])/span if span else 0
            scores.append(1-value if key=="quality_score" and span else value)
        return max(scores)+0.05*sum(scores)
    return {"Accuracy":max(front,key=lambda r:r["quality_score"]),
            "Time":min(front,key=lambda r:r["estimated_full_seconds"]),
            "Memory":min(front,key=lambda r:r["memory_mb"]),
            "Balanced":min(front,key=balance)},front


def report_markdown(report,selection):
    dataset=report["dataset"]
    c=selection["config"]
    lines=["# Cấu hình huấn luyện tự động", "", f"Trạng thái: **{selection['status']}**. Task: **{dataset['task']}**.","",
           f"Dataset: {dataset['source']}",f"Thiết bị: {selection['device']}; train/val/test: {dataset['counts']}.","",
           "| Tham số | Giá trị |","|---|---|"]
    for key,value in c.items():
        lines.append(f"| {key} | {value} |")
    lines += ["", f"Metric chọn cấu hình: {report.get('quality_metric','chưa đo')}. Tìm kiếm {report['search_wall_seconds']:.1f} giây / {report['budget_seconds']:.1f} giây.","",
              "Đây là cấu hình tốt nhất theo quy tắc Balanced trong các ứng viên đã đo; không phải chứng minh tối ưu toàn cục hoặc accuracy cuối."]
    if dataset["task"]=="captioning":
        lines += ["", "Caption v2: encoder ảnh + GRU đọc ngữ cảnh ảnh ở từng bước và spatial cross-attention. Chọn theo BLEU-4/chrF của câu sinh trên validation; encoder pretrained chỉ bật khi tải đúng weights. Checkpoint cũ CNN+GRU vẫn đọc được. Đây không phải ViECAP gốc hoặc chứng minh tối ưu toàn cục."]
    if dataset["task"]=="reconstruction":
        lines += ["", "Ảnh chưa có nhãn được đưa vào pretraining tái tạo ảnh. Đây là task reconstruction, không phải supervised detection."]
    lines += ["", "Chạy full training:","", "```shell",f'python -m yolo_tuner train --selection "{selection["selection_path"]}"',"```","",
              "## Ghi chú",""]
    lines.extend(f"- {w}" for w in report["warnings"][:20])
    if len(report["warnings"])>20:
        lines.append("- Xem report.json để đọc toàn bộ ghi chú và các ảnh trùng bị loại.")
    return "\n".join(lines)+"\n"


def prepare_caption_backend(args,folder,hw,deadline,manifest_path):
    choice=getattr(args,"caption_encoder","auto")
    if choice=="cnn" or getattr(args,"plan_only",False) or not hw["ml_runtime_ready"]:
        return {"encoder":"resnet18","pretrained":True,"imgsz":224} if choice=="resnet18" else {"encoder":"cnn","pretrained":False}
    cli_ui.warning("Đang chuẩn bị encoder ResNet18 và cache đặc trưng train/val; test không được đưa vào cache này.")
    result_path=folder/"feature_preparation.json"
    log_path=folder/"feature_preparation.log"
    batch=16 if hw["device"] not in {"cpu","mps"} else 4
    last_error="Ngân sách không đủ để chuẩn bị encoder"
    for attempt in range(5):
        if deadline.remaining<1:
            break
        cmd=[sys.executable,"-m","yolo_tuner.caption_features","--manifest",str(manifest_path),
             "--folder",str(folder/"features"),"--result",str(result_path),"--device",hw["device"],"--batch",str(batch)]
        if getattr(args,"offline",False):
            cmd.append("--offline")
        with log_path.open("a",encoding="utf-8") as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=max(0.01,deadline.remaining))
            except subprocess.TimeoutExpired:
                stop_process_tree(process)
                last_error="Feature preparation reached the search wall-clock budget"
                break
            except BaseException:
                stop_process_tree(process)
                raise
        if process.returncode==0 and result_path.is_file():
            metadata=json.loads(result_path.read_text(encoding="utf-8"))
            cli_ui.success("Encoder pretrained đã kiểm tra SHA256; cache đặc trưng đã sẵn sàng.")
            return {"encoder":"resnet18","pretrained":True,"encoder_sha256":metadata["sha256"],
                    "feature_cache":metadata["feature_cache"],"imgsz":224}
        last_error=log_path.read_text(encoding="utf-8")[-2000:]
        if "out of memory" in last_error.lower() and batch>1:
            batch=max(1,batch//2)
            cli_ui.warning(f"Cache encoder hết bộ nhớ; thử lại batch={batch}.")
            continue
        break
    if choice=="resnet18":
        raise ValueError(f"Cannot prepare pretrained encoder: {last_error}; log: {log_path}")
    cli_ui.warning(f"Không chuẩn bị được pretrained encoder; dùng attention CNN học từ đầu, không giả nhận pretrained. Xem {log_path}")
    return {"encoder":"cnn","pretrained":False}


def launch_full(args,folder,selection):
    if not getattr(args,"train",False):
        return None
    from types import SimpleNamespace
    from .cli import train
    evaluate_test=selection["task"]=="captioning" and bool(json.loads((folder/"dataset.json").read_text(encoding="utf-8"))["splits"]["test"])
    return train(SimpleNamespace(selection=str(folder/"selected.json"),device=args.device,data=None,resume=None,
                                  output=str(folder/"full"),timeout_hours=getattr(args,"timeout_hours",24),
                                  oom_retries=2,evaluate_test=evaluate_test,offline=getattr(args,"offline",False)))


def finish_and_train(args,folder,report,selection):
    code=finish(folder,report,selection)
    full_code=launch_full(args,folder,selection)
    return code if full_code is None else full_code


def optimize(args):
    if getattr(args,"plan_only",False) and getattr(args,"train",False):
        raise ValueError("--plan-only and --train cannot be combined")
    if not math.isfinite(args.trial_timeout) or args.trial_timeout<=0 or args.full_epochs<1:
        raise ValueError("trial-timeout and full-epochs must be positive")
    if min(args.train_subset,args.val_subset,args.max_candidates)<1:
        raise ValueError("Subset/candidate limits must be positive")
    if getattr(args,"max_batch",64)<1:
        raise ValueError("max-batch must be positive")
    if args.rungs!=sorted(set(args.rungs)) or min(args.rungs)<1 or max(args.rungs)>args.full_epochs:
        raise ValueError("Rungs must increase and fit within full-epochs")
    folder=new_session(args.output)
    deadline=Deadline(args.budget*60)
    cli_ui.heading("AUTO TRAIN CONFIG", "Tự nhận diện dataset • đo cấu hình trên máy hiện tại • lưu bằng chứng")
    print("  Đang kiểm tra dataset và phần cứng…", flush=True)
    manifest=discover(args.data,args.seed,args.task)
    write_json(folder/"dataset.json",manifest)
    hw=runtime_info(args.device)
    cli_ui.hardware_summary(hw)
    cli_ui.dataset_summary(manifest)
    if manifest["task"]=="detection":
        if not manifest["validation_ready"]:
            raise ValueError("Detection needs independent validation images for tuning")
        data=export_yolo(manifest,folder/"yolo_data")
        from .cli import parser,tune
        models=args.models or ([str(Path("weights/yolo26n.pt").resolve())] if Path("weights/yolo26n.pt").is_file() else ["yolo11n.yaml"])
        argv=["tune","--data",str(data),"--device",hw["device"],"--budget",str(max(0.01,deadline.remaining/60)),
              "--trial-timeout",str(args.trial_timeout),"--full-epochs",str(args.full_epochs),"--force-search",
              "--max-candidates",str(args.max_candidates),"--train-subset",str(args.train_subset),
              "--val-subset",str(args.val_subset),"--output",str(folder/"detection_search"),
              "--freeze","0","--rungs",*[str(e) for e in args.rungs],"--models",*models]
        legacy=parser().parse_args(argv)
        if args.plan_only or not hw["ml_runtime_ready"] or deadline.remaining<1:
            selection={"schema_version":2,"task":"detection","backend":"ultralytics",
                       "status":"recommended_unverified","config":{"model":models[0],
                       "imgsz":640,"batch":1,"precision":"fp16" if hw.get("fp16") else "fp32", "freeze":0,"nbs":64,"optimizer":"AdamW","lr0":0.001},
                       "data":str(data),"device":hw["device"],"full_epochs":args.full_epochs,"seed":args.seed,
                       "final_accuracy_verified":False,"selection_path":str(folder/"selected.json")}
            report={"dataset":manifest,"hardware":hw,"warnings":manifest["warnings"],
                    "search_wall_seconds":deadline.elapsed,"budget_seconds":args.budget*60}
        else:
            code=tune(legacy)
            candidates=sorted((folder/"detection_search").glob("*/selected.json"))
            if not candidates:
                # Always leave a resource-conservative plan when no trial completes.
                args.plan_only=True
                return _detection_fallback(args,folder,manifest,hw,data,deadline,code)
            selection=json.loads(candidates[-1].read_text(encoding="utf-8"))
            selection.update(schema_version=2,task="detection",backend="ultralytics",status="measured_best_observed",
                             selection_path=str(folder/"selected.json"))
            report=json.loads(Path(selection["provenance"]).read_text(encoding="utf-8"))
            report.update(dataset=manifest,hardware=hw,search_wall_seconds=deadline.elapsed,budget_seconds=args.budget*60,
                          quality_metric="early validation mAP50")
        return finish_and_train(args,folder,report,selection)

    sampled=subset(manifest,args.train_subset,args.val_subset)
    write_json(folder/"subset.json",sampled)
    vocab,max_tokens=language_policy(manifest)
    configs=candidate_configs(manifest["task"],hw,manifest,args.max_candidates)
    if manifest["task"]=="captioning":
        if hw["ml_runtime_ready"] and not getattr(args,"plan_only",False):
            import sacrebleu
        backend=prepare_caption_backend(args,folder,hw,deadline,folder/"dataset.json")
        configs=[{**c,**backend} for c in configs]
    for c in configs:
        c["epochs"]=args.full_epochs
        c["batch"]=min(c["batch"],getattr(args,"max_batch",64))
    cli_ui.search_plan(args, configs)
    print(f"  Subset train/val: {sampled['counts']['train']} / {sampled['counts']['val']}")
    if manifest["task"]=="captioning":
        print(f"  Vocabulary: {len(vocab)} token • Giới hạn: {max_tokens} token/caption • Metric: caption sinh BLEU-4/chrF ↑")
    history,cohort=[],[]
    complete=False
    can_measure=hw["ml_runtime_ready"] and sampled["validation_ready"] and bool(sampled["splits"]["val"]) and not args.plan_only
    if not can_measure:
        reason = "--plan-only" if args.plan_only else "thiếu runtime ML hoặc validation độc lập"
        cli_ui.warning(f"Bỏ qua đo thử ({reason}); chỉ xuất cấu hình ban đầu.")
    if can_measure:
        runner=VisionRunner(folder/"trials",folder/"subset.json",sampled,manifest,hw,args.full_epochs,vocab,max_tokens,args.seed)
        active=configs
        for rung in args.rungs:
            wave=[]; finished=True
            cli_ui.search_rung(rung, len(active))
            for index,config in enumerate(active,1):
                if deadline.remaining<1:
                    finished=False; break
                result=None
                for attempt in range(4):
                    cli_ui.trial_start(config,rung,index,len(active),deadline.remaining)
                    result=runner.run(config,rung,min(args.trial_timeout,deadline.remaining))
                    history.append(result)
                    cli_ui.trial(result)
                    if result["status"]=="oom" and config["batch"]>1 and deadline.remaining>1:
                        config={**config,"batch":max(1,config["batch"]//2)}
                        cli_ui.warning(f"Hết bộ nhớ: thử lại với batch={config['batch']}.")
                        continue
                    break
                if result["status"]=="ok" and result["epochs_completed"]==rung and result.get("resource_feasible"):
                    wave.append(result)
                elif result["status"]=="timeout":
                    finished=False; break
            if not finished:
                cohort=cohort or wave
                break
            if not wave:
                break
            cohort=wave
            active=[r["config"] for r in sorted(wave,key=lambda r:(-r["quality_score"],r["estimated_full_seconds"]))[:max(1,math.ceil(len(wave)/2))]]
            complete=rung==args.rungs[-1]
        # Validate throughput at larger batches for the winning architecture.
        # Same effective batch and same rung; only complete probes can replace it.
        if cohort and deadline.remaining>10:
            presets,_=select_presets(cohort)
            winner=presets[args.preset] if presets else max(cohort,key=lambda r:r["quality_score"])
            best=winner
            for factor in (2,4):
                next_batch=winner["config"]["batch"]*factor
                if next_batch>min(getattr(args,"max_batch",64),winner["config"]["effective_batch"]) or deadline.remaining<best["worker_wall_seconds"]*1.3:
                    break
                candidate={**winner["config"],"batch":next_batch}
                cli_ui.trial_start(candidate,winner["epochs_completed"],remaining=deadline.remaining,batch_probe=True)
                result=runner.run(candidate,winner["epochs_completed"],min(args.trial_timeout,deadline.remaining))
                result["batch_verification"]=True; history.append(result)
                cli_ui.trial(result)
                if result["status"]!="ok" or not result.get("resource_feasible"):
                    break
                if (result["throughput"]>best["throughput"]*1.05 and result["quality_score"]>=winner["quality_score"]*0.98
                        and (not winner.get("quality_eligible") or result.get("quality_eligible"))):
                    best=result
            if best is not winner:
                cohort=[best if r is winner else r for r in cohort]
                cli_ui.success(f"Batch {best['config']['batch']} được chấp nhận sau kiểm chứng throughput/chất lượng.")
            else:
                print(f"  Giữ batch {winner['config']['batch']}: tăng batch chưa đạt điều kiện throughput/chất lượng.",flush=True)
    presets,front=select_presets(cohort)
    chosen=presets.get(args.preset)
    measured=chosen or (max(cohort,key=lambda r:r["quality_score"]) if cohort else None)
    selection={"schema_version":2,"task":manifest["task"],"backend":"compact_vision",
               "status":"measured_best_observed" if chosen else "measured_quality_not_established" if measured else "recommended_unverified","preset":args.preset,
               "config":measured["config"] if measured else configs[0],"device":hw["device"],
               "manifest":str(folder/"dataset.json"),"vocabulary":vocab,"max_tokens":max_tokens,
               "full_epochs":args.full_epochs,"seed":args.seed,"names":manifest["names"],"final_accuracy_verified":False,
               "quality_measurement":{k:measured[k] for k in ("metric_name","validation_loss","quality_score","epochs_completed")} if measured else None,
               "caption_metrics":measured.get("caption_metrics") if measured else None,
               "quality_assessment":measured.get("quality_assessment") if measured else None,
               "trainable":manifest["validation_ready"] and hw["ml_runtime_ready"],
               "selection_path":str(folder/"selected.json")}
    report={"schema_version":2,"task":manifest["task"],"dataset":manifest,"hardware":hw,
            "search_wall_seconds":deadline.elapsed,"budget_seconds":args.budget*60,"search_complete":complete,
            "quality_metric":measured["metric_name"] if measured else "unmeasured",
            "presets":presets,"pareto":front,"history":history,
            "warnings":manifest["warnings"]+["Short validation measurements do not certify final or globally optimal accuracy.",
                        "Full-training time is a measured extrapolation, not a guaranteed runtime."]}
    if measured and not chosen:
        report["warnings"].append("Measured candidates did not pass the generated-caption baseline gate; saved config is not quality-certified.")
    elif not chosen:
        report["warnings"].append("No complete feasible trial: a conservative, unverified training recommendation was saved.")
    return finish_and_train(args,folder,report,selection)


def _detection_fallback(args,folder,manifest,hw,data,deadline,code):
    from .core import Config
    c=Config(args.models[0] if args.models else "yolo11n.yaml",batch=1,precision="fp16" if hw.get("fp16") else "fp32")
    from dataclasses import asdict
    selection={"schema_version":2,"task":"detection","backend":"ultralytics","status":"recommended_unverified",
               "config":asdict(c),"data":str(data),"device":hw["device"],"full_epochs":args.full_epochs,"seed":args.seed,
               "final_accuracy_verified":False,"selection_path":str(folder/"selected.json")}
    report={"dataset":manifest,"hardware":hw,"warnings":manifest["warnings"]+["No full candidate completed; fallback is unverified."],
            "search_wall_seconds":deadline.elapsed,"budget_seconds":args.budget*60}
    return finish_and_train(args,folder,report,selection)


def finish(folder,report,selection):
    report["selected_status"]=selection["status"]
    write_json(folder/"report.json",report)
    write_json(folder/"selected.json",selection)
    import yaml
    (folder/"train.yaml").write_text(yaml.safe_dump(selection,sort_keys=False,allow_unicode=True),encoding="utf-8")
    (folder/"REPORT.md").write_text(report_markdown(report,selection),encoding="utf-8")
    cli_ui.pipeline_results(report,selection,folder)
    return 0


def train_selected(args,selection):
    hw=runtime_info(getattr(args,"device","auto"))
    if not hw["ml_runtime_ready"]:
        raise ValueError("Training needs PyTorch. Install the CPU version first; configuration remains saved.")
    manifest=discover(args.data,selection["seed"],selection["task"]) if getattr(args,"data",None) else json.loads(Path(selection["manifest"]).read_text(encoding="utf-8"))
    if selection["task"]=="classification" and selection.get("names",manifest["names"])!=manifest["names"]:
        raise ValueError("Classification class definitions differ from the selected model")
    if not manifest["validation_ready"]:
        raise ValueError("Full training needs independent validation images; add another image or validation split")
    if getattr(args,"evaluate_test",False):
        if selection["task"]!="captioning" or not manifest["splits"]["test"]:
            raise ValueError("--evaluate-test needs a captioning dataset with independent test images")
        import sacrebleu  # Fail before starting hours of training if the extra is absent.
    cli_ui.heading("FULL TRAINING", f"{selection['task']} • {selection['full_epochs']} epoch")
    cli_ui.hardware_summary(hw)
    cli_ui.dataset_summary(manifest)
    folder=new_session(args.output)
    write_json(folder/"dataset.json",manifest)
    config=dict(selection["config"])
    deadline=Deadline(args.timeout_hours*3600)
    if config.get("encoder")=="resnet18":
        from types import SimpleNamespace
        from .caption_features import cache_matches
        if config.get("feature_cache") and cache_matches(config["feature_cache"],manifest,config):
            cli_ui.success("Tái sử dụng cache encoder đã khớp đúng ảnh và preprocessing của full dataset.")
        else:
            prep=SimpleNamespace(caption_encoder="resnet18",plan_only=False,offline=getattr(args,"offline",False))
            config.update(prepare_caption_backend(prep,folder,hw,deadline,folder/"dataset.json"))
    if hw["device"] in {"cpu","mps"}:
        config["precision"]="fp32"
    runtime_selection={**selection,"config":config,"device":hw["device"],"manifest":str(folder/"dataset.json"),
                       "source_selection":selection.get("selection_path"),"selection_path":str(folder/"selected.json")}
    if selection.get("research_provenance"):
        from .research import deployment_scope_changes, verify_dataset_files
        verify_dataset_files(manifest)
        changes=deployment_scope_changes(selection,config,manifest,hw)
        if changes:
            runtime_selection.update(status="transferred_config_optimality_not_verified",research_scope_changes=changes)
            cli_ui.warning("Đã đổi protocol nghiên cứu: "+", ".join(changes)+". Cấu hình dùng được nhưng tính tối ưu chưa được kiểm chứng lại.")
    write_json(folder/"selected.json",runtime_selection)
    runner=VisionRunner(folder/"trials",folder/"dataset.json",manifest,manifest,hw,selection["full_epochs"],
                        selection["vocabulary"],selection["max_tokens"],selection["seed"])
    checkpoint=getattr(args,"resume",None); attempts=[]
    for retry in range(args.oom_retries+1):
        cli_ui.trial_start(config,selection["full_epochs"],remaining=deadline.remaining)
        result=runner.run(config,selection["full_epochs"],deadline.remaining,"train",checkpoint)
        attempts.append(result)
        runtime_selection["config"]=result.get("config",config)
        if selection.get("research_provenance"):
            changes=deployment_scope_changes(selection,runtime_selection["config"],manifest,hw)
            if changes:
                runtime_selection.update(status="modified_config_optimality_not_verified",research_scope_changes=changes)
        write_json(folder/"selected.json",runtime_selection)
        write_json(folder/"training_report.json",{"selection":runtime_selection,"hardware":hw,"attempts":attempts})
        cli_ui.trial(result)
        if result["status"]!="oom" or config["batch"]==1 or deadline.remaining<1:
            break
        config={**config,"batch":max(1,config["batch"]//2)}
        checkpoint=result.get("last_checkpoint")
        if retry<args.oom_retries:
            cli_ui.warning(f"Hết bộ nhớ: giảm batch xuống {config['batch']} và thử lại.")
    completed=training_finished(result,selection["full_epochs"])
    if completed:
        cli_ui.success(f"Hoàn tất training: {result['epochs_completed']}/{selection['full_epochs']} epoch"+
                       (" (early stopping theo caption validation)." if result.get("early_stopped") else "."))
    cli_ui.artifact("Báo cáo huấn luyện",folder/"training_report.json")
    if completed and getattr(args,"evaluate_test",False):
        from types import SimpleNamespace
        from .captioning import evaluate
        best=Path(result["last_checkpoint"]).parent/"best.pt"
        return evaluate(SimpleNamespace(checkpoint=str(best),manifest=str(folder/"dataset.json"),split="test",
                                        device=hw["device"],batch=config["batch"],output=str(folder/"evaluation"),training_run=str(folder)))
    return 0 if completed else 2
