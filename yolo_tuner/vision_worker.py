"""Actual short/full training for captions, image classes and reconstruction."""

from collections import OrderedDict,defaultdict
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import traceback


def tokens(text):
    return re.findall(r"\w+|[^\w\s]",text.lower(),flags=re.UNICODE)


def atomic(path,obj):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    temp.replace(path)


def weighted_epoch_time(phase_seconds,subset_examples,full_examples,source_io,full_by_source,epochs_measured=1):
    """Keep compute/sample scaling separate from source-dependent image decoding."""
    io_seconds=sum(v[0] for v in source_io.values())/epochs_measured
    compute=max(0,phase_seconds-io_seconds)*full_examples/subset_examples
    if not full_by_source:
        return compute+io_seconds*full_examples/subset_examples
    full_io=0.0
    for source,n in full_by_source.items():
        observed=source_io.get(source)
        average=observed[0]/observed[1] if observed and observed[1] else io_seconds/subset_examples
        full_io+=average*n
    return compute+full_io


def execute(request,result_path):
    if request["config"].get("model") == "attn_gru_captioner":
        from .caption_worker import execute as execute_caption
        return execute_caption(request, result_path)
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader,Dataset
    import torch.nn.functional as F
    from .vision_models import build

    random.seed(request["seed"]); np.random.seed(request["seed"]); torch.manual_seed(request["seed"])
    torch.set_num_threads(request.get("threads",min(4,psutil.cpu_count() or 1)))
    device = torch.device(request["device"])
    config = request["config"]
    if config.get("deterministic"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    task = request["task"]
    manifest = json.loads(Path(request["manifest"]).read_text(encoding="utf-8"))
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    vocabulary = request.get("vocabulary",{})
    max_tokens = request.get("max_tokens",48)

    class LocalDataset(Dataset):
        def __init__(self,records):
            self.items = []
            self.cache = OrderedDict()
            self.cache_bytes = 0
            self.io_stats=defaultdict(lambda:[0.0,0])
            # A small bounded image cache helps repeated captions and never assumes
            # all images fit in RAM. workers=0 avoids process-copy amplification.
            self.cache_limit = request.get("cache_mb",32)*2**20
            for record in records:
                if task == "captioning":
                    self.items.extend((record,cap) for cap in record["captions"])
                else:
                    self.items.append((record,None))

        def __len__(self):
            return len(self.items)

        def __getitem__(self,index):
            io_start=time.monotonic()
            record,caption = self.items[index]
            path = record["image"]
            image = self.cache.pop(path,None)
            if image is None:
                with Image.open(path) as source:
                    source = source.convert("RGB").resize((config["imgsz"],config["imgsz"]))
                    image = torch.from_numpy(np.asarray(source).copy()).permute(2,0,1).float()/255
                size = image.numel()*image.element_size()
                if size<=self.cache_limit:
                    while self.cache and self.cache_bytes+size>self.cache_limit:
                        _,old = self.cache.popitem(last=False)
                        self.cache_bytes -= old.numel()*old.element_size()
                    self.cache[path] = image
                    self.cache_bytes += size
            else:
                self.cache[path] = image
            self.io_stats[record["source"]][0]+=time.monotonic()-io_start
            self.io_stats[record["source"]][1]+=1
            if task == "captioning":
                ids = [2]+[vocabulary.get(t,1) for t in tokens(caption)[:max_tokens-2]]+[3]
                ids += [0]*(max_tokens-len(ids))
                return image,torch.tensor(ids,dtype=torch.long)
            if task == "classification":
                return image,torch.tensor(record["class_id"],dtype=torch.long)
            return image,image

    train_set,val_set = [LocalDataset(manifest["splits"][s]) for s in ("train","val")]
    if not train_set or not val_set:
        raise ValueError("Actual quality search requires independent train and validation images")
    generator = torch.Generator().manual_seed(request["seed"])
    train_loader = DataLoader(train_set,batch_size=config["batch"],shuffle=True,num_workers=0,generator=generator)
    val_loader = DataLoader(val_set,batch_size=config["batch"],shuffle=False,num_workers=0)
    model = build(task,config,len(vocabulary),len(manifest["names"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config["lr0"],weight_decay=0.01)
    precision = config["precision"]
    use_amp = cuda and precision in {"fp16","bf16"}
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda",enabled=use_amp and precision=="fp16")
    best_loss = math.inf
    folder = Path(request["project"])
    checkpoint = folder/"last.pt"
    start_epoch = 0
    if request.get("resume"):
        ckpt = torch.load(request["resume"],map_location=device,weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        best_loss = ckpt["best_loss"]
    state = {"status":"running","task":task,"config":config,"epochs_completed":start_epoch,
             "torch_version":torch.__version__,"device":str(device),"learning_curve":[],
             "peak_reserved_mb":0,"peak_allocated_mb":0,"peak_rss_mb":0,
             "pretrained":False,"model_origin":"self-contained CNN baseline trained from scratch",
             "metric_name":{"captioning":"validation_token_cross_entropy","classification":"validation_accuracy",
                            "reconstruction":"validation_pixel_mse"}[task]}
    if start_epoch>=request["full_epochs"]:
        state.update(status="ok",already_completed=True,estimated_full_seconds=0.0,
                     **ckpt.get("metrics",{}))
        state["last_checkpoint"]=request["resume"]
        atomic(result_path,state)
        return
    process = psutil.Process()
    started = time.monotonic()
    times = []
    io_times=[]

    def sync():
        if cuda:
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def resource():
        state["peak_rss_mb"] = max(state["peak_rss_mb"],process.memory_info().rss/2**20)
        if cuda:
            state["peak_reserved_mb"] = max(state["peak_reserved_mb"],torch.cuda.max_memory_reserved(device)/2**20)
            state["peak_allocated_mb"] = max(state["peak_allocated_mb"],torch.cuda.max_memory_allocated(device)/2**20)
        state["memory_mb"] = state["peak_reserved_mb"] if cuda else state["peak_rss_mb"]
        state["last_checkpoint"] = str(checkpoint) if checkpoint.exists() else None

    def loss_for(images,targets):
        if task == "captioning":
            logits = model(images,targets[:,:-1])
            truth = targets[:,1:]
            count = max(1,int((truth!=0).sum().item()))
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),truth.reshape(-1),ignore_index=0,reduction="sum")
            return loss,count,0
        if task == "classification":
            logits = model(images)
            return F.cross_entropy(logits,targets,reduction="sum"),len(targets),int((logits.argmax(1)==targets).sum().item())
        output = model(images)
        return F.mse_loss(output,targets,reduction="sum"),targets.numel(),0

    try:
        full_epochs = request["full_epochs"]
        target_epochs = full_epochs if request["mode"]=="train" else request["trial_epochs"]
        for epoch in range(start_epoch,target_epochs):
            sync(); epoch_start = time.monotonic()
            model.train(); optimizer.zero_grad(set_to_none=True)
            train_sum = 0.0; train_count = 0; pending_count = 0; pending_images = 0; examples = 0
            io_before=[{k:list(v) for k,v in ds.io_stats.items()} for ds in (train_set,val_set)]
            for i,(images,targets) in enumerate(train_loader):
                images,targets = images.to(device),targets.to(device)
                with torch.autocast(device_type=device.type,dtype=dtype,enabled=use_amp):
                    loss,count,_ = loss_for(images,targets)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; reduce learning rate or precision")
                # Accumulate summed gradients then divide by actual tokens/pixels/
                # examples at each update, including the final partial group.
                scaler.scale(loss).backward()
                pending_count += count; pending_images += len(images); examples += len(images)
                train_sum += float(loss.detach().float().item()); train_count += count
                if pending_images >= config["effective_batch"] or i+1==len(train_loader):
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(pending_count)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    pending_count=0; pending_images=0
                resource()
            sync(); val_started = time.monotonic()
            model.eval(); total=0.0; count=0; correct=0
            with torch.no_grad():
                for images,targets in val_loader:
                    images,targets=images.to(device),targets.to(device)
                    with torch.autocast(device_type=device.type,dtype=dtype,enabled=use_amp):
                        loss,denom,hits=loss_for(images,targets)
                    total+=float(loss.float().item()); count+=denom; correct+=hits
                    resource()
            sync(); ended=time.monotonic()
            val_loss=total/max(1,count)
            if not math.isfinite(val_loss):
                raise ValueError("Non-finite validation loss")
            quality = correct/count if task=="classification" else (math.exp(-min(val_loss,80)) if task=="captioning" else 1/(1+val_loss))
            times.append((val_started-epoch_start,ended-val_started))
            io_times.append([{k:[value[0]-before.get(k,[0,0])[0],value[1]-before.get(k,[0,0])[1]]
                              for k,value in ds.io_stats.items()} for ds,before in zip((train_set,val_set),io_before)])
            measured=times[1:] or times
            train_time=statistics_mean(t[0] for t in measured)
            val_time=statistics_mean(t[1] for t in measured)
            counts=request["full_counts"]
            measured_io=io_times[1:] or io_times
            extrapolated=[]
            for index,(split,phase_s,ds) in enumerate((("train",train_time,train_set),("val",val_time,val_set))):
                source_io=defaultdict(lambda:[0.0,0])
                for epoch_io in measured_io:
                    for source,(seconds,n) in epoch_io[index].items():
                        source_io[source][0]+=seconds; source_io[source][1]+=n
                extrapolated.append(weighted_epoch_time(phase_s,len(ds),counts[split],source_io,
                                    request.get("full_counts_by_source",{}).get(split,{}),len(measured_io)))
            full_time=full_epochs*sum(extrapolated)
            state.update(status="ok",epochs_completed=epoch+1,validation_loss=val_loss,
                         quality_score=quality,train_loss=train_sum/max(1,train_count),
                         estimated_full_seconds=full_time,train_epoch_seconds=train_time,val_epoch_seconds=val_time,
                         throughput=examples/max(times[-1][0],1e-9),
                         time_estimate_basis="compute scaled by examples + measured image loading weighted by full source counts; point estimate only",
                         measured_loop_seconds=sum(sum(t) for t in times),
                         validation_examples=len(val_set),train_examples=len(train_set),
                         validation_accuracy=correct/count if task=="classification" else None)
            state["learning_curve"].append({"epoch":epoch+1,"validation_loss":val_loss,"quality_score":quality})
            if request["mode"]=="train":
                best_loss=min(best_loss,val_loss)
                payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                         "epoch":epoch+1,"best_loss":best_loss,"task":task,"config":config,
                         "vocabulary":vocabulary,"names":manifest["names"],"max_tokens":max_tokens,
                         "metrics":{"validation_loss":val_loss,"quality_score":quality}}
                temporary=folder/"checkpoint.tmp"
                torch.save(payload,temporary); temporary.replace(checkpoint)
                if val_loss==best_loss:
                    torch.save(payload,folder/"best.pt")
            resource(); state["wall_seconds"]=time.monotonic()-started
            atomic(result_path,state)
        state["wall_seconds"]=time.monotonic()-started
        atomic(result_path,state)
    except Exception as exc:
        resource()
        state["status"]="oom" if "out of memory" in str(exc).lower() else "error"
        state["error"]=str(exc); state["wall_seconds"]=time.monotonic()-started
        atomic(result_path,state); traceback.print_exc()


def statistics_mean(values):
    values=list(values)
    return sum(values)/len(values)


def main():
    request_path,result_path=sys.argv[1:3]
    request=json.loads(Path(request_path).read_text(encoding="utf-8"))
    try:
        execute(request,result_path)
    except Exception as exc:
        state=json.loads(Path(result_path).read_text(encoding="utf-8")) if Path(result_path).is_file() else {}
        state.update(status="oom" if "out of memory" in str(exc).lower() else "error",config=request["config"],error=str(exc),training_completed=False)
        checkpoint=Path(request["project"])/"last.pt"
        if checkpoint.is_file():
            state["last_checkpoint"]=str(checkpoint)
        atomic(result_path,state)
        traceback.print_exc()


if __name__=="__main__":
    main()
