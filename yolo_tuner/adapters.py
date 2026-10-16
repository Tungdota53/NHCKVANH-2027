"""Task-aware local image dataset ingestion without importing any ML runtime.

Inputs: YOLO YAML/layout, COCO boxes/captions JSON, caption CSV, ImageFolder,
or unlabelled image folders. Source files are never modified.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re

from .dataset import IMAGE_SUFFIXES, annotations, images_from_entry, label_path

SKIP = {".git", ".venv", ".venv-cpu", "__pycache__", "runs", "node_modules"}


def file_hash(path):
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def walk(root):
    import os
    for directory, children, files in os.walk(root):
        children[:] = sorted(d for d in children if d not in SKIP and not d.startswith("."))
        for name in sorted(files):
            yield Path(directory) / name


def split_name(path):
    for part in reversed(path.parts):
        tokens = re.split(r"[_.\-]", part.lower())
        for split in ("train", "val", "valid", "validation", "test"):
            if split in tokens or any(t.startswith(split + "20") for t in tokens):
                return "val" if split in {"valid", "validation"} else split
    return "unsplit"


def image_index(root):
    by_name = defaultdict(list)
    images = []
    for path in walk(root):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path.resolve())
            by_name[path.name].append(path.resolve())
    return images, by_name


def resolve_image(name, annotation, root, index):
    raw = Path(name)
    candidates = [raw] if raw.is_absolute() else [annotation.parent / raw, root / raw]
    candidates += [annotation.parent / folder / raw for folder in ("images", "image", "train2017", "val2017")]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    matches = index.get(raw.name, [])
    # Narrow by nearest source folder before accepting a global basename match.
    near = [p for p in matches if annotation.parent in p.parents]
    if len(near) == 1:
        return near[0]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Missing/ambiguous image {name!r} referenced by {annotation}")


def caption_records(file, root, index):
    doc = json.loads(file.read_text(encoding="utf-8-sig"))
    if not isinstance(doc, dict) or not isinstance(doc.get("images"), list) or not isinstance(doc.get("annotations"), list):
        return None
    anns = doc["annotations"]
    if not anns or not any("caption" in a for a in anns):
        return None
    captions = defaultdict(list)
    for ann in anns:
        cap = ann.get("caption")
        if not isinstance(cap, str) or not cap.strip():
            raise ValueError(f"Empty caption in {file}")
        captions[ann["image_id"]].append(cap.strip())
    known = {i["id"] for i in doc["images"]}
    if set(captions) - known:
        raise ValueError(f"Orphan caption image IDs in {file}")
    records = []
    for image in doc["images"]:
        if not captions[image["id"]]:
            raise ValueError(f"Image {image['id']} has no caption in {file}")
        path = resolve_image(image["file_name"], file, root, index)
        records.append({"image": str(path), "captions": sorted(set(captions[image["id"]])),
                        "source": str(file.parent), "split": split_name(file)})
    return records


def coco_records(file, root, index):
    from PIL import Image
    doc = json.loads(file.read_text(encoding="utf-8-sig"))
    if not isinstance(doc, dict) or not doc.get("categories") or not doc.get("images"):
        return None
    if not any("bbox" in a for a in doc.get("annotations", [])):
        return None
    categories = sorted(doc["categories"], key=lambda c: c["id"])
    ids = {c["id"]: i for i,c in enumerate(categories)}
    boxes = defaultdict(list)
    for ann in doc["annotations"]:
        if ann.get("iscrowd"):
            continue
        if ann["category_id"] not in ids:
            raise ValueError(f"Unknown COCO category in {file}")
        boxes[ann["image_id"]].append([ids[ann["category_id"]], *ann["bbox"]])
    records = []
    for image in doc["images"]:
        path = resolve_image(image["file_name"], file, root, index)
        with Image.open(path) as im:
            w,h = im.size
        normalized = []
        for cls,x,y,bw,bh in boxes[image["id"]]:
            if not all(math.isfinite(v) for v in (x,y,bw,bh)) or bw <= 0 or bh <= 0:
                raise ValueError(f"Invalid COCO box in {file}")
            x1,y1,x2,y2 = max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"COCO box outside image in {file}")
            normalized.append([cls,(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h])
        records.append({"image": str(path), "boxes": normalized, "source": str(file.parent), "split": split_name(file)})
    return records, [c["name"] for c in categories]


def external_box_records(files,root,index):
    """Pascal VOC XML and LabelMe rectangle annotations to normalized boxes."""
    import xml.etree.ElementTree as ET
    from PIL import Image
    records=[]
    for file in files:
        boxes=[]; filename=None
        if file.suffix.lower()==".xml":
            doc=ET.parse(file).getroot()
            if doc.tag!="annotation":
                continue
            filename=doc.findtext("filename")
            for obj in doc.findall("object"):
                box=obj.find("bndbox")
                if box is None:
                    continue
                # Pascal VOC uses inclusive 1-based coordinates.
                boxes.append((obj.findtext("name"),float(box.findtext("xmin"))-1,
                              float(box.findtext("ymin"))-1,float(box.findtext("xmax")),float(box.findtext("ymax"))))
        elif file.suffix.lower()==".json":
            doc=json.loads(file.read_text(encoding="utf-8-sig"))
            if not isinstance(doc,dict) or "shapes" not in doc or "imagePath" not in doc:
                continue
            filename=doc["imagePath"]
            for shape in doc["shapes"]:
                if shape.get("shape_type")!="rectangle":
                    raise ValueError(f"LabelMe shape {shape.get('shape_type')} is not a detection rectangle: {file}")
                points=shape["points"]
                if len(points)!=2:
                    raise ValueError(f"Invalid LabelMe rectangle: {file}")
                (x1,y1),(x2,y2)=points
                boxes.append((shape["label"],min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)))
        if not filename:
            continue
        path=resolve_image(filename,file,root,index)
        with Image.open(path) as image:
            width,height=image.size
        normalized=[]
        for name,x1,y1,x2,y2 in boxes:
            if not name or not all(math.isfinite(x) for x in (x1,y1,x2,y2)):
                raise ValueError(f"Invalid box/label: {file}")
            x1,y1,x2,y2=max(0,x1),max(0,y1),min(width,x2),min(height,y2)
            if x2<=x1 or y2<=y1:
                raise ValueError(f"Box outside image: {file}")
            normalized.append([name,(x1+x2)/2/width,(y1+y2)/2/height,(x2-x1)/width,(y2-y1)/height])
        records.append({"image":str(path),"boxes":normalized,"source":str(root),"split":split_name(path.relative_to(root))})
    names=sorted({box[0] for record in records for box in record["boxes"]})
    mapping={name:i for i,name in enumerate(names)}
    for record in records:
        record["boxes"]=[[mapping[box[0]],*box[1:]] for box in record["boxes"]]
    return (records,names) if records and names else None


def yolo_yaml(file):
    import yaml
    doc = yaml.safe_load(file.read_text(encoding="utf-8-sig"))
    if not isinstance(doc, dict) or "train" not in doc or "names" not in doc:
        return None
    root = Path(doc.get("path") or file.parent)
    root = root.resolve() if root.is_absolute() else (file.parent / root).resolve()
    names = doc["names"]
    if isinstance(names, dict):
        if {str(i) for i in names} != {str(i) for i in range(len(names))}:
            raise ValueError("YOLO class names must start at 0 and be contiguous")
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if not isinstance(names, list) or not names:
        raise ValueError("YOLO names must be nonempty")
    records = []
    for split in ("train", "val", "test"):
        if not doc.get(split):
            continue
        for image in images_from_entry(doc[split], root):
            annotations(image, len(names))
            label = label_path(image)
            values = [[float(v) for v in line.split()] for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
            records.append({"image": str(image), "boxes": values, "source": str(root), "split": split})
    return records, names


def partition(records, seed, warnings):
    """Deduplicate by image content; reject conflicting labels across input splits."""
    rng = random.Random(seed)
    splits = {k: [] for k in ("train", "val", "test", "unsplit")}
    by_hash = {}
    # Keep training before val; discard duplicate validation to avoid leakage.
    for record in sorted(records, key=lambda r: ({"train":0,"val":1,"test":2,"unsplit":3}[r["split"]], r["image"])):
        path = Path(record["image"])
        digest = file_hash(path)
        record["sha256"] = digest
        old = by_hash.get(digest)
        if old:
            if old["split"] == record["split"] and "captions" in record:
                old["captions"] = sorted(set(old["captions"]) | set(record["captions"]))
            elif "class_id" in record and record["class_id"] != old.get("class_id"):
                raise ValueError("Identical image content has conflicting classification labels")
            warnings.append(f"Duplicate image excluded: {path.name} ({record['split']})")
            continue
        by_hash[digest] = record
        splits[record["split"]].append(record)
    if splits["unsplit"]:
        records = splits.pop("unsplit")
        groups = defaultdict(list)
        for r in records:
            groups[r.get("class_id", r["source"])].append(r)
        for group in groups.values():
            rng.shuffle(group)
            nval = max(1, round(len(group)*0.2)) if len(group)>1 else 0
            splits["val"].extend(group[:nval])
            splits["train"].extend(group[nval:])
        warnings.append("Created deterministic train/val split by image; audit source/video groups separately.")
    else:
        splits.pop("unsplit")
    if not splits["train"]:
        # A single supplied annotation file must not be stranded as val-only/test-only.
        if splits["val"] and not splits["test"]:
            splits["train"], splits["val"] = splits["val"], []
            warnings.append("Only validation input supplied; treating it as development data and making a new split.")
        else:
            raise ValueError("No training images found. Test-only data cannot be used for automatic tuning.")
    if not splits["val"] and len(splits["train"])>1:
        by_class = defaultdict(list)
        for r in splits["train"]:
            by_class[r.get("class_id", "all")].append(r)
        train, val = [], []
        for group in by_class.values():
            rng.shuffle(group)
            n = max(1, round(len(group)*0.2)) if len(group)>1 else 0
            val.extend(group[:n]); train.extend(group[n:])
        splits["train"],splits["val"] = train,val
        warnings.append("Created validation holdout from train; original test was preserved.")
    # A merged dataset must not validate one source exclusively on another.
    train_sources={r["source"] for r in splits["train"]}
    val_sources={r["source"] for r in splits["val"]}
    for source in sorted(train_sources-val_sources):
        group=[r for r in splits["train"] if r["source"]==source]
        if len(group)>1:
            rng.shuffle(group)
            held=group[:max(1,round(len(group)*0.2))]
            held_ids={id(r) for r in held}
            splits["train"]=[r for r in splits["train"] if id(r) not in held_ids]
            splits["val"].extend(held)
            warnings.append(f"Created source-specific validation holdout: {source}; test remained untouched.")
    if not splits["val"]:
        warnings.append("Insufficient independent validation images; quality search is unavailable.")
    for split,values in splits.items():
        for record in values:
            record["split"]=split
    return splits


def discover(source, seed=17, task="auto"):
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Dataset not found: {source}")
    root = source if source.is_dir() else source.parent
    if source.is_file() and root.name.lower()=="annotations":
        root=root.parent
    images,index = image_index(root)
    if not images:
        raise ValueError("No local image files found. Supply image data or an adapter manifest.")
    files = list(walk(root)) if source.is_dir() else [source]
    warnings, datasets = [], []
    # Native YOLO YAML wins unless another task was explicitly requested.
    if task in {"auto", "detection"}:
        for file in files:
            if file.suffix.lower() in {".yaml", ".yml"}:
                parsed = yolo_yaml(file)
                if parsed:
                    records,names = parsed
                    if not any(r["boxes"] for r in records):
                        warnings.append("YOLO input contains no object boxes; using image pretraining instead of empty-label detection.")
                        break
                    return finalize(source, "detection", records, names, warnings, seed)
    for file in files:
        if file.suffix.lower() != ".json" or file.stem.endswith(("_pred","_gt")):
            # Repo prediction/GT outputs are evaluation artifacts, not dataset sources.
            continue
        parsed = caption_records(file, root, index)
        if parsed:
            datasets.append(("captioning", parsed, []))
            continue
        parsed = coco_records(file, root, index)
        if parsed:
            datasets.append(("detection", *parsed))
    for file in files:
        if file.suffix.lower() != ".csv" or file.with_suffix(".json").exists():
            continue
        with file.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            image_key = next((k for k in ("image", "image_path", "file_name", "filename") if k in fields), None)
            cap_key = next((k for k in ("caption", "text") if k in fields), None)
            if not image_key or not cap_key:
                continue
            grouped = defaultdict(list)
            for row in reader:
                if not row[cap_key].strip():
                    raise ValueError(f"Empty caption in {file}")
                grouped[str(resolve_image(row[image_key],file,root,index))].append(row[cap_key].strip())
            records = [{"image":p,"captions":sorted(set(c)),"source":str(file.parent),"split":split_name(file)} for p,c in grouped.items()]
            datasets.append(("captioning",records,[]))
    kinds = {d[0] for d in datasets}
    if len(kinds)>1 and task == "auto":
        raise ValueError("Mixed caption/detection inputs found; specify --task or point at a single dataset folder.")
    chosen = task if task != "auto" else next(iter(kinds), None)
    selected = [d for d in datasets if d[0] == chosen]
    if selected:
        names = selected[0][2]
        if any(d[2] != names for d in selected):
            raise ValueError("COCO category definitions differ across sources; use a consistent category map.")
        return finalize(source,chosen,[r for d in selected for r in d[1]],names,warnings,seed)
    if task in {"auto","detection"}:
        external=external_box_records(files,root,index)
        if external:
            return finalize(source,"detection",*external,warnings,seed)
    if task in {"auto","detection"}:
        labelled = [p for p in images if label_path(p).is_file()]
        if labelled:
            all_boxes, maximum = {},-1
            for image in images:
                label = label_path(image)
                boxes = [[float(v) for v in line.split()] for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
                if any(len(box)!=5 for box in boxes):
                    raise ValueError("Detected non-box YOLO labels. This adapter supports five-column detection labels.")
                for box in boxes:
                    maximum = max(maximum,int(box[0]))
                all_boxes[image] = boxes
            if maximum >= 0:
                for image in images:
                    annotations(image,maximum+1)
                records = [{"image":str(p),"boxes":all_boxes[p],"source":str(root),"split":split_name(p.relative_to(root))} for p in images]
                warnings.append("YOLO names not supplied; class names inferred as class_0, class_1, ...")
                return finalize(source,"detection",records,[f"class_{i}" for i in range(maximum+1)],warnings,seed)
    # ImageFolder classification requires at least two genuine class directories.
    class_root = root / "train" if (root/"train").is_dir() else root
    classes = sorted(p.name for p in class_root.iterdir() if p.is_dir() and p.name.lower() not in {"images","labels","val","test","annotations","dataset","src","pretrained"}
                     and any(i.suffix.lower() in IMAGE_SUFFIXES for i in walk(p)))
    if len(classes)>=2 and task in {"auto","classification"}:
        records = []
        for path in images:
            relative = path.relative_to(root)
            parts = relative.parts
            cls = next((p for p in parts[:-1] if p in classes),None)
            if cls:
                records.append({"image":str(path),"class_id":classes.index(cls),"source":str(root),"split":split_name(relative)})
        return finalize(source,"classification",records,classes,warnings,seed)
    if task not in {"auto","reconstruction"}:
        raise ValueError(f"No labels for requested task {task}. Use --task auto for image reconstruction pretraining.")
    warnings.append("No supervised labels found: recommending reconstruction pretraining, not supervised detection accuracy.")
    records = [{"image":str(p),"source":str(root),"split":split_name(p.relative_to(root))} for p in images]
    return finalize(source,"reconstruction",records,[],warnings,seed)


def finalize(source,task,records,names,warnings,seed):
    from PIL import Image
    if not records:
        raise ValueError("Dataset contains no usable image records")
    sizes=[]; usable=[]
    for record in records:
        try:
            with Image.open(record["image"]) as image:
                image.load()
                if len(sizes)<64:
                    sizes.append(list(image.size))
            usable.append(record)
        except (OSError,ValueError) as exc:
            warnings.append(f"Excluded unreadable image {Path(record['image']).name}: {exc}")
    if not usable:
        raise ValueError("No decodable images remain after validation")
    splits = partition(usable,seed,warnings)
    return {"schema_version":1,"task":task,"source":str(source),"seed":seed,"names":names,
            "splits":splits,"warnings":warnings,"sampled_image_sizes":sizes,
            "counts":{k:len(v) for k,v in splits.items()},
            "captions":{k:sum(len(r.get('captions',[])) for r in v) for k,v in splits.items()},
            "validation_ready":bool(splits["val"])}


def subset(manifest,train_size=128,val_size=48):
    result = {**manifest,"splits":{}}
    rng = random.Random(manifest["seed"])
    for name,limit in (("train",train_size),("val",val_size),("test",0)):
        groups = defaultdict(list)
        for record in manifest["splits"][name]:
            groups[record.get("class_id",record["source"])].append(record)
        for group in groups.values():
            rng.shuffle(group)
        values = []
        while groups and len(values)<limit:
            for key in list(groups):
                if len(values)>=limit:
                    break
                values.append(groups[key].pop())
                if not groups[key]:
                    del groups[key]
        result["splits"][name] = values
    result["counts"] = {k:len(v) for k,v in result["splits"].items()}
    result["captions"] = {k:sum(len(r.get('captions',[])) for r in v) for k,v in result["splits"].items()}
    return result


def export_yolo(manifest,folder):
    """Materialize an isolated image/label layout; hardlink or copy, never edit source."""
    import os
    import shutil
    import yaml
    folder = Path(folder).resolve()
    for split in ("train","val","test"):
        for record in manifest["splits"][split]:
            src = Path(record["image"])
            name = record["sha256"] + src.suffix.lower()
            image = folder/"images"/split/name
            label = folder/"labels"/split/Path(name).with_suffix(".txt")
            image.parent.mkdir(parents=True,exist_ok=True)
            label.parent.mkdir(parents=True,exist_ok=True)
            if not image.exists():
                try:
                    os.link(src,image)
                except OSError:
                    shutil.copyfile(src,image)
            label.write_text("\n".join(" ".join(str(v) for v in box) for box in record["boxes"])+"\n",encoding="utf-8")
    spec = {"path":folder.as_posix(),"train":"images/train","val":"images/val","names":manifest["names"]}
    if manifest["splits"]["test"]:
        spec["test"] = "images/test"
    path = folder/"data.yaml"
    path.write_text(yaml.safe_dump(spec,allow_unicode=True),encoding="utf-8")
    return path
