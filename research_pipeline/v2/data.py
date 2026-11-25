"""Strict normalized manifests, read-only quality/anti-leakage audits, and clean dedup copies."""

from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil

from yolo_tuner import adapters as legacy
from yolo_tuner.dataset import IMAGE_SUFFIXES, label_path
from .input_diagnostics import DatasetInputError, check_explicit_file, no_images_error


def _clean_group_id(val):
    if val is None:
        return None
    text = str(val).strip()
    return text if text else None


def _extract_group_id(mapping):
    for key in ("group_id", "scene_id", "video_id", "subject_id"):
        gid = _clean_group_id(mapping.get(key))
        if gid is not None:
            return gid
    return None


def _abspath(path):
    return Path(os.path.abspath(Path(path).expanduser()))


def _is_link_or_junction(path):
    p = Path(path)
    try:
        if p.is_symlink():
            return True
        if hasattr(p, "is_junction") and p.is_junction():
            return True
    except OSError:
        return False
    return False


def _is_redirected_component(cur):
    if _is_link_or_junction(cur):
        return True
    try:
        resolved = cur.resolve()
        expected = cur.parent.resolve() / cur.name
        return os.path.normcase(str(resolved)) != os.path.normcase(str(expected))
    except OSError:
        return False


def _collect_link_targets_between(base_path, target_path, include_base=False, anchor_root=None):
    """Collect resolved targets of symlinks/junctions encountered along path from base_path to target_path."""
    base_abs = _abspath(base_path)
    target_abs = _abspath(target_path)
    anchor_abs = _abspath(anchor_root) if anchor_root is not None else base_abs
    try:
        anchor_res = anchor_abs.resolve()
    except OSError:
        anchor_res = anchor_abs
    targets = []
    if not target_abs.is_relative_to(base_abs):
        return targets

    def _maybe_add(cur_path):
        if not _is_redirected_component(cur_path):
            return
        try:
            resolved = cur_path.resolve()
        except OSError:
            return
        candidate = resolved if resolved.is_dir() else resolved.parent
        # Never add an ancestor above anchor_root (e.g. OS temp root)
        if (anchor_abs.is_relative_to(candidate) and candidate != anchor_abs) or (
            anchor_res.is_relative_to(candidate) and candidate != anchor_res
        ):
            return
        targets.append(candidate)

    cur = base_abs
    if include_base:
        _maybe_add(cur)
    rel = target_abs.relative_to(base_abs)
    for part in rel.parts:
        cur = cur / part
        _maybe_add(cur)
    return targets


def _collect_link_targets_common(anchor_abs, target_abs):
    """Collect intermediate link targets between the common ancestor of anchor_abs/target_abs and target_abs."""
    anchor_p = _abspath(anchor_abs)
    target_p = _abspath(target_abs)
    try:
        common = Path(os.path.commonpath([str(anchor_p), str(target_p)]))
    except ValueError:
        if not target_p.anchor:
            return []
        common = Path(target_p.anchor)
    return _collect_link_targets_between(
        common,
        target_p,
        include_base=(common == anchor_p),
        anchor_root=anchor_p,
    )


def _walk_dir_images_and_links(base_abs, skip_names=None, skip_hidden=False):
    images = []
    img_abs = []
    link_targets = []
    active_dirs = set()
    base_abs = _abspath(base_abs)
    if not base_abs.is_dir():
        return images, img_abs, link_targets

    def _dfs(dp):
        try:
            st = os.stat(dp)
            dir_key = (st.st_dev, st.st_ino)
        except OSError:
            return
        if dir_key in active_dirs:
            return
        active_dirs.add(dir_key)
        try:
            if _is_redirected_component(dp):
                try:
                    link_targets.append(dp.resolve())
                except OSError:
                    pass
            try:
                entries = sorted(os.scandir(dp), key=lambda e: e.name)
            except OSError:
                return
            dir_entries = []
            for entry in entries:
                name = entry.name
                try:
                    is_dir = entry.is_dir(follow_symlinks=True)
                except OSError:
                    is_dir = False
                if is_dir:
                    if (skip_names and name in skip_names) or (skip_hidden and name.startswith(".")):
                        continue
                    sub = dp / name
                    if _is_redirected_component(sub):
                        try:
                            link_targets.append(sub.resolve())
                        except OSError:
                            pass
                    dir_entries.append(sub)
                    continue
                fp = dp / name
                if fp.suffix.lower() in IMAGE_SUFFIXES and fp.is_file():
                    try:
                        fp_res = fp.resolve()
                        images.append(fp_res)
                        img_abs.append(_abspath(fp))
                        if _is_redirected_component(fp):
                            link_targets.append(fp_res.parent)
                    except OSError:
                        pass
            for sub in dir_entries:
                _dfs(sub)
        finally:
            active_dirs.discard(dir_key)

    _dfs(base_abs)
    return images, img_abs, link_targets


def _walk_dir_files_safe(base_abs, skip_names=None, skip_hidden=False):
    files = []
    seen_dirs = set()
    seen_files = set()
    base_abs = _abspath(base_abs)
    if not base_abs.is_dir():
        return files

    def _dfs(dp):
        try:
            st = os.stat(dp)
            dir_key = (st.st_dev, st.st_ino)
        except OSError:
            return
        if dir_key in seen_dirs:
            return
        seen_dirs.add(dir_key)
        try:
            entries = sorted(os.scandir(dp), key=lambda e: e.name)
        except OSError:
            return
        dir_entries = []
        for entry in entries:
            name = entry.name
            try:
                is_dir = entry.is_dir(follow_symlinks=True)
            except OSError:
                is_dir = False
            if is_dir:
                if (skip_names and name in skip_names) or (skip_hidden and name.startswith(".")):
                    continue
                dir_entries.append(dp / name)
                continue
            fp = dp / name
            if fp.is_file():
                try:
                    res_key = os.path.normcase(str(fp.resolve()))
                except OSError:
                    res_key = os.path.normcase(str(_abspath(fp)))
                if res_key not in seen_files:
                    seen_files.add(res_key)
                    files.append(_abspath(fp))
        for sub in dir_entries:
            _dfs(sub)

    _dfs(base_abs)
    return files


def _scan_yolo_entry_images_and_roots(entry, root, root_abs=None):
    root_abs = _abspath(root_abs if root_abs is not None else root)
    root_res = Path(root).resolve()
    pairs = {}
    extra_roots = []
    extra_roots_abs = []
    for raw in entry if isinstance(entry, list) else [entry]:
        raw_p = Path(raw).expanduser()
        p_abs = _abspath(raw_p if raw_p.is_absolute() else root_abs / raw_p)
        p_from_resolved = _abspath(raw_p if raw_p.is_absolute() else root_res / raw_p)
        if not raw_p.is_absolute() and root_abs != root_res:
            extra_roots.extend(_collect_link_targets_common(root_abs, p_from_resolved))
            extra_roots.extend(_collect_link_targets_common(root_res, p_from_resolved))
            if not p_abs.exists():
                p_abs = p_from_resolved
        path = (raw_p if raw_p.is_absolute() else root_res / raw_p).resolve()
        extra_roots.append(path if path.is_dir() else path.parent)
        extra_roots_abs.append(p_abs if p_abs.is_dir() else p_abs.parent)
        extra_roots.extend(_collect_link_targets_common(root_abs, p_abs))
        extra_roots.extend(_collect_link_targets_common(root_res, p_abs))
        if path.is_dir():
            walk_base = p_abs if p_abs.is_dir() else path
            imgs, imgs_abs, dir_links = _walk_dir_images_and_links(walk_base)
            extra_roots.extend(dir_links)
            for res_p, abs_p in zip(imgs, imgs_abs):
                pairs.setdefault(os.path.normcase(str(abs_p)), (res_p, abs_p))
                extra_roots_abs.append(abs_p.parent)
        elif path.suffix.lower() == ".txt" and path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                item = Path(line.strip())
                if not item.is_absolute():
                    item_abs = _abspath(p_abs.parent / item)
                    item_from_resolved = _abspath(path.parent / item)
                    if p_abs.parent != path.parent and not item_abs.exists():
                        item_abs = item_from_resolved
                    item_res = (path.parent / item).resolve()
                    extra_roots.extend(_collect_link_targets_common(root_abs, item_from_resolved))
                    extra_roots.extend(_collect_link_targets_common(root_res, item_from_resolved))
                    extra_roots.extend(_collect_link_targets_common(path.parent, item_from_resolved))
                else:
                    item_abs = _abspath(item)
                    item_res = item.resolve()
                pairs.setdefault(os.path.normcase(str(item_abs)), (item_res, item_abs))
                extra_roots.append(item_res.parent)
                extra_roots_abs.append(item_abs.parent)
                extra_roots.extend(_collect_link_targets_common(root_abs, item_abs))
                extra_roots.extend(_collect_link_targets_common(root_res, item_abs))
                extra_roots.extend(_collect_link_targets_common(p_abs.parent, item_abs))
        elif path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            pairs.setdefault(os.path.normcase(str(p_abs)), (path, p_abs))
            extra_roots.append(path.parent)
            extra_roots_abs.append(p_abs.parent)
        else:
            raise ValueError(f"Dataset split path does not exist: {path}")
    ordered = sorted(pairs.values(), key=lambda pair: str(pair[1]))
    return ordered, extra_roots, extra_roots_abs


def _resolve_yolo_yaml_images(entry, root):
    ordered, _, _ = _scan_yolo_entry_images_and_roots(entry, root)
    return [res_p for res_p, _ in ordered]


def _parse_yaml_names(raw_names):
    if isinstance(raw_names, dict):
        if {str(i) for i in raw_names} != {str(i) for i in range(len(raw_names))}:
            raise ValueError("YOLO class names must start at 0 and be contiguous")
        return [str(raw_names.get(i, raw_names.get(str(i)))) for i in range(len(raw_names))]
    if isinstance(raw_names, list) and raw_names:
        return [str(n) for n in raw_names]
    raise ValueError("YOLO names must be nonempty")


def _yolo_yaml_raw(yaml_file):
    import yaml

    yaml_abs = _abspath(yaml_file)
    yaml_res = yaml_abs.resolve()
    doc = yaml.safe_load(yaml_res.read_text(encoding="utf-8-sig"))
    if not isinstance(doc, dict) or "train" not in doc or "names" not in doc:
        return None
    raw_path_val = doc.get("path")
    dataset_roots = [yaml_res.parent]
    dataset_roots_abs = [yaml_abs.parent]
    if raw_path_val:
        raw_root = Path(raw_path_val).expanduser()
        root_abs = _abspath(raw_root if raw_root.is_absolute() else (yaml_abs.parent / raw_root))
        if not raw_root.is_absolute() and yaml_abs.parent != yaml_res.parent:
            root_from_resolved = _abspath(yaml_res.parent / raw_root)
            dataset_roots.extend(_collect_link_targets_common(yaml_abs.parent, root_from_resolved))
            dataset_roots.extend(_collect_link_targets_common(yaml_res.parent, root_from_resolved))
            if not root_abs.exists():
                root_abs = root_from_resolved
        dataset_roots.extend(_collect_link_targets_common(yaml_abs.parent, root_abs))
        dataset_roots.extend(_collect_link_targets_common(yaml_res.parent, root_abs))
    else:
        root_abs = yaml_abs.parent
    root = root_abs.resolve()
    names = _parse_yaml_names(doc["names"])
    records = []
    dataset_roots.append(root)
    dataset_roots_abs.append(root_abs)
    for split in ("train", "val", "test"):
        if not doc.get(split):
            continue
        ordered, extra_roots, extra_roots_abs = _scan_yolo_entry_images_and_roots(doc[split], root, root_abs=root_abs)
        dataset_roots.extend(extra_roots)
        dataset_roots_abs.extend(extra_roots_abs)
        for image, image_abs in ordered:
            lbl = label_path(image)
            lbl_abs = label_path(image_abs)
            if not lbl.is_file() and lbl_abs.is_file():
                lbl = lbl_abs.resolve()
            dataset_roots_abs.append(lbl_abs.parent)
            dataset_roots.extend(_collect_link_targets_common(root_abs, lbl_abs))
            dataset_roots.extend(_collect_link_targets_common(root, lbl_abs))
            records.append({
                "image": str(image),
                "image_abs": str(image_abs),
                "split": split,
                "source": str(root),
                "label_file": str(lbl),
                "label_file_abs": str(lbl_abs),
                "group_id": None,
            })
    return {
        "format": "yolo_yaml",
        "task": "detection",
        "names": names,
        "records": records,
        "dataset_roots": list(dict.fromkeys(dataset_roots)),
        "dataset_roots_abs": list(dict.fromkeys(dataset_roots_abs)),
        "yaml_doc": doc,
    }


def _coco_raw(json_file, root, index, task, root_abs=None, json_abs=None):
    json_res = Path(json_file).resolve()
    doc = json.loads(json_res.read_text(encoding="utf-8-sig"))
    if not isinstance(doc, dict) or not isinstance(doc.get("images"), list) or not isinstance(doc.get("annotations"), list):
        return None
    root_res = Path(root).resolve()
    root_abs = _abspath(root_abs if root_abs is not None else root)
    json_abs = _abspath(json_abs if json_abs is not None else json_file)

    def _coco_img_cands(file_name):
        raw_fn = Path(file_name).expanduser()
        if raw_fn.is_absolute():
            return [_abspath(raw_fn)]
        cands = [_abspath(json_abs.parent / raw_fn), _abspath(root_abs / raw_fn)]
        if json_abs.parent != json_res.parent or root_abs != root_res:
            cands.extend([_abspath(json_res.parent / raw_fn), _abspath(root_res / raw_fn)])
        return cands

    def _coco_img_links_and_abs(file_name):
        links = []
        chosen_abs = None
        for cand_abs in _coco_img_cands(file_name):
            if cand_abs.exists():
                if chosen_abs is None:
                    chosen_abs = cand_abs
                for anchor in (root_abs, root_res, json_abs.parent, json_res.parent):
                    links.extend(_collect_link_targets_common(anchor, cand_abs))
        return links, chosen_abs

    def _coco_resolve(file_name):
        try:
            return legacy.resolve_image(file_name, json_res, root_res, index)
        except ValueError:
            if json_abs != json_res or root_abs != root_res:
                return legacy.resolve_image(file_name, json_abs, root_abs, index)
            raise

    anns = doc["annotations"]
    cap = any("caption" in a for a in anns)
    if cap and task in {"auto", "captioning"}:
        captions_by_img = defaultdict(list)
        for ann in anns:
            c = ann.get("caption")
            if isinstance(c, str) and c.strip():
                captions_by_img[ann.get("image_id")].append(c.strip())
        records = []
        extra_roots = [root_res, json_res.parent]
        extra_roots_abs = [root_abs, json_abs.parent]
        for image in doc["images"]:
            links, img_abs = _coco_img_links_and_abs(image["file_name"])
            extra_roots.extend(links)
            path = _coco_resolve(image["file_name"])
            extra_roots.append(path.parent)
            if img_abs is not None:
                extra_roots_abs.append(img_abs.parent)
            split = image.get("split") if image.get("split") in {"train", "val", "test"} else legacy.split_name(json_res)
            gid = _extract_group_id(image)
            records.append({
                "image": str(path),
                "image_abs": str(img_abs or path),
                "split": split,
                "source": str(json_res.parent),
                "captions": sorted(set(captions_by_img.get(image["id"], []))),
                "group_id": gid,
            })
        return {
            "format": "coco_caption",
            "task": "captioning",
            "names": [],
            "records": records,
            "dataset_roots": list(dict.fromkeys(extra_roots)),
            "dataset_roots_abs": list(dict.fromkeys(extra_roots_abs)),
        }

    if not doc.get("categories") or not any("bbox" in a for a in anns):
        return None
    if task not in {"auto", "detection"}:
        raise ValueError("COCO task differs")
    categories = sorted(doc["categories"], key=lambda c: c["id"])
    ids = {c["id"]: i for i, c in enumerate(categories)}
    names = [str(c["name"]) for c in categories]
    raw_boxes = defaultdict(list)
    for ann in anns:
        if ann.get("iscrowd"):
            continue
        cid = ids.get(ann.get("category_id"), -1)
        raw_boxes[ann["image_id"]].append((cid, ann.get("bbox")))
    records = []
    extra_roots = [root_res, json_res.parent]
    extra_roots_abs = [root_abs, json_abs.parent]
    from PIL import Image
    for image in doc["images"]:
        links, img_abs = _coco_img_links_and_abs(image["file_name"])
        extra_roots.extend(links)
        path = _coco_resolve(image["file_name"])
        extra_roots.append(path.parent)
        if img_abs is not None:
            extra_roots_abs.append(img_abs.parent)
        split = image.get("split") if image.get("split") in {"train", "val", "test"} else legacy.split_name(json_res)
        gid = _extract_group_id(image)
        try:
            with Image.open(path) as im:
                w, h = im.size
        except (OSError, ValueError):
            w, h = 0, 0
        boxes = []
        malformed = []
        for cid, bbox in raw_boxes.get(image["id"], []):
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                malformed.append({"raw": repr(bbox), "reason": "expected 4-element COCO bbox [x,y,w,h]"})
                continue
            try:
                bx, by, bw, bh = [float(v) for v in bbox]
            except (TypeError, ValueError):
                malformed.append({"raw": repr(bbox), "reason": "non-numeric COCO bbox"})
                continue
            if not all(math.isfinite(v) for v in (bx, by, bw, bh)) or w <= 0 or h <= 0:
                malformed.append({"raw": repr(bbox), "reason": "non-finite COCO bbox or unreadable image"})
                continue
            boxes.append([float(cid), (bx + bw / 2.0) / w, (by + bh / 2.0) / h, bw / w, bh / h])
        records.append({
            "image": str(path),
            "image_abs": str(img_abs or path),
            "split": split,
            "source": str(json_res.parent),
            "boxes": boxes,
            "pre_malformed": malformed,
            "group_id": gid,
        })
    return {
        "format": "coco_detection",
        "task": "detection",
        "names": names,
        "records": records,
        "dataset_roots": list(dict.fromkeys(extra_roots)),
        "dataset_roots_abs": list(dict.fromkeys(extra_roots_abs)),
    }


def _yolo_label_pair(res_img, abs_img):
    lbl_res = label_path(res_img)
    lbl_abs = label_path(abs_img)
    if lbl_res.is_file():
        return lbl_res, (lbl_abs if lbl_abs.is_file() else lbl_res)
    if lbl_abs.is_file():
        return lbl_abs.resolve(), lbl_abs
    return lbl_res, lbl_abs


def _scan_raw_dataset(source, task="auto"):
    source_abs = _abspath(source)
    source = source_abs.resolve()
    check_explicit_file(source)
    if not source.exists():
        raise ValueError(f"Dataset not found: {source}")
    root_abs = source_abs if source_abs.is_dir() else source_abs.parent
    root = source if source.is_dir() else source.parent
    if source.is_file() and root.name.lower() == "annotations":
        root_abs = root_abs.parent
        root = root.parent

    if source.is_file() and source.suffix.lower() in {".yaml", ".yml"} and task in {"auto", "detection"}:
        parsed = _yolo_yaml_raw(source_abs)
        if parsed:
            if not parsed["records"]:
                raise no_images_error(source)
            return parsed

    if source.is_file() and source.suffix.lower() == ".json":
        doc = json.loads(source.read_text(encoding="utf-8-sig"))
        if doc.get("task") in {"detection", "classification", "captioning"} and isinstance(doc.get("splits"), dict):
            actual_task = doc["task"]
            if task not in {"auto", actual_task}:
                raise ValueError("Manifest task differs from requested task")
            names = doc.get("names", [])
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=lambda k: int(k))]
            names = list(map(str, names))
            records = []
            extra_roots = [root.resolve()]
            extra_roots_abs = [root_abs]
            for split, rows in doc["splits"].items():
                if split not in {"train", "val", "test", "unsplit"}:
                    raise ValueError("Unsupported manifest split")
                for row in rows:
                    raw_img = Path(row["image"]).expanduser()
                    path_abs = _abspath(raw_img if raw_img.is_absolute() else root_abs / raw_img)
                    if not raw_img.is_absolute() and root_abs != root:
                        path_from_resolved = _abspath(root / raw_img)
                        extra_roots.extend(_collect_link_targets_common(root_abs, path_from_resolved))
                        extra_roots.extend(_collect_link_targets_common(root, path_from_resolved))
                        if not path_abs.exists():
                            path_abs = path_from_resolved
                    path_res = path_abs.resolve()
                    extra_roots_abs.append(path_abs.parent)
                    extra_roots.append(path_res.parent)
                    extra_roots.extend(_collect_link_targets_common(root_abs, path_abs))
                    extra_roots.extend(_collect_link_targets_common(root, path_abs))
                    r = {**row, "image": str(path_res), "image_abs": str(path_abs), "source": str(root), "split": split}
                    gid = _clean_group_id(r.get("group_id"))
                    if gid is not None:
                        r["group_id"] = gid
                    else:
                        r.pop("group_id", None)
                    if actual_task == "detection" and "boxes" not in row:
                        r["missing_label"] = True
                    records.append(r)
            if not records:
                raise no_images_error(source)
            return {
                "format": "v2_manifest",
                "task": actual_task,
                "names": names,
                "records": records,
                "dataset_roots": list(dict.fromkeys(extra_roots)),
                "dataset_roots_abs": list(dict.fromkeys(extra_roots_abs)),
            }
        if isinstance(doc.get("images"), list) and isinstance(doc.get("annotations"), list):
            if root_abs.is_dir():
                walked_res, _, _ = _walk_dir_images_and_links(
                    root_abs, skip_names=legacy.SKIP, skip_hidden=True
                )
                index = defaultdict(list)
                seen_by_name = defaultdict(set)
                for res_p in walked_res:
                    key = os.path.normcase(str(res_p))
                    if key not in seen_by_name[res_p.name]:
                        seen_by_name[res_p.name].add(key)
                        index[res_p.name].append(res_p)
            else:
                _, index = legacy.image_index(root)
            parsed = _coco_raw(source, root, index, task, root_abs=root_abs, json_abs=source_abs)
            if parsed:
                return parsed

    dir_links = []
    if root_abs.is_dir():
        walked_res, walked_abs, dir_links = _walk_dir_images_and_links(
            root_abs, skip_names=legacy.SKIP, skip_hidden=True
        )
        image_pairs = sorted(zip(walked_res, walked_abs), key=lambda pair: str(pair[1]))
        images = [res_p for res_p, _ in image_pairs]
        index = defaultdict(list)
        seen_by_name = defaultdict(set)
        for res_p in images:
            key = os.path.normcase(str(res_p))
            if key not in seen_by_name[res_p.name]:
                seen_by_name[res_p.name].add(key)
                index[res_p.name].append(res_p)
    else:
        images, index = legacy.image_index(root)
        image_pairs = [(p, _abspath(p)) for p in images]
    if not image_pairs:
        raise no_images_error(source)
    files = _walk_dir_files_safe(root_abs, skip_names=legacy.SKIP, skip_hidden=True) if source.is_dir() else [source]

    def _rel_for(res_p, abs_p):
        if abs_p.is_relative_to(root_abs):
            return abs_p.relative_to(root_abs)
        if res_p.is_relative_to(root):
            return res_p.relative_to(root)
        return res_p

    if task in {"auto", "detection"}:
        for f in files:
            if f.suffix.lower() in {".yaml", ".yml"}:
                parsed = _yolo_yaml_raw(f)
                if parsed and parsed["records"]:
                    return parsed

    json_datasets = []
    for f in files:
        if f.suffix.lower() != ".json" or f.stem.endswith(("_pred", "_gt")):
            continue
        parsed = _coco_raw(f, root, index, "auto", root_abs=root_abs, json_abs=f)
        if parsed:
            json_datasets.append(parsed)
    if json_datasets:
        kinds = {d["task"] for d in json_datasets}
        if len(kinds) > 1 and task == "auto":
            raise ValueError("Mixed caption/detection inputs found; specify --task or point at a single dataset folder.")
        chosen = task if task != "auto" else next(iter(kinds))
        selected = [d for d in json_datasets if d["task"] == chosen]
        if selected:
            names = selected[0]["names"]
            merged_roots = [root.resolve(), *dir_links]
            merged_roots_abs = [root_abs]
            for d in selected:
                merged_roots.extend(d.get("dataset_roots", []))
                merged_roots_abs.extend(d.get("dataset_roots_abs", []))
            return {
                "format": selected[0]["format"],
                "task": chosen,
                "names": names,
                "records": [r for d in selected for r in d["records"]],
                "dataset_roots": list(dict.fromkeys(merged_roots)),
                "dataset_roots_abs": list(dict.fromkeys(merged_roots_abs)),
            }

    for f in files:
        if f.suffix.lower() != ".csv" or f.with_suffix(".json").exists():
            continue
        with f.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            image_key = next((k for k in ("image", "image_path", "file_name", "filename") if k in fields), None)
            cap_key = next((k for k in ("caption", "text") if k in fields), None)
            if not image_key or not cap_key:
                continue
            grouped = defaultdict(list)
            for row in reader:
                grouped[str(legacy.resolve_image(row[image_key], f, root, index))].append((row[cap_key] or "").strip())
            records = [
                {"image": p, "captions": [c for c in caps if c], "source": str(f.parent), "split": legacy.split_name(f), "group_id": None}
                for p, caps in grouped.items()
            ]
            if task in {"auto", "captioning"} and records:
                return {
                    "format": "caption_csv",
                    "task": "captioning",
                    "names": [],
                    "records": records,
                    "dataset_roots": list(dict.fromkeys([root.resolve(), *dir_links])),
                }

    if task in {"auto", "detection"}:
        pair_labels = [(p, p_abs, *_yolo_label_pair(p, p_abs)) for p, p_abs in image_pairs]
        labelled = [(p, p_abs, lbl, lbl_abs) for p, p_abs, lbl, lbl_abs in pair_labels if lbl.is_file()]
        if labelled:
            max_cid = -1
            for _, _, lbl, _ in labelled:
                for line in lbl.read_text(encoding="utf-8-sig").splitlines():
                    parts = line.split()
                    if len(parts) == 5:
                        try:
                            val = float(parts[0])
                            if math.isfinite(val) and val == int(val) and int(val) >= 0:
                                max_cid = max(max_cid, int(val))
                        except ValueError:
                            pass
            if max_cid >= 0 or task == "detection":
                nc = max(1, max_cid + 1)
                names = [f"class_{i}" for i in range(nc)]
                extra_roots = [root.resolve(), *dir_links]
                extra_roots_abs = [root_abs]
                records = []
                for p, p_abs, lbl, lbl_abs in pair_labels:
                    extra_roots.extend(_collect_link_targets_common(root_abs, p_abs))
                    extra_roots.extend(_collect_link_targets_common(root_abs, lbl_abs))
                    extra_roots_abs.extend([p_abs.parent, lbl_abs.parent])
                    records.append({
                        "image": str(p),
                        "image_abs": str(p_abs),
                        "split": legacy.split_name(_rel_for(p, p_abs)),
                        "source": str(root),
                        "label_file": str(lbl),
                        "label_file_abs": str(lbl_abs),
                        "group_id": None,
                    })
                return {
                    "format": "yolo_dir",
                    "task": "detection",
                    "names": names,
                    "records": records,
                    "dataset_roots": list(dict.fromkeys(extra_roots)),
                    "dataset_roots_abs": list(dict.fromkeys(extra_roots_abs)),
                }

    class_root = root_abs / "train" if (root_abs / "train").is_dir() else root_abs
    classes = sorted(
        p.name for p in class_root.iterdir()
        if p.is_dir() and p.name.lower() not in {"images", "labels", "val", "test", "annotations", "dataset", "src", "pretrained"}
        and bool(_walk_dir_images_and_links(p, skip_names=legacy.SKIP, skip_hidden=True)[0])
    )
    if len(classes) >= 2 and task in {"auto", "classification"}:
        records = []
        extra_roots = [root.resolve(), *dir_links]
        extra_roots_abs = [root_abs]
        for path, p_abs in image_pairs:
            relative = _rel_for(path, p_abs)
            cls = next((part for part in relative.parts[:-1] if part in classes), None)
            if cls:
                extra_roots.extend(_collect_link_targets_common(root_abs, p_abs))
                extra_roots_abs.append(p_abs.parent)
                records.append({
                    "image": str(path),
                    "image_abs": str(p_abs),
                    "class_id": classes.index(cls),
                    "source": str(root),
                    "split": legacy.split_name(relative),
                    "group_id": None,
                })
        return {
            "format": "image_folder",
            "task": "classification",
            "names": classes,
            "records": records,
            "dataset_roots": list(dict.fromkeys(extra_roots)),
            "dataset_roots_abs": list(dict.fromkeys(extra_roots_abs)),
        }

    if task not in {"auto", "reconstruction"}:
        raise ValueError(f"No labels for requested task {task}.")
    extra_roots = [root.resolve(), *dir_links]
    extra_roots_abs = [root_abs]
    records = []
    for p, p_abs in image_pairs:
        extra_roots.extend(_collect_link_targets_common(root_abs, p_abs))
        extra_roots_abs.append(p_abs.parent)
        records.append({
            "image": str(p),
            "image_abs": str(p_abs),
            "source": str(root),
            "split": legacy.split_name(_rel_for(p, p_abs)),
            "group_id": None,
        })
    return {
        "format": "raw_images",
        "task": "reconstruction",
        "names": [],
        "records": records,
        "dataset_roots": list(dict.fromkeys(extra_roots)),
        "dataset_roots_abs": list(dict.fromkeys(extra_roots_abs)),
    }


def _record_image_info(record):
    from PIL import Image

    path = Path(record["image"])
    sha256 = legacy.file_hash(path) if path.is_file() else None
    width, height, unreadable_err = 0, 0, None
    try:
        with Image.open(path) as im:
            im.load()
            width, height = im.size
    except (OSError, ValueError) as exc:
        unreadable_err = str(exc)
    return sha256, width, height, unreadable_err


def audit_dataset(source, task="auto"):
    """Read-only preflight audit: SHA-256 duplicates, cross-split leakage, and OD bbox/class stats."""
    scanned = _scan_raw_dataset(source, task=task)
    actual_task = scanned["task"]
    names = list(scanned["names"])
    nc = len(names)
    records = scanned["records"]

    split_order = ("train", "val", "test", "unsplit")
    counts = {s: 0 for s in split_order}
    unreadable_images = []
    by_split_hash = {s: defaultdict(list) for s in split_order}
    explicit_hash_splits = defaultdict(lambda: defaultdict(list))
    explicit_group_splits = defaultdict(lambda: defaultdict(list))

    # Detection accumulators
    missing_items = []
    empty_items = []
    malformed_items = []
    oob_items = []
    degenerate_items = []
    invalid_cls_items = []
    total_cls_counts = Counter()
    split_cls_counts = {s: Counter() for s in split_order}
    coco_sizes = Counter()
    norm_sizes = Counter()
    coco_eval_boxes = 0
    norm_eval_boxes = 0
    total_boxes = 0

    for r in records:
        split = r.get("split", "unsplit")
        if split not in counts:
            split = "unsplit"
        counts[split] += 1
        img_str = str(r["image"])
        sha256, img_w, img_h, unreadable_err = _record_image_info(r)
        r["_sha256"] = sha256
        r["_size"] = (img_w, img_h)
        if unreadable_err is not None:
            unreadable_images.append({"split": split, "image": img_str, "error": unreadable_err})
        if sha256 is not None:
            by_split_hash[split][sha256].append(img_str)
            if split in {"train", "val", "test"}:
                explicit_hash_splits[sha256][split].append(img_str)
        gid = _clean_group_id(r.get("group_id"))
        if gid is not None and split in {"train", "val", "test"}:
            explicit_group_splits[gid][split].append(img_str)

        if actual_task == "detection":
            for bad in r.get("pre_malformed", []):
                malformed_items.append({"split": split, "image": img_str, "label": "coco_json", **bad})
            if "label_file" in r:
                lbl = Path(r["label_file"])
                if not lbl.exists():
                    missing_items.append({"split": split, "image": img_str, "label": str(lbl)})
                    continue
                try:
                    raw_lines = lbl.read_text(encoding="utf-8-sig").splitlines()
                except (OSError, UnicodeError) as exc:
                    malformed_items.append({"split": split, "image": img_str, "label": str(lbl), "reason": str(exc)})
                    continue
                non_empty = [(idx + 1, line) for idx, line in enumerate(raw_lines) if line.strip()]
                if not non_empty:
                    empty_items.append({"split": split, "image": img_str, "label": str(lbl)})
                    continue
                parsed_boxes = []
                for line_no, text in non_empty:
                    tokens = text.split()
                    if len(tokens) != 5:
                        malformed_items.append({"split": split, "image": img_str, "label": str(lbl), "line": line_no, "raw": text, "reason": "expected 5 columns"})
                        continue
                    try:
                        vals = [float(t) for t in tokens]
                    except ValueError:
                        malformed_items.append({"split": split, "image": img_str, "label": str(lbl), "line": line_no, "raw": text, "reason": "non-numeric token"})
                        continue
                    if any(not math.isfinite(v) for v in vals):
                        malformed_items.append({"split": split, "image": img_str, "label": str(lbl), "line": line_no, "raw": text, "reason": "non-finite value"})
                        continue
                    parsed_boxes.append((line_no, str(lbl), vals))
            else:
                if r.get("missing_label"):
                    missing_items.append({"split": split, "image": img_str, "label": "manifest"})
                    continue
                boxes = r.get("boxes") or []
                if not boxes and not r.get("pre_malformed"):
                    empty_items.append({"split": split, "image": img_str, "label": "manifest"})
                    continue
                parsed_boxes = []
                for idx, box in enumerate(boxes):
                    if not isinstance(box, (list, tuple)) or len(box) != 5:
                        malformed_items.append({"split": split, "image": img_str, "label": "manifest", "line": idx + 1, "raw": repr(box), "reason": "expected 5 elements"})
                        continue
                    try:
                        vals = [float(v) for v in box]
                    except (TypeError, ValueError):
                        malformed_items.append({"split": split, "image": img_str, "label": "manifest", "line": idx + 1, "raw": repr(box), "reason": "non-numeric value"})
                        continue
                    if any(not math.isfinite(v) for v in vals):
                        malformed_items.append({"split": split, "image": img_str, "label": "manifest", "line": idx + 1, "raw": repr(box), "reason": "non-finite value"})
                        continue
                    parsed_boxes.append((idx + 1, "manifest", vals))

            for line_no, lbl_ref, (cls_val, x, y, w, h) in parsed_boxes:
                total_boxes += 1
                is_int_cls = (cls_val == int(cls_val))
                cls_id = int(cls_val) if is_int_cls else None
                if not is_int_cls or cls_id < 0 or cls_id >= nc:
                    invalid_cls_items.append({
                        "split": split, "image": img_str, "label": lbl_ref,
                        "line": line_no, "class_id": cls_val, "nc": nc,
                    })
                else:
                    total_cls_counts[cls_id] += 1
                    split_cls_counts[split][cls_id] += 1
                if w <= 0.0 or h <= 0.0:
                    degenerate_items.append({
                        "split": split, "image": img_str, "label": lbl_ref,
                        "line": line_no, "box": [cls_val, x, y, w, h],
                    })
                if (not all(0.0 <= v <= 1.0 for v in (x, y, w, h))
                        or min(x - w / 2.0, y - h / 2.0) < -1e-6
                        or max(x + w / 2.0, y + h / 2.0) > 1.0 + 1e-6):
                    oob_items.append({
                        "split": split, "image": img_str, "label": lbl_ref,
                        "line": line_no, "box": [cls_val, x, y, w, h],
                    })
                if w > 0.0 and h > 0.0:
                    norm_area = w * h
                    norm_eval_boxes += 1
                    if norm_area < 0.01:
                        norm_sizes["small"] += 1
                    elif norm_area < 0.09:
                        norm_sizes["medium"] += 1
                    else:
                        norm_sizes["large"] += 1
                    if img_w > 0 and img_h > 0:
                        px_area = (w * img_w) * (h * img_h)
                        coco_eval_boxes += 1
                        if px_area < 32 * 32:
                            coco_sizes["small"] += 1
                        elif px_area < 96 * 96:
                            coco_sizes["medium"] += 1
                        else:
                            coco_sizes["large"] += 1

    intra_by_split = {}
    all_intra_groups = []
    all_intra_files = []
    for s in split_order:
        groups = []
        dup_files = []
        for digest_val, paths in sorted(by_split_hash[s].items()):
            ordered = sorted(paths)
            if len(ordered) > 1:
                g = {"sha256": digest_val, "kept": ordered[0], "duplicates": ordered[1:], "count": len(ordered)}
                groups.append(g)
                dup_files.extend(ordered[1:])
                all_intra_groups.append({"split": s, **g})
                all_intra_files.extend(ordered[1:])
        intra_by_split[s] = {
            "duplicate_count": len(dup_files),
            "group_count": len(groups),
            "groups": groups,
            "duplicate_files": dup_files,
        }

    sha256_leaks = []
    leaked_files_set = set()
    leak_pairs = set()
    for digest_val, split_map in sorted(explicit_hash_splits.items()):
        if len(split_map) >= 2:
            splits_present = sorted(split_map, key=lambda x: ("train", "val", "test").index(x))
            pairs = [f"{splits_present[i]}<->{splits_present[j]}" for i in range(len(splits_present)) for j in range(i + 1, len(splits_present))]
            leak_pairs.update(pairs)
            files_flat = sorted(p for s in splits_present for p in split_map[s])
            leaked_files_set.update(files_flat)
            sha256_leaks.append({
                "sha256": digest_val,
                "splits": splits_present,
                "split_pairs": pairs,
                "files_by_split": {s: sorted(split_map[s]) for s in splits_present},
                "files": files_flat,
            })

    group_id_leaks = []
    for gid, split_map in sorted(explicit_group_splits.items()):
        if len(split_map) >= 2:
            splits_present = sorted(split_map, key=lambda x: ("train", "val", "test").index(x))
            pairs = [f"{splits_present[i]}<->{splits_present[j]}" for i in range(len(splits_present)) for j in range(i + 1, len(splits_present))]
            leak_pairs.update(pairs)
            files_flat = sorted(p for s in splits_present for p in split_map[s])
            leaked_files_set.update(files_flat)
            group_id_leaks.append({
                "group_id": gid,
                "splits": splits_present,
                "split_pairs": pairs,
                "files_by_split": {s: sorted(split_map[s]) for s in splits_present},
                "files": files_flat,
            })

    result = {
        "schema_version": 1,
        "read_only": True,
        "format": scanned["format"],
        "task": actual_task,
        "source": str(Path(source).expanduser().resolve()),
        "names": names,
        "nc": nc,
        "total_images": len(records),
        "counts": counts,
        "unreadable_images": unreadable_images,
        "intra_split_duplicates": {
            "total_duplicate_files": len(all_intra_files),
            "total_groups": len(all_intra_groups),
            "by_split": intra_by_split,
            "groups": all_intra_groups,
            "duplicate_files": sorted(all_intra_files),
        },
        "cross_split_leakage": {
            "has_leakage": bool(sha256_leaks or group_id_leaks),
            "sha256_leak_count": len(sha256_leaks),
            "group_id_leak_count": len(group_id_leaks),
            "sha256_leaks": sha256_leaks,
            "group_id_leaks": group_id_leaks,
            "split_pairs": sorted(leak_pairs),
            "leaked_files": sorted(leaked_files_set),
        },
    }

    if actual_task == "detection":
        def pack_issue(items):
            return {"count": len(items), "files": sorted({it["image"] for it in items}), "items": items}

        counts_list = [total_cls_counts[i] for i in range(nc)]
        nonzero = [c for c in counts_list if c > 0]
        if not counts_list:
            imb_ratio = None
        elif len(counts_list) == 1:
            imb_ratio = 1.0 if counts_list[0] > 0 else None
        elif min(counts_list) == 0:
            imb_ratio = None
        else:
            imb_ratio = float(max(counts_list) / min(counts_list))
        nonzero_imb = float(max(nonzero) / min(nonzero)) if nonzero else None

        def pack_sizes(counter, denom):
            d = max(1, denom)
            return {
                "small": int(counter["small"]),
                "medium": int(counter["medium"]),
                "large": int(counter["large"]),
                "evaluated_boxes": int(denom),
                "small_fraction": float(counter["small"] / d) if denom else 0.0,
                "medium_fraction": float(counter["medium"] / d) if denom else 0.0,
                "large_fraction": float(counter["large"] / d) if denom else 0.0,
            }

        result.update({
            "missing_labels": pack_issue(missing_items),
            "empty_labels": pack_issue(empty_items),
            "malformed_labels": pack_issue(malformed_items),
            "out_of_bounds_boxes": pack_issue(oob_items),
            "degenerate_boxes": pack_issue(degenerate_items),
            "invalid_class_ids": pack_issue(invalid_cls_items),
            "class_counts": {
                "total": {names[i]: int(total_cls_counts[i]) for i in range(nc)},
                "by_split": {s: {names[i]: int(split_cls_counts[s][i]) for i in range(nc)} for s in split_order},
                "total_objects": int(sum(total_cls_counts.values())),
            },
            "zero_count_classes": [names[i] for i in range(nc) if total_cls_counts[i] == 0],
            "imbalance_ratio": imb_ratio,
            "nonzero_imbalance_ratio": nonzero_imb,
            "object_size_distribution": {
                "total_boxes": int(total_boxes),
                "coco_pixel_area": pack_sizes(coco_sizes, coco_eval_boxes),
                "normalized_area": pack_sizes(norm_sizes, norm_eval_boxes),
                "definitions": {
                    "coco_pixel_area": "small: < 32^2 px, medium: [32^2, 96^2) px, large: >= 96^2 px",
                    "normalized_area": "small: w*h < 0.01, medium: [0.01, 0.09), large: w*h >= 0.09",
                },
            },
        })
    return result


def dedup_dataset(source, destination, task="auto"):
    """Write a deduplicated, leak-free copy of a dataset without modifying raw files."""
    import yaml
    from yolo_tuner.pipeline import write_json

    source_abs = _abspath(source)
    source_path = source_abs.resolve()
    dest_abs = _abspath(destination)
    dest_path = dest_abs.resolve()
    scanned = _scan_raw_dataset(source_abs, task=task)
    src_root_abs = source_abs if source_abs.is_dir() else source_abs.parent
    src_root = source_path if source_path.is_dir() else source_path.parent
    roots = [src_root.resolve(), *(Path(p).resolve() for p in scanned.get("dataset_roots", []))]
    roots_abs = [src_root_abs, *(_abspath(p) for p in scanned.get("dataset_roots_abs", []))]
    for rec in scanned["records"]:
        if rec.get("image"):
            img_abs = _abspath(rec["image_abs"] if rec.get("image_abs") else rec["image"])
            roots.append(Path(rec["image"]).expanduser().resolve().parent)
            roots_abs.append(img_abs.parent)
            roots.extend(_collect_link_targets_common(src_root_abs, img_abs))
            roots.extend(_collect_link_targets_common(src_root, img_abs))
        if rec.get("label_file"):
            lbl_abs = _abspath(rec["label_file_abs"] if rec.get("label_file_abs") else rec["label_file"])
            roots.append(Path(rec["label_file"]).expanduser().resolve().parent)
            roots_abs.append(lbl_abs.parent)
            roots.extend(_collect_link_targets_common(src_root_abs, lbl_abs))
            roots.extend(_collect_link_targets_common(src_root, lbl_abs))
    if scanned.get("format") in {"yolo_dir", "image_folder", "raw_images", "coco_detection", "coco_caption", "caption_csv"}:
        for walk_base in list(dict.fromkeys([src_root_abs, src_root, *(_abspath(p) for p in scanned.get("dataset_roots", []))])):
            if walk_base.is_dir():
                _, _, dir_links = _walk_dir_images_and_links(walk_base)
                roots.extend(dir_links)
    roots = list(dict.fromkeys(Path(r).resolve() for r in roots))
    roots_abs = list(dict.fromkeys(_abspath(r) for r in roots_abs))
    for r in roots:
        if dest_path == r or dest_path.is_relative_to(r):
            raise ValueError(f"Refusing in-place or nested destination inside raw dataset: {dest_path}")
    for r_abs in roots_abs:
        if dest_abs == r_abs or dest_abs.is_relative_to(r_abs):
            raise ValueError(f"Refusing in-place or nested destination inside raw dataset: {dest_path}")
    if dest_path.exists():
        if not dest_path.is_dir() or any(dest_path.iterdir()):
            raise ValueError(f"Destination already exists and is non-empty: {dest_path}")

    before_audit = audit_dataset(source_path, task=task)
    actual_task = scanned["task"]
    names = list(scanned["names"])
    priority = {"train": 0, "val": 1, "test": 2, "unsplit": 3}
    ordered = sorted(scanned["records"], key=lambda r: (priority.get(r.get("split", "unsplit"), 3), str(r["image"])))

    seen_hashes = {}
    seen_groups = {}
    kept = []
    removed = []
    for r in ordered:
        img = Path(r["image"])
        if not img.is_file():
            continue
        digest_val = legacy.file_hash(img)
        split = r.get("split", "unsplit")
        if split not in priority:
            split = "unsplit"
        if digest_val in seen_hashes:
            kept_rec = seen_hashes[digest_val]
            reason = "intra_split_duplicate" if kept_rec["split"] == split else "cross_split_leakage"
            removed.append({
                "image": str(img),
                "split": split,
                "sha256": digest_val,
                "reason": reason,
                "kept_image": str(kept_rec["image"]),
                "kept_split": kept_rec["split"],
            })
            continue
        gid_key = _clean_group_id(r.get("group_id"))
        if gid_key is not None and gid_key in seen_groups and seen_groups[gid_key]["split"] != split:
            kept_grp = seen_groups[gid_key]
            removed.append({
                "image": str(img),
                "split": split,
                "sha256": digest_val,
                "group_id": gid_key,
                "reason": "cross_split_group_leakage",
                "kept_image": str(kept_grp["image"]),
                "kept_split": kept_grp["split"],
            })
            continue
        rec_copy = {**r, "split": split, "sha256": digest_val}
        if gid_key is not None:
            rec_copy["group_id"] = gid_key
        else:
            rec_copy.pop("group_id", None)
        seen_hashes[digest_val] = rec_copy
        if gid_key is not None and gid_key not in seen_groups:
            seen_groups[gid_key] = rec_copy
        kept.append(rec_copy)

    if not kept:
        raise ValueError("No valid images remaining after deduplication")
    had_explicit = any(r.get("split") in {"train", "val"} for r in scanned["records"])
    kept_splits = Counter(r.get("split", "unsplit") for r in kept)
    if had_explicit and (kept_splits["train"] == 0 or kept_splits["val"] == 0):
        raise ValueError("Deduplication left train or val split empty")

    dest_path.mkdir(parents=True, exist_ok=True)
    used_names = defaultdict(set)
    used_stems = defaultdict(set)

    def unique_dest_name(split_name, src_file, sha256_val, *, stem_unique=False):
        candidate = src_file.name
        stem_key = Path(candidate).stem.lower()
        if candidate.lower() in used_names[split_name] or (stem_unique and stem_key in used_stems[split_name]):
            candidate = f"{src_file.stem}_{sha256_val[:8]}{src_file.suffix.lower()}"
            suffix_idx = 1
            while candidate.lower() in used_names[split_name] or (
                stem_unique and Path(candidate).stem.lower() in used_stems[split_name]
            ):
                candidate = f"{src_file.stem}_{sha256_val[:8]}_{suffix_idx}{src_file.suffix.lower()}"
                suffix_idx += 1
        used_names[split_name].add(candidate.lower())
        used_stems[split_name].add(Path(candidate).stem.lower())
        return candidate

    if (
        scanned["format"] == "v2_manifest"
        or actual_task in {"classification", "captioning"}
        or (kept_splits["train"] == 0 and scanned["format"] != "yolo_dir")
    ):
        manifest_splits = {"train": [], "val": [], "test": [], "unsplit": []}
        for r in kept:
            split = r.get("split", "unsplit")
            if split not in manifest_splits:
                split = "unsplit"
            src_img = Path(r["image"])
            fname = unique_dest_name(split, src_img, r["sha256"])
            rel_img = Path("images") / split / fname
            dst_img = dest_path / rel_img
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_img, dst_img)
            row = {"image": rel_img.as_posix()}
            for k in ("group_id", "class_id", "captions", "boxes"):
                if k in r and r[k] is not None:
                    row[k] = r[k]
            manifest_splits[split].append(row)
        manifest_splits = {k: v for k, v in manifest_splits.items() if v or k in {"train", "val", "test"}}
        entrypoint = dest_path / "manifest.json"
        write_json(entrypoint, {"schema_version": 1, "task": actual_task, "names": names, "splits": manifest_splits})
    else:
        has_test = False
        for r in kept:
            split = r.get("split", "unsplit")
            if split == "test":
                has_test = True
            src_img = Path(r["image"])
            fname = unique_dest_name(split, src_img, r["sha256"], stem_unique=True)
            dst_img = dest_path / "images" / split / fname
            dst_lbl = (dest_path / "labels" / split / fname).with_suffix(".txt")
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_img, dst_img)
            if "label_file" in r and Path(r["label_file"]).is_file():
                shutil.copyfile(Path(r["label_file"]), dst_lbl)
            elif "boxes" in r:
                lines = []
                for box in r["boxes"]:
                    cls_v = int(box[0]) if box[0] == int(box[0]) else box[0]
                    lines.append(" ".join([str(cls_v), *(str(v) for v in box[1:])]))
                dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if kept_splits["train"] > 0:
            spec = {"path": dest_path.as_posix(), "train": "images/train", "names": names}
            if kept_splits["val"] > 0:
                spec["val"] = "images/val"
            if has_test:
                spec["test"] = "images/test"
            entrypoint = dest_path / "data.yaml"
            entrypoint.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
        else:
            entrypoint = dest_path

    after_audit = audit_dataset(entrypoint, task=actual_task)
    audit_payload = {
        "source": str(source_path),
        "destination": str(dest_path),
        "entrypoint": str(entrypoint),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "removed_files": removed,
        "before_audit": before_audit,
        "after_audit": after_audit,
    }
    audit_file = dest_path / "dedup_audit.json"
    write_json(audit_file, audit_payload)
    write_json(dest_path / "data_audit.json", after_audit)
    return {
        "source": str(source_path),
        "destination": str(dest_path),
        "entrypoint": str(entrypoint),
        "audit_file": str(audit_file),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "removed_files": removed,
        "before_audit": before_audit,
        "after_audit": after_audit,
    }


def _verify_no_explicit_cross_split_sha256(records):
    seen = {}
    for r in records:
        split = r.get("split")
        if split not in {"train", "val", "test"}:
            continue
        p = Path(r["image"])
        if not p.is_file():
            continue
        digest_val = r.get("sha256") or legacy.file_hash(p)
        r["sha256"] = digest_val
        if digest_val in seen and seen[digest_val] != split:
            raise ValueError("Exact image content crosses split boundary")
        seen[digest_val] = split


def check_manifest(data):
    task, names = data["task"], data["names"]
    if task not in {"detection", "classification", "captioning"}:
        raise ValueError("Only genuine supervised detection/classification/captioning are supported")
    if task != "captioning" and (not names or len(set(names)) != len(names)):
        raise ValueError("Names must be a nonempty unique taxonomy")
    if task == "classification" and len(names) < 2:
        raise ValueError("Single-class classification accuracy is degenerate; need at least two classes")
    hashes, groups = {}, {}
    for split, records in data["splits"].items():
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        for r in records:
            if r["sha256"] in hashes and hashes[r["sha256"]] != split:
                raise ValueError("Exact image content crosses split boundary")
            hashes[r["sha256"]] = split
            group = _clean_group_id(r.get("group_id"))
            if group is not None:
                if group in groups and groups[group] != split:
                    raise ValueError("Declared scene/video/subject group crosses splits")
                groups[group] = split
            if task == "classification" and (type(r.get("class_id")) is not int or not 0 <= r["class_id"] < len(names)):
                raise ValueError("Invalid class_id/taxonomy")
            if task == "captioning" and (not r.get("captions") or any(not isinstance(c, str) or not c.strip() for c in r["captions"])):
                raise ValueError("Caption labels must be nonempty strings")
            if task == "detection":
                for box in r.get("boxes", []):
                    if (len(box) != 5 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                            or box[0] != int(box[0]) or not 0 <= box[0] < len(names)):
                        raise ValueError("Detection boxes must be normalized [class,xc,yc,w,h]")
                    _, x, y, w, h = box
                    if not 0 < w <= 1 or not 0 < h <= 1 or min(x - w / 2, y - h / 2) < -1e-6 or max(x + w / 2, y + h / 2) > 1 + 1e-6:
                        raise ValueError("Normalized box outside image bounds")
    if not data["splits"].get("train") or not data["splits"].get("val"):
        raise ValueError("Independent nonempty train/validation required")
    if task == "detection" and any(not any(r.get("boxes") for r in data["splits"][s]) for s in ("train", "val")):
        raise ValueError("Detection mAP requires real positive boxes in train and validation; no reconstruction fallback")
    data["grouping_status"] = "declared_groups_verified_disjoint" if groups else "true_groups_not_supplied; near-duplicate scene leakage not excluded"
    data["validation_ready"] = True
    return data


def finalize(source, task, records, names, seed):
    _verify_no_explicit_cross_split_sha256(records)
    for r in records:
        gid = _clean_group_id(r.get("group_id"))
        if gid is not None:
            r["group_id"] = gid
        else:
            r.pop("group_id", None)
    known = {r["group_id"] for r in records if r.get("group_id") is not None}
    if not known:
        return check_manifest(legacy.finalize(source, task, records, names, [], seed))
    from PIL import Image
    # Explicit groups are never split by the legacy image-level splitter.
    for r in records:
        with Image.open(r["image"]) as im:
            im.load()
        r["sha256"] = legacy.file_hash(Path(r["image"]))
    splits = {s: [r for r in records if r["split"] == s] for s in ("train", "val", "test")}
    unsplit = [r for r in records if r["split"] == "unsplit"]
    if not splits["val"]:
        unsplit += splits["train"]
        splits["train"] = []
    if unsplit:
        grouped = defaultdict(list)
        for r in unsplit:
            gid = _clean_group_id(r.get("group_id"))
            key = gid if gid is not None else r["sha256"]
            grouped[key].append(r)
        keys = sorted(grouped)
        random.Random(seed).shuffle(keys)
        nval = max(1, round(len(keys) * 0.2))
        if len(keys) < 2:
            raise ValueError("At least two independent groups required to derive validation")
        for i, key in enumerate(keys):
            split = "val" if i < nval else "train"
            for r in grouped[key]:
                r["split"] = split
                splits[split].append(r)
    data = {"schema_version": 1, "task": task, "source": str(source), "seed": seed, "names": names,
            "splits": splits, "warnings": [], "sampled_image_sizes": [],
            "counts": {s: len(rs) for s, rs in splits.items()},
            "captions": {s: sum(len(r.get("captions", [])) for r in rs) for s, rs in splits.items()}}
    return check_manifest(data)


def discover(source, seed=101, task="auto"):
    source = Path(source).expanduser().resolve()
    check_explicit_file(source)
    root = source.parent
    if source.is_file() and source.suffix.lower() in {".yaml", ".yml"} and task in {"auto", "detection"}:
        parsed = legacy.yolo_yaml(source)
        if parsed:
            records, names = parsed
            return finalize(source, "detection", records, names, seed)
    if source.is_file() and source.suffix.lower() == ".json":
        doc = json.loads(source.read_text(encoding="utf-8-sig"))
        if doc.get("task") in {"detection", "classification", "captioning"} and isinstance(doc.get("splits"), dict):
            actual_task = doc["task"]
            if task not in {"auto", actual_task}:
                raise ValueError("Manifest task differs from requested task")
            records = []
            for split, rows in doc["splits"].items():
                if split not in {"train", "val", "test", "unsplit"}:
                    raise ValueError("Unsupported manifest split")
                for row in rows:
                    path = Path(row["image"])
                    path = path if path.is_absolute() else root / path
                    r = {**row, "image": str(path.resolve()), "source": str(root), "split": split}
                    gid = _clean_group_id(r.get("group_id"))
                    if gid is not None:
                        r["group_id"] = gid
                    else:
                        r.pop("group_id", None)
                    records.append(r)
            names = doc.get("names", [])
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=lambda k: int(k))]
            return finalize(source, actual_task, records, list(map(str, names)), seed)
        if isinstance(doc.get("images"), list) and isinstance(doc.get("annotations"), list):
            index = defaultdict(list)
            cap = any("caption" in a for a in doc["annotations"])
            actual_task = "captioning" if cap and task in {"auto", "captioning"} else "detection"
            if task not in {"auto", actual_task}:
                raise ValueError("COCO task differs")
            parsed = legacy.caption_records(source, root, index) if actual_task == "captioning" else legacy.coco_records(source, root, index)
            if parsed:
                records, names = (parsed, []) if actual_task == "captioning" else parsed
                by_path = {str(legacy.resolve_image(i["file_name"], source, root, index)): i for i in doc["images"]}
                for r in records:
                    metadata = by_path[r["image"]]
                    if metadata.get("split") in {"train", "val", "test"}:
                        r["split"] = metadata["split"]
                    gid = _extract_group_id(metadata)
                    if gid is not None:
                        r["group_id"] = gid
                return finalize(source, actual_task, records, names, seed)
    try:
        try:
            pre_audit = audit_dataset(source, task=task)
        except DatasetInputError:
            raise
        except ValueError:
            pre_audit = None
        if pre_audit and pre_audit["cross_split_leakage"]["sha256_leaks"]:
            raise ValueError("Exact image content crosses split boundary")
        return check_manifest(legacy.discover(source, seed, task))
    except ValueError as exc:
        if "No local image files found" in str(exc):
            raise no_images_error(source) from exc
        raise
