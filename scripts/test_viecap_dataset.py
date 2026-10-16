"""Audit local caption datasets and optionally run YOLO inference, never invent labels.

Caption annotations do not define detection ground truth. Inference results are
operational measurements only; this script cannot calculate detection mAP.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

from PIL import Image


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def audit(root):
    datasets, all_images = {}, {}
    for folder in sorted((root / "dataset").iterdir()):
        if not folder.is_dir():
            continue
        json_files = sorted(folder.glob("*.json"))
        if not json_files:
            continue
        splits, split_files = {}, {}
        for file in json_files:
            doc = json.loads(file.read_text(encoding="utf-8-sig"))
            if not isinstance(doc, dict) or "images" not in doc or "annotations" not in doc:
                continue
            split = next((s for s in ("train", "val", "test") if s in file.stem), file.stem)
            images = doc["images"]
            annotations = doc["annotations"]
            ids = [image["id"] for image in images]
            names = [image["file_name"] for image in images]
            known_ids = set(ids)
            captions = defaultdict(list)
            orphans, empty, bbox_count, category_count = [], [], 0, 0
            annotation_keys = set()
            for ann in annotations:
                annotation_keys.update(ann)
                bbox_count += "bbox" in ann
                category_count += "category_id" in ann
                if ann.get("image_id") not in known_ids:
                    orphans.append(ann.get("image_id"))
                cap = ann.get("caption")
                if not isinstance(cap, str) or not cap.strip():
                    empty.append(ann.get("image_id"))
                else:
                    captions[ann["image_id"]].append(cap.strip())
            missing, dimensions = [], []
            paths = []
            for entry in images:
                name = entry["file_name"]
                candidates = [folder / "images" / name, folder / "image" / name]
                path = next((p for p in candidates if p.is_file()), None)
                if path is None:
                    missing.append(name)
                    continue
                path = path.resolve()
                paths.append(path)
                all_images[path] = (folder.name, name)
                dimensions.append((name, entry.get("width"), entry.get("height")))
            counts = Counter(len(captions[i]) for i in known_ids)
            splits[split] = {
                "annotation_file": str(file), "images": len(images), "captions": len(annotations),
                "captions_per_image_distribution": dict(sorted(counts.items())),
                "non_five_caption_images": [{"image_id": i, "captions": len(captions[i])}
                                             for i in known_ids if len(captions[i]) != 5],
                "images_without_captions": [i for i in known_ids if not captions[i]],
                "missing_images": missing, "orphan_caption_image_ids": orphans,
                "empty_captions": empty, "duplicate_image_ids": [i for i,c in Counter(ids).items() if c > 1],
                "duplicate_file_names": [i for i,c in Counter(names).items() if c > 1],
                "duplicate_caption_instances": sum(len(v)-len(set(v)) for v in captions.values()),
                "annotations_with_bbox": bbox_count, "annotations_with_category_id": category_count,
                "annotation_keys": sorted(annotation_keys),
                "caption_words_mean": statistics.mean(len(c.split()) for v in captions.values() for c in v) if captions else 0,
            }
            split_files[split] = {"paths": paths, "names": set(names), "dimensions": dimensions}
        overlaps = []
        keys = list(split_files)
        for i, a in enumerate(keys):
            for b in keys[i+1:]:
                shared = sorted(split_files[a]["names"] & split_files[b]["names"])
                overlaps.append({"splits": [a,b], "same_file_names": shared})
        datasets[folder.name] = {"splits": splits, "file_name_overlaps": overlaps,
                                 "local_images": len(list((folder / "images").glob("*"))) if (folder / "images").exists() else 0}
    print(f"Checking {len(all_images)} referenced image files...", flush=True)
    hashes = defaultdict(list)
    image_info, corrupt, dimension_mismatches = {}, [], []
    total_bytes = 0
    for index, (path, (dataset, name)) in enumerate(all_images.items()):
        content = path.read_bytes()
        total_bytes += len(content)
        hashes[hashlib.sha256(content).hexdigest()].append(path)
        try:
            with Image.open(path) as im:
                width, height = im.size
                mode = im.mode
                im.verify()
            # verify alone does not fully decode all pixel data.
            with Image.open(path) as im:
                im.load()
            image_info[path] = {"width": width, "height": height, "mode": mode}
        except Exception as exc:
            corrupt.append({"image": str(path), "error": str(exc)})
        if (index + 1) % 1000 == 0:
            print(f"Decoded {index+1}/{len(all_images)} images", flush=True)
    for dataset, info in datasets.items():
        for split, stats in info["splits"].items():
            doc = json.loads(Path(stats["annotation_file"]).read_text(encoding="utf-8-sig"))
            for image in doc["images"]:
                path = (root / "dataset" / dataset / "images" / image["file_name"]).resolve()
                actual = image_info.get(path)
                if actual and (image.get("width") != actual["width"] or image.get("height") != actual["height"]):
                    dimension_mismatches.append({"dataset": dataset, "split": split, "image": image["file_name"],
                                                 "annotated": [image.get("width"), image.get("height")],
                                                 "actual": [actual["width"], actual["height"]]})
    memberships = defaultdict(set)
    for dataset, info in datasets.items():
        for split, stats in info["splits"].items():
            doc = json.loads(Path(stats["annotation_file"]).read_text(encoding="utf-8-sig"))
            for image in doc["images"]:
                memberships[(root / "dataset" / dataset / "images" / image["file_name"]).resolve()].add(f"{dataset}/{split}")
    duplicate_groups = [{"sha256": h, "images": [str(p) for p in paths],
                         "memberships": sorted(set().union(*(memberships[p] for p in paths)))}
                        for h, paths in hashes.items() if len(paths) > 1]
    caption_predictions = []
    for path in sorted((root / "annotations").glob("*_pred.json")):
        pred = json.loads(path.read_text(encoding="utf-8-sig"))
        gt_path = path.with_name(path.name.replace("_pred.json", "_gt.json"))
        gt = json.loads(gt_path.read_text(encoding="utf-8-sig")) if gt_path.exists() else {}
        gt_ids = {str(i["id"]) for i in gt.get("images", [])}
        pred_ids = [str(p["image_id"]) for p in pred]
        inferred_splits = []
        for dataset, info in datasets.items():
            for split, stats in info["splits"].items():
                doc = json.loads(Path(stats["annotation_file"]).read_text(encoding="utf-8-sig"))
                split_names = {i["file_name"] for i in doc["images"]}
                if set(pred_ids) == split_names:
                    inferred_splits.append(f"{dataset}/{split}")
        caption_predictions.append({"file": str(path), "predictions": len(pred), "ground_truth_images": len(gt_ids),
                                    "missing_predictions": sorted(gt_ids-set(pred_ids)),
                                    "unknown_prediction_ids": sorted(set(pred_ids)-gt_ids),
                                    "duplicate_prediction_ids": [i for i,c in Counter(pred_ids).items() if c>1],
                                    "matches_dataset_split_by_file_name": inferred_splits,
                                    "provenance": "existing repository output, not generated by this test"})
    bbox_count = sum(s["annotations_with_bbox"] for d in datasets.values() for s in d["splits"].values())
    report = {"root": str(root), "datasets": datasets, "unique_referenced_image_files": len(all_images),
              "image_bytes": total_bytes, "decoded_images": len(image_info), "corrupt_images": corrupt,
              "dimension_mismatches": dimension_mismatches, "exact_duplicate_image_groups": duplicate_groups,
              "cross_split_duplicate_groups": [d for d in duplicate_groups if len(d["memberships"]) > 1],
              "near_duplicates_checked": False, "existing_caption_predictions": caption_predictions,
              "detection_ground_truth_boxes": bbox_count,
              "detection_training_ready": bbox_count > 0,
              "conclusion": "Caption dataset. Detection mAP and supervised YOLO tuning require bbox/category ground truth."}
    return report, all_images


def inference(root, paths, weights, output, per_split):
    import os
    os.environ["YOLO_CONFIG_DIR"] = str(output / "library_config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import random
    import torch
    import ultralytics
    from ultralytics import YOLO
    ultralytics.settings.update({"sync": False})
    device = "0" if torch.cuda.is_available() else "cpu"
    selected = []
    for folder in sorted((root / "dataset").iterdir()):
        if not folder.is_dir():
            continue
        for file in sorted(folder.glob("*.json")):
            doc = json.loads(file.read_text(encoding="utf-8-sig"))
            if not isinstance(doc, dict) or "images" not in doc:
                continue
            candidates = [(folder / "images" / i["file_name"]).resolve() for i in doc["images"]]
            candidates = [p for p in candidates if p in paths]
            random.Random(17).shuffle(candidates)
            selected.extend((folder.name, file.stem, p) for p in candidates[:per_split])
    model = YOLO(str(weights))
    gpu_name = torch.cuda.get_device_name(0) if device != "cpu" else None
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    rows = []
    started = time.monotonic()
    for dataset, split, path in selected:
        t = time.monotonic()
        result = model.predict(str(path), imgsz=320, device=device, conf=0.25, verbose=False)[0]
        detections = [{"class_id": int(b.cls.item()), "class_name": model.names[int(b.cls.item())],
                       "confidence": float(b.conf.item()), "xyxy": b.xyxy[0].tolist()}
                      for b in result.boxes]
        rows.append({"dataset": dataset, "split": split, "image": str(path),
                     "wall_seconds": time.monotonic()-t, "speed_ms": result.speed,
                     "detections": detections})
        if len(rows) % 500 == 0:
            print(f"YOLO inference: {len(rows)}/{len(selected)} images", flush=True)
    report = {"mode": "pretrained_detection_inference_only", "weights": str(weights),
              "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "device": device, "torch_version": torch.__version__, "ultralytics_version": ultralytics.__version__,
              "gpu": gpu_name,
              "peak_reserved_mb": torch.cuda.max_memory_reserved()/2**20 if device != "cpu" else None,
              "peak_allocated_mb": torch.cuda.max_memory_allocated()/2**20 if device != "cpu" else None,
              "imgsz": 320, "conf": 0.25, "seed": 17, "images_tested": len(rows),
              "wall_seconds_total": time.monotonic()-started, "successful_images": len(rows),
              "wall_seconds_per_image_median_excluding_first": statistics.median(r["wall_seconds"] for r in rows[1:]) if len(rows)>1 else None,
              "detections_total": sum(len(r["detections"]) for r in rows),
              "class_counts": dict(Counter(d["class_name"] for r in rows for d in r["detections"])),
              "map50": None, "map50_95": None,
              "metric_limitation": "No detection ground truth; predictions are not pseudo-labels for training or mAP evaluation.",
              "results": rows}
    save(output / "inference.json", report)
    return report


def markdown(report, infer):
    lines = ["# Kiểm tra dataset ViECAP", "", "Đã kiểm tra file nhãn và giải mã toàn bộ ảnh được tham chiếu. Dữ liệu là image captioning tiếng Việt.", "",
             "| Dataset | Split | Ảnh | Caption | Thiếu ảnh | Không có caption |", "|---|---|---:|---:|---:|---:|"]
    for dataset, info in report["datasets"].items():
        for split, s in info["splits"].items():
            lines.append(f"| {dataset} | {split} | {s['images']} | {s['captions']} | {len(s['missing_images'])} | {len(s['images_without_captions'])} |")
    lines += ["", f"Ảnh giải mã thành công: {report['decoded_images']}/{report['unique_referenced_image_files']}. Ảnh lỗi: {len(report['corrupt_images'])}. Sai kích thước so với JSON: {len(report['dimension_mismatches'])}.", "",
              f"Nhóm ảnh trùng nội dung SHA256: {len(report['exact_duplicate_image_groups'])}; nhóm trùng xuất hiện trong nhiều split/dataset: {len(report['cross_split_duplicate_groups'])}. Chưa kiểm tra ảnh gần trùng.", "",
              f"Bounding box ground truth: {report['detection_ground_truth_boxes']}. Không thể tính detection mAP hoặc huấn luyện/tuning YOLO có giám sát trực tiếp từ caption.", ""]
    for pred in report["existing_caption_predictions"]:
        lines.append(f"- {Path(pred['file']).name}: {pred['predictions']} caption dự đoán, khớp split {', '.join(pred['matches_dataset_split_by_file_name']) or 'chưa xác định'}; thiếu {len(pred['missing_predictions'])}, ID lạ {len(pred['unknown_prediction_ids'])}. Đây là kết quả có sẵn trong repo.")
    lines += ["", "## Các điểm cần chú ý", ""]
    for dataset, info in report["datasets"].items():
        for split, stats in info["splits"].items():
            unusual = stats["non_five_caption_images"]
            repeats = stats["duplicate_caption_instances"]
            if unusual or repeats:
                lines.append(f"- {dataset}/{split}: {len(unusual)} ảnh có số caption khác 5; {repeats} caption lặp nguyên văn trong cùng ảnh. Chi tiết nằm trong dataset_audit.json.")
    for group in report["exact_duplicate_image_groups"]:
        lines.append(f"- Trùng nội dung ảnh ({', '.join(group['memberships'])}): {', '.join(Path(p).name for p in group['images'])}.")
    if infer:
        lines += ["", "## YOLO inference thực tế", "",
                  f"Model {Path(infer['weights']).name}, device {infer['device']}, imgsz {infer['imgsz']}; chạy thành công {infer['images_tested']} ảnh, phát hiện {infer['detections_total']} đối tượng. Tổng wall time {infer['wall_seconds_total']:.2f} giây, gồm cả lần chạy khởi động đầu tiên.", "",
                  f"Runtime: PyTorch {infer['torch_version']}, Ultralytics {infer['ultralytics_version']}; GPU: {infer['gpu']}. Peak reserved CUDA: {infer['peak_reserved_mb']} MiB (chỉ inference, không phải memory huấn luyện).", "",
                  "Kết quả chi tiết ở inference.json. Đây là kiểm tra chạy detector trên ảnh; không có ground truth nên không báo độ chính xác hoặc dùng đầu ra làm nhãn thật."]
    lines += ["", "## Để kiểm thử bộ tối ưu huấn luyện YOLO", "",
              "Cần bbox và category_id chính xác cho ảnh. UIT-ViIC dùng ID và tên ảnh COCO; có thể ghép nhãn instances COCO chính thức theo image_id, giữ nguyên split UIT-ViIC. Flickr cần nhãn bounding box riêng. Caption và danh sách tên vật thể không chứa tọa độ để chuyển thẳng sang YOLO.", "",
              "Đánh giá mô hình sinh caption ViECAP cần checkpoint ClipCap/GPT và image features phù hợp. Thư mục được cung cấp chưa có checkpoint captioning dùng trong evaluation.sh; các CSV/JSON dự đoán sẵn không chứng minh mô hình chạy lại được."]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="runs/viecap_dataset_test")
    parser.add_argument("--weights", help="Existing local YOLO checkpoint for optional inference test")
    parser.add_argument("--images-per-split", type=int, default=6)
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report, images = audit(root)
    save(output / "dataset_audit.json", report)
    infer = inference(root, images, Path(args.weights).resolve(), output, args.images_per_split) if args.weights else None
    (output / "REPORT.md").write_text(markdown(report, infer), encoding="utf-8")
    print(f"Report: {output / 'REPORT.md'}", flush=True)
    print(json.dumps({"decoded_images": report["decoded_images"], "corrupt_images": len(report["corrupt_images"]),
                      "bbox_ground_truth": report["detection_ground_truth_boxes"],
                      "inference_images": infer["images_tested"] if infer else 0}))


if __name__ == "__main__":
    main()
