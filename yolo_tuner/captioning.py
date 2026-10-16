"""Use trained compact caption checkpoints and evaluate freely generated captions.

SacreBLEU/chrF use multiple references on a held-out split. The protocol uses
the training regex tokenizer, not COCO's English PTB tokenizer; scores must
not be compared directly with differently tokenized published benchmarks.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from . import cli_ui
from .vision_worker import tokens


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text):
    return " ".join(tokens(text))


def decode_ids(ids, vocabulary):
    inverse = {value: key for key, value in vocabulary.items()}
    words = []
    for value in ids:
        value = int(value)
        if value == 3:
            break
        if value not in (0, 2):
            words.append(inverse.get(value, "<unk>"))
    return " ".join(words)


def score_captions(predictions, references):
    from sacrebleu.metrics import BLEU, CHRF
    if not predictions or len(predictions) != len(references) or any(not row for row in references):
        raise ValueError("Evaluation needs one prediction and at least one reference per image")
    hypotheses = [normalize(text) for text in predictions]
    # SacreBLEU expects reference streams, not a row per image. None handles
    # images with different reference counts without inventing duplicate refs.
    streams = [[normalize(row[index]) if index < len(row) else None for row in references]
               for index in range(max(map(len, references)))]
    bleu = BLEU(tokenize="none", smooth_method="exp", effective_order=False, force=True)
    chrf = CHRF()
    bleu_score = bleu.corpus_score(hypotheses, streams)
    chrf_score = chrf.corpus_score(hypotheses, streams)
    return {"bleu4": bleu_score.score, "bleu4_signature": str(bleu.get_signature()),
            "chrf": chrf_score.score, "chrf_signature": str(chrf.get_signature()),
            "bleu_ngram_precisions": bleu_score.precisions,
            "exact_reference_match_rate": sum(hyp in {normalize(r) for r in refs}
                                               for hyp, refs in zip(hypotheses, references)) / len(hypotheses),
            "unique_caption_rate": len(set(hypotheses)) / len(hypotheses),
            "empty_caption_rate": sum(not h for h in hypotheses) / len(hypotheses),
            "scale": "BLEU/chrF 0-100; rates 0-1. These are not accuracy percentages."}


def validate_holdout(manifest, split):
    """Do not silently score training images as independent validation/test."""
    records = manifest["splits"][split]
    excluded = {"train"} if split == "val" else {"train", "val"}
    known_paths, known_hashes = set(), set()
    for name in excluded:
        for record in manifest["splits"].get(name, []):
            known_paths.add(str(Path(record["image"]).resolve()).casefold())
            if record.get("sha256"):
                known_hashes.add(record["sha256"])
    for record in records:
        if (str(Path(record["image"]).resolve()).casefold() in known_paths
                or record.get("sha256") and record["sha256"] in known_hashes):
            raise ValueError(f"{split} image overlaps training/validation: {record['image']}")


def quality_assessment(metrics, baseline_metrics, predictions):
    unique = len({normalize(text) for text in predictions})
    beats_baseline = (metrics["bleu4"] > baseline_metrics["bleu4"]
                      and metrics["chrf"] > baseline_metrics["chrf"])
    collapsed = len(predictions) > 1 and unique <= 1
    return {"status": "exceeds_text_only_baseline_not_human_verified" if beats_baseline and not collapsed
            else "quality_not_established", "beats_text_only_baseline_on_both_metrics": beats_baseline,
            "single_caption_collapse": collapsed, "unique_captions": unique,
            "human_semantic_accuracy_verified": False,
            "rule": "Must exceed the fixed training-caption baseline on both BLEU4 and chrF, with >1 unique caption; this is a diagnostic gate, not a production-quality guarantee."}


def verify_training_run(run, manifest_path, predictor):
    """Check this pipeline's recorded training request, not arbitrary weights alone."""
    folder = Path(run).resolve()
    report = json.loads((folder / "training_report.json").read_text(encoding="utf-8"))
    last = report["attempts"][-1]
    from .core import training_finished
    if not training_finished(last, report["selection"]["full_epochs"]):
        raise ValueError("Training provenance requires a completed full run")
    checkpoint_folder = Path(last["last_checkpoint"]).resolve().parent
    if predictor.checkpoint not in {checkpoint_folder / "best.pt", checkpoint_folder / "last.pt"}:
        raise ValueError("Checkpoint does not belong to the provided training run")
    request = json.loads((checkpoint_folder / "request.json").read_text(encoding="utf-8"))
    if Path(request["manifest"]).resolve() != manifest_path or request["task"] != "captioning":
        raise ValueError("Evaluation manifest differs from the recorded training request")
    for key in ("model", "encoder", "imgsz", "hidden", "width", "feature_grid", "encoder_sha256"):
        if request["config"].get(key) != predictor.config.get(key):
            raise ValueError(f"Checkpoint architecture/preprocessing differs: {key}")
    return {"training_run": str(folder), "request_sha256": file_hash(checkpoint_folder / "request.json"),
            "training_report_sha256": file_hash(folder / "training_report.json"),
            "basis": "Local pipeline run metadata and checkpoint path/config; not a cryptographic proof against tampering."}


class CaptionPredictor:
    def __init__(self, checkpoint, device="auto"):
        import torch
        from .runner import resolve_device
        from .vision_models import build
        self.checkpoint = Path(checkpoint).resolve()
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if payload.get("task") != "captioning":
            raise ValueError("predict/evaluate-caption require a compact captioning checkpoint")
        self.config = payload["config"]
        self.vocabulary = payload["vocabulary"]
        if any(self.vocabulary.get(k) != v for k, v in {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}.items()):
            raise ValueError("Checkpoint special-token definitions are invalid")
        self.max_tokens = payload["max_tokens"]
        self.epoch = payload["epoch"]
        selected = resolve_device(device)
        self.device = torch.device(selected if selected in {"cpu", "mps"} else f"cuda:{selected}")
        torch.set_num_threads(4)
        self.model = build("captioning", self.config, len(self.vocabulary))
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device).eval()

    def predict(self, paths):
        import torch
        from .caption_features import image_tensor
        images = []
        for path in paths:
            images.append(image_tensor(path, self.config))
        if not images:
            return []
        generated = self.model.generate(torch.stack(images).to(self.device), self.max_tokens).cpu().tolist()
        return [decode_ids(ids, self.vocabulary) for ids in generated]


def predict(args):
    predictor = CaptionPredictor(args.checkpoint, args.device)
    captions = predictor.predict(args.image)
    cli_ui.heading("SINH CAPTION TỪ CHECKPOINT", f"Epoch {predictor.epoch} • {predictor.device} • greedy decoding")
    for path, caption in zip(args.image, captions):
        print(f"\n  Ảnh: {Path(path).resolve()}\n  Caption: {caption}")
    return 0


def evaluate(args):
    from .pipeline import new_session, write_json
    if args.batch < 1:
        raise ValueError("Evaluation batch must be positive")
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "captioning":
        raise ValueError("Caption evaluation requires a captioning manifest")
    records = manifest["splits"][args.split]
    if not records:
        raise ValueError(f"No images in the {args.split} split; cannot claim held-out quality")
    validate_holdout(manifest, args.split)
    # Fail before model inference if metric support was not installed.
    import sacrebleu
    predictor = CaptionPredictor(args.checkpoint, args.device)
    provenance = verify_training_run(args.training_run, manifest_path, predictor) if getattr(args, "training_run", None) else None
    cli_ui.heading("ĐÁNH GIÁ CAPTION SINH TỰ DO", f"{args.split}: {len(records)} ảnh • checkpoint epoch {predictor.epoch}")
    folder = new_session(args.output)
    started = time.monotonic()
    predictions = []
    for offset in range(0, len(records), args.batch):
        group = records[offset:offset + args.batch]
        captions = predictor.predict([r["image"] for r in group])
        predictions.extend({"image": r["image"], "source": r["source"], "caption": cap,
                            "references": r["captions"]} for r, cap in zip(group, captions))
        if offset == 0 or offset + args.batch >= len(records) or offset // args.batch % 10 == 0:
            print(f"  Đã sinh {len(predictions)}/{len(records)} caption • {cli_ui.duration(time.monotonic() - started)}", flush=True)
    hypotheses = [r["caption"] for r in predictions]
    references = [r["references"] for r in predictions]
    metrics = score_captions(hypotheses, references)
    common = Counter(normalize(cap) for r in manifest["splits"]["train"] for cap in r["captions"])
    baseline = common.most_common(1)[0][0] if common else ""
    baseline_metrics = score_captions([baseline] * len(records), references)
    assessment = quality_assessment(metrics, baseline_metrics, hypotheses)
    by_source = {}
    for source in sorted({r["source"] for r in predictions}):
        group = [r for r in predictions if r["source"] == source]
        by_source[source] = {"images": len(group), "metrics": score_captions([r["caption"] for r in group], [r["references"] for r in group])}
    # Rotated predictions break alignment between images and generated captions.
    # A small delta warns of generic captions; it does not prove visual grounding.
    mismatched = score_captions(hypotheses[1:] + hypotheses[:1], references) if len(records) > 1 else None
    report = {"task": "captioning", "split": args.split, "images": len(records),
              "checkpoint": str(predictor.checkpoint), "checkpoint_sha256": file_hash(predictor.checkpoint),
              "checkpoint_epoch": predictor.epoch, "manifest": str(manifest_path), "manifest_sha256": file_hash(manifest_path),
              "device": str(predictor.device), "config": predictor.config, "metrics": metrics, "quality_assessment": assessment,
              "by_source": by_source, "most_common_train_caption_baseline": {"caption": baseline, "metrics": baseline_metrics},
              "rotated_caption_alignment_check": mismatched, "elapsed_seconds": time.monotonic() - started,
              "sacrebleu_version": sacrebleu.__version__, "protocol": "greedy generation from image only; lowercase regex-tokenized multi-reference BLEU4/chrF",
              "inference_preprocessing": "ImageNet center crop/normalization + float16 frozen spatial features" if predictor.config.get("encoder")=="resnet18" else "RGB resize to model size",
              "manifest_holdout_disjoint": True,
              "checkpoint_training_provenance_verified": provenance is not None, "training_provenance": provenance,
              "warnings": ["BLEU/chrF are not human accuracy and not COCO CIDEr; published scores with other tokenizers are not directly comparable.",
                           "Caption evaluation measures this checkpoint only, not global configuration optimality.",
                           "Manifest splits are disjoint; an arbitrary imported checkpoint's training history cannot be inferred from weights."]}
    if provenance:
        report["warnings"].pop()
    write_json(folder / "predictions.json", predictions)
    write_json(folder / "evaluation.json", report)
    lines = ["# Đánh giá caption thực", "", f"Checkpoint: {predictor.checkpoint}",
             f"Epoch: {predictor.epoch}. Split: {args.split}. Images: {len(records)}.", "",
             f"BLEU-4: {metrics['bleu4']:.4f}; chrF: {metrics['chrf']:.4f} (thang 0–100, không phải % accuracy).",
             f"Baseline không nhìn ảnh: BLEU-4 {baseline_metrics['bleu4']:.4f}; chrF {baseline_metrics['chrf']:.4f}.",
             f"Tỷ lệ caption khác nhau: {metrics['unique_caption_rate']:.2%}.", "", report["protocol"], "",
             f"Kết luận đối chứng: {assessment['status']}. Chưa xác nhận độ đúng ngữ nghĩa bằng đánh giá con người.", "",
             "## Caption thực sinh ra", ""]
    for row in predictions[:20]:
        lines += [f"- Ảnh: {row['image']}", f"  - Dự đoán: {row['caption']}", f"  - Tham chiếu: {row['references'][0]}"]
    lines += ["", "## Giới hạn", ""] + [f"- {w}" for w in report["warnings"]]
    (folder / "EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  BLEU-4: {metrics['bleu4']:.4f} • chrF: {metrics['chrf']:.4f}")
    print(f"  Baseline không nhìn ảnh: BLEU-4 {baseline_metrics['bleu4']:.4f} • chrF {baseline_metrics['chrf']:.4f}")
    print(f"  Caption khác nhau: {metrics['unique_caption_rate']:.2%}")
    if assessment["single_caption_collapse"]:
        cli_ui.warning("MODEL LẶP CAPTION: chỉ sinh một câu cho tất cả ảnh được đánh giá.")
    if not assessment["beats_text_only_baseline_on_both_metrics"]:
        cli_ui.warning("CHƯA ĐẠT ĐỐI CHỨNG: không vượt caption cố định không nhìn ảnh trên cả hai metric.")
    else:
        print("  Đã vượt đối chứng không nhìn ảnh; vẫn cần xem caption và đánh giá chất lượng ngữ nghĩa.")
    cli_ui.warning("Các điểm trên không phải % accuracy và không chứng minh cấu hình tối ưu toàn cục.")
    cli_ui.artifact("Báo cáo đánh giá", folder / "EVALUATION.md")
    cli_ui.artifact("Toàn bộ caption", folder / "predictions.json")
    return 0 if assessment["status"] == "exceeds_text_only_baseline_not_human_verified" else 3
