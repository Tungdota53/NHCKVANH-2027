"""Post-decision selection-rule sensitivity ablation & 3-domain held-out test evaluation."""

import json
import subprocess
import sys
from pathlib import Path

from research_pipeline.v2.paper_export import export_selection_rule_ablation_svg
from research_pipeline.v2.selection import select
from research_pipeline.v2.workflow import audit_run

ROOT = Path(__file__).resolve().parents[1]


def _latest_subdir(parent: Path) -> Path:
    dirs = sorted(p for p in parent.iterdir() if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No subdirectories in {parent}")
    return dirs[-1]


def run_selection_ablation() -> dict:
    out_dir = ROOT / "runs" / "nckh_2027_v2_selection_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_specs = [
        ("Wildlife (mAP@50)", _latest_subdir(_latest_subdir(ROOT / "runs" / "nckh_2027_v2_study1_wildlife_n8") / "reference")),
        ("Penn-Fudan (mAP@50:95)", _latest_subdir(_latest_subdir(ROOT / "runs" / "nckh_2027_v2_study2_pennfudan_n8") / "reference")),
        ("CIFAR-10 (Top-1 Acc)", _latest_subdir(_latest_subdir(ROOT / "runs" / "nckh_2027_v2_study3_cifar10_gpu_n8") / "reference")),
    ]

    all_rows = []
    domains_summary = {}
    for domain_label, ref_path in ref_specs:
        _, proto, rows, _ = audit_run(ref_path)
        feas = [r for r in rows if r["state"] == "confirmed_feasible"]
        by_cid = {r["config_id"]: r for r in feas}
        domain_entries = []

        rule_configs = [
            ("weighted", None, "Weighted"),
            ("tchebycheff", None, "Tchebycheff"),
            ("topsis", None, "TOPSIS"),
            ("knee", None, "Knee"),
        ] + [("epsilon_constraint", eps, f"eps={eps:g}") for eps in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10)]

        for rule_name, eps_val, lbl in rule_configs:
            p_mod = json.loads(json.dumps(proto))
            p_mod["selection_rule"] = rule_name
            if eps_val is not None:
                p_mod["quality_tolerance"] = float(eps_val)
            dec = select(rows, p_mod)
            cid = dec["selected_config_id"]
            chosen = by_cid.get(cid)
            entry = {
                "domain": domain_label,
                "rule": rule_name,
                "epsilon": eps_val,
                "label": lbl,
                "selected_config_id": cid,
                "selected_index": dec["selected_index"],
                "quality_mean": chosen["quality_mean"] if chosen else None,
                "time_mean": chosen["time_mean"] if chosen else None,
                "memory_max": chosen["memory_max"] if chosen else None,
                "utility": chosen["utility"] if chosen else None,
                "fallback": dec.get("selection_fallback"),
            }
            domain_entries.append(entry)
            all_rows.append(entry)
        domains_summary[domain_label] = domain_entries

    svg_path = export_selection_rule_ablation_svg(all_rows, out_dir / "pareto_epsilon_tradeoff.svg", lang="vi")
    export_selection_rule_ablation_svg(all_rows, out_dir / "pareto_epsilon_tradeoff_en.svg", lang="en")

    summary = {
        "schema_version": 1,
        "domains": domains_summary,
        "svg_vi": str(svg_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Selection-Rule & Epsilon-Constraint Sensitivity Ablation",
        "",
        "| Domain | Rule / Tolerance | Selected Config | Quality Mean | Time Mean (s) | Memory Max (MiB) | Utility U | Fallback |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in all_rows:
        md_lines.append(
            f"| {r['domain']} | {r['label']} | `{r['selected_config_id']}` (c{r['selected_index']:02d}) | "
            f"{r['quality_mean']:.4f} | {r['time_mean']:.1f} | {r['memory_max']:.0f} | {r['utility']:.4f} | {r['fallback'] or '—'} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary


def _train_and_eval_3seed(study_root: Path, eval_out: Path, data_arg: str, task: str, full_epochs: int, extra_args: list[str], repeat_seed: int = 17) -> dict:
    study_sub = _latest_subdir(study_root)
    src_run = _latest_subdir(study_sub / "conformal_mf" / f"repeat_{repeat_seed}")
    _, src_proto, src_rows, _ = audit_run(src_run)
    dec = select(src_rows, src_proto)
    sel_cfg = src_rows[dec["selected_index"]]["config"]

    existing_evals = sorted(eval_out.glob("optimize_3seed/*/evaluation/*/evaluation.json"))
    if existing_evals:
        eval_file = existing_evals[-1]
        ev_data = json.loads(eval_file.read_text(encoding="utf-8"))
        return {
            "source_run": str(src_run),
            "selected_config_id": dec["selected_config_id"],
            "selected_config": sel_cfg,
            "trained_run": str(eval_file.parent.parent.parent),
            "evaluation_file": str(eval_file),
            "evaluated_seeds": ev_data.get("evaluated_seeds"),
            "multi_seed_summary": ev_data.get("multi_seed_summary"),
            "validation_vs_test": ev_data.get("validation_vs_test"),
            "latency": ev_data.get("latency"),
        }

    eval_out.mkdir(parents=True, exist_ok=True)
    cfg_path = eval_out / "selected_single_config.json"
    cfg_path.write_text(json.dumps([sel_cfg], indent=2), encoding="utf-8")
    opt_root = eval_out / "optimize_3seed"

    cmd_opt = [
        sys.executable, "-m", "research_pipeline.v2", "optimize",
        "--data", data_arg,
        "--task", task,
        "--device", "0",
        "--grid", str(cfg_path),
        "--full-epochs", str(full_epochs),
        "--seeds", "17", "29", "43",
        "--method", "default",
        "--budget-seconds", "1800",
        "--weights", "0.90", "0.07", "0.03",
        "--memory-scale", "2048",
        "--selection-rule", "epsilon_constraint",
        "--output", str(opt_root),
        *extra_args,
    ]
    subprocess.run(cmd_opt, check=True)
    trained_run = _latest_subdir(opt_root)

    cmd_eval = [
        sys.executable, "-m", "research_pipeline.v2", "evaluate-v2",
        "--run", str(trained_run),
        "--seed-scope", "all",
    ]
    subprocess.run(cmd_eval, check=True)
    eval_dir = _latest_subdir(trained_run / "evaluation")
    ev_data = json.loads((eval_dir / "evaluation.json").read_text(encoding="utf-8"))
    return {
        "source_run": str(src_run),
        "selected_config_id": dec["selected_config_id"],
        "selected_config": sel_cfg,
        "trained_run": str(trained_run),
        "evaluation_file": str(eval_dir / "evaluation.json"),
        "evaluated_seeds": ev_data.get("evaluated_seeds"),
        "multi_seed_summary": ev_data.get("multi_seed_summary"),
        "validation_vs_test": ev_data.get("validation_vs_test"),
        "latency": ev_data.get("latency"),
    }


def main():
    ablation = run_selection_ablation()
    pennfudan_eval = _train_and_eval_3seed(
        ROOT / "runs" / "nckh_2027_v2_study2_pennfudan_n8",
        ROOT / "runs" / "nckh_2027_v2_eval_pennfudan",
        data_arg="runs/datasets/penn_fudan_research/data.yaml",
        task="detection",
        full_epochs=12,
        extra_args=["--max-memory", "1850", "--detection-metric", "map50_95", "--quality-tolerance", "0.01"],
    )
    cifar_eval = _train_and_eval_3seed(
        ROOT / "runs" / "nckh_2027_v2_study3_cifar10_gpu_n8",
        ROOT / "runs" / "nckh_2027_v2_eval_cifar10",
        data_arg="runs/datasets/cifar10_research",
        task="classification",
        full_epochs=20,
        extra_args=["--quality-tolerance", "0.02"],
    )
    wild_ev_path = _latest_subdir(_latest_subdir(ROOT / "runs" / "nckh_2027_study3_wildlife_full") / "evaluation") / "evaluation.json"
    wild_ev = json.loads(wild_ev_path.read_text(encoding="utf-8"))

    consolidated = {
        "wildlife_20ep_3seed": {
            "evaluation_file": str(wild_ev_path),
            "multi_seed_summary": wild_ev.get("multi_seed_summary"),
            "validation_vs_test": wild_ev.get("validation_vs_test"),
            "latency": wild_ev.get("latency"),
        },
        "pennfudan_12ep_3seed": pennfudan_eval,
        "cifar10_20ep_3seed": cifar_eval,
    }
    out_path = ROOT / "runs" / "nckh_2027_v2_selection_ablation" / "heldout_test_summary.json"
    out_path.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    print("Selection ablation & 3-domain test evaluations saved to", out_path)


if __name__ == "__main__":
    main()
