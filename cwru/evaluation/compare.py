# -*- coding: utf-8 -*-
"""compare：汇总一个批次的 90 组实验，生成对比 CSV、均值标准差总表与对比图。"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics

from cwru.config import EXPERIMENT_ORDER, MODELS, SPLIT_SETS, WINDOW_PLAN_ORDER
from cwru.evaluation import plots

METRIC_KEYS = [
    ("window_accuracy", "窗口Accuracy"),
    ("window_macro_f1", "窗口Macro-F1"),
    ("window_mae_mil", "窗口直径MAE(mil)"),
    ("window_rmse_mil", "窗口直径RMSE(mil)"),
    ("file_accuracy", "文件Accuracy"),
    ("file_macro_f1", "文件Macro-F1"),
    ("file_mae_mil", "文件直径MAE(mil)"),
    ("file_rmse_mil", "文件直径RMSE(mil)"),
]

CSV_FIELDS = ["split", "window_plan", "experiment", "model", "best_epoch", "val_loss",
              "n_test_windows", "n_test_files",
              "window_accuracy", "window_macro_f1", "window_mae_mil", "window_rmse_mil",
              "file_accuracy", "file_macro_f1", "file_mae_mil", "file_rmse_mil",
              "mil28_accuracy", "mil28_mae_mil"]


def collect_batch_metrics(batch_dir: str) -> list[dict]:
    """读取批次内全部 metrics.json，返回汇总行。"""
    rows = []
    for split_name in SPLIT_SETS:
        for plan in WINDOW_PLAN_ORDER:
            for exp in EXPERIMENT_ORDER:
                for model in MODELS:
                    path = os.path.join(batch_dir, split_name, plan, exp, model, "metrics.json")
                    if not os.path.exists(path):
                        continue
                    with open(path, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    rows.append({
                        "split": split_name,
                        "window_plan": plan,
                        "experiment": exp,
                        "model": model,
                        "best_epoch": m.get("checkpoint", {}).get("epoch"),
                        "val_loss": m.get("checkpoint", {}).get("val_loss"),
                        "n_test_windows": m.get("n_test_windows"),
                        "n_test_files": m.get("n_test_files"),
                        "window_accuracy": m["window_level"]["accuracy"],
                        "window_macro_f1": m["window_level"]["macro_f1"],
                        "window_mae_mil": m["window_level"]["regression"].get("mae_mil"),
                        "window_rmse_mil": m["window_level"]["regression"].get("rmse_mil"),
                        "file_accuracy": m["file_level"]["accuracy"],
                        "file_macro_f1": m["file_level"]["macro_f1"],
                        "file_mae_mil": m["file_level"]["regression"].get("mae_mil"),
                        "file_rmse_mil": m["file_level"]["regression"].get("rmse_mil"),
                        "mil28_accuracy": m.get("mil28", {}).get("accuracy"),
                        "mil28_mae_mil": m.get("mil28", {}).get("mae_mil"),
                    })
    return rows


def _write_csv(path: str, fields: list[str], rows: list[dict]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})
    return path


def _stat(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def build_mean_std(rows: list[dict]) -> tuple[list[dict], dict]:
    """按 (窗口, 输入, 模型) 汇聚 A/B/C 的均值与标准差。"""
    summary_rows = []
    summary = {}
    for plan in WINDOW_PLAN_ORDER:
        for exp in EXPERIMENT_ORDER:
            for model in MODELS:
                sel = [r for r in rows if r["window_plan"] == plan
                       and r["experiment"] == exp and r["model"] == model]
                if not sel:
                    continue
                key = f"{plan}/{exp}/{model}"
                entry = {"window_plan": plan, "experiment": exp, "model": model,
                         "n_splits": len({r["split"] for r in sel})}
                for metric, _ in METRIC_KEYS:
                    st = _stat([r.get(metric) for r in sel])
                    entry[f"{metric}_mean"] = st["mean"]
                    entry[f"{metric}_std"] = st["std"]
                    entry[f"{metric}_min"] = st["min"]
                    entry[f"{metric}_max"] = st["max"]
                summary[key] = entry
                summary_rows.append(entry)
    return summary_rows, summary


def summarize_conclusions(rows: list[dict], summary: dict) -> dict:
    """生成关键比较结论（最佳分类/回归模型、重叠与划分影响）。"""
    def mean_of(plan: str, exp: str, model: str, metric: str):
        e = summary.get(f"{plan}/{exp}/{model}")
        return e.get(f"{metric}_mean") if e else None

    # 最佳分类：窗口 Macro-F1 平均最高
    cls_candidates = [(k, v.get("window_macro_f1_mean")) for k, v in summary.items()
                      if v.get("window_macro_f1_mean") is not None]
    best_cls = max(cls_candidates, key=lambda t: t[1]) if cls_candidates else (None, None)

    # 最佳回归：窗口 MAE 平均最低
    reg_candidates = [(k, v.get("window_mae_mil_mean")) for k, v in summary.items()
                      if v.get("window_mae_mil_mean") is not None]
    best_reg = min(reg_candidates, key=lambda t: t[1]) if reg_candidates else (None, None)

    # 模型/重叠/划分层面的平均
    def avg(metric: str, **filters) -> float | None:
        vals = [v.get(metric) for v in summary.values()
                if all(v.get(k) == val for k, val in filters.items())
                and v.get(metric) is not None]
        return statistics.fmean(vals) if vals else None

    model_avg = {m: avg("window_macro_f1_mean", model=m) for m in MODELS}
    model_mae = {m: avg("window_mae_mil_mean", model=m) for m in MODELS}
    overlap_avg = {p: avg("window_macro_f1_mean", window_plan=p) for p in WINDOW_PLAN_ORDER}
    overlap_mae = {p: avg("window_mae_mil_mean", window_plan=p) for p in WINDOW_PLAN_ORDER}

    split_avg = {}
    for s in SPLIT_SETS:
        sel = [r for r in rows if r["split"] == s and r.get("window_macro_f1") is not None]
        split_avg[s] = {
            "window_macro_f1": statistics.fmean([r["window_macro_f1"] for r in sel]) if sel else None,
            "window_accuracy": statistics.fmean([r["window_accuracy"] for r in sel]) if sel else None,
        }

    exp_avg = {}
    for exp in EXPERIMENT_ORDER:
        v = avg("window_macro_f1_mean", experiment=exp)
        m = avg("window_mae_mil_mean", experiment=exp)
        exp_avg[exp] = {"window_macro_f1": v, "window_mae_mil": m}

    return {
        "best_classification": best_cls[0],
        "best_classification_macro_f1": best_cls[1],
        "best_regression": best_reg[0],
        "best_regression_mae_mil": best_reg[1],
        "model_mean_macro_f1": model_avg,
        "model_mean_mae_mil": model_mae,
        "overlap_mean_macro_f1": overlap_avg,
        "overlap_mean_mae_mil": overlap_mae,
        "split_mean": split_avg,
        "experiment_mean": exp_avg,
    }


def run_compare(batch_dir: str, verbose: bool = True) -> dict:
    """汇总批次：写出 comparisons/ 下的 CSV 与对比图，返回结论字典。"""
    rows = collect_batch_metrics(batch_dir)
    if not rows:
        raise RuntimeError(f"批次目录中未找到任何 metrics.json：{batch_dir}")

    comp_dir = os.path.join(batch_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    all_csv = _write_csv(os.path.join(comp_dir, "metrics_all.csv"), CSV_FIELDS, rows)

    summary_rows, summary = build_mean_std(rows)
    ms_fields = list(summary_rows[0].keys())
    ms_csv = _write_csv(os.path.join(comp_dir, "mean_std_summary.csv"), ms_fields, summary_rows)

    plots.plot_split_comparison(rows, comp_dir)
    plots.plot_model_comparison(rows, comp_dir)
    plots.plot_overlap_comparison(rows, comp_dir)

    conclusions = summarize_conclusions(rows, summary)
    with open(os.path.join(comp_dir, "conclusions.json"), "w", encoding="utf-8") as f:
        json.dump(conclusions, f, ensure_ascii=False, indent=2)
    plots.plot_final_overview(rows, conclusions, comp_dir)

    if verbose:
        print(f"[compare] {len(rows)} 组结果 -> {comp_dir}")
    return {"n_runs": len(rows), "rows": rows, "summary": summary,
            "summary_rows": summary_rows, "conclusions": conclusions,
            "metrics_all_csv": all_csv, "mean_std_csv": ms_csv, "comp_dir": comp_dir}
