# -*- coding: utf-8 -*-
"""compare：汇总 10 组实验，生成对比 CSV 与对比图。"""
from __future__ import annotations

import csv
import json
import os

from cwru.config import EXPERIMENTS, EXPERIMENT_ORDER, METRICS_DIR, MODELS, ensure_dirs
from cwru.evaluation import plots


def collect_metrics() -> list[dict]:
    rows = []
    for exp in EXPERIMENT_ORDER:
        for model in MODELS:
            path = os.path.join(METRICS_DIR, f"metrics_{exp}_{model}.json")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                m = json.load(f)
            rows.append({
                "experiment": exp,
                "model": model,
                "display": m["display"],
                "accuracy": m["window_level"]["accuracy"],
                "macro_f1": m["window_level"]["macro_f1"],
                "mae_mil": m["window_level"]["regression"].get("mae_mil"),
                "rmse_mil": m["window_level"]["regression"].get("rmse_mil"),
                "file_accuracy": m["file_level"]["accuracy"],
                "file_macro_f1": m["file_level"]["macro_f1"],
                "file_mae_mil": m["file_level"]["regression"].get("mae_mil"),
                "file_rmse_mil": m["file_level"]["regression"].get("rmse_mil"),
                "mil28_accuracy": m.get("mil28", {}).get("accuracy"),
                "mil28_mae_mil": m.get("mil28", {}).get("mae_mil"),
            })
    return rows


def run_compare(verbose: bool = True) -> str:
    ensure_dirs()
    rows = collect_metrics()
    if len(rows) != 10:
        raise RuntimeError(f"期望 10 组实验指标，仅找到 {len(rows)} 组，请先运行全部训练与评估")

    # 汇总 CSV
    csv_path = os.path.join(METRICS_DIR, "compare_all.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plots.plot_compare_classification(rows)
    plots.plot_compare_regression(rows)
    plots.plot_compare_window_file(rows)

    if verbose:
        print(f"已汇总 {len(rows)} 组实验 -> {csv_path}")
        for r in rows:
            print(f"  {r['experiment']:8s} {r['model']:7s} Acc={r['accuracy']:.4f} "
                  f"F1={r['macro_f1']:.4f} MAE={r['mae_mil']:.2f}mil 文件Acc={r['file_accuracy']:.4f}")
    return csv_path
