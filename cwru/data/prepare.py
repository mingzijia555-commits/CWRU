# -*- coding: utf-8 -*-
"""prepare 流水线：元数据、固定划分、实验清单与归一化参数。"""
from __future__ import annotations

import csv
import json
import os

from cwru.config import ARTIFACTS_DIR, EXPERIMENTS, MANIFESTS_DIR, ensure_dirs
from cwru.data.audit import audit_all
from cwru.data.dataset import build_experiment_arrays, save_experiment_manifest
from cwru.data.split import build_split, save_splits


def write_metadata_csv(records, path: str) -> None:
    fields = ["filename", "end", "fault", "diameter_mil", "load", "or_clock",
              "file_id", "var_de", "var_fe", "n_de", "n_fe", "in_set_101", "in_set_109"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = {k: r.to_dict()[k] for k in fields if k != "extras"}
            writer.writerow(row)


def run_prepare(verbose: bool = True) -> dict:
    """生成元数据 CSV、两个固定划分、5 个实验清单（含归一化参数与类别权重）。"""
    ensure_dirs()
    records = audit_all()

    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.csv")
    write_metadata_csv(records, meta_path)

    splits = build_split(records)
    save_splits(splits)

    summary = {}
    for exp_id, exp_cfg in EXPERIMENTS.items():
        split_of = splits[exp_cfg["files"]]
        arrays = build_experiment_arrays(exp_id, exp_cfg, records, split_of)
        manifest_path = save_experiment_manifest(exp_id, exp_cfg, arrays, split_of)
        summary[exp_id] = {
            "manifest": manifest_path,
            "n_windows": int(len(arrays["y_cls"])),
            "train_windows": int(sum(m["n_windows"] for m in arrays["files"] if m["split"] == "train")),
            "val_windows": int(sum(m["n_windows"] for m in arrays["files"] if m["split"] == "val")),
            "test_windows": int(sum(m["n_windows"] for m in arrays["files"] if m["split"] == "test")),
            "class_weights": arrays["class_weights"],
            "train_class_counts": arrays["train_class_counts"],
            "norm": arrays["norm"],
        }
        if verbose:
            print(f"[{exp_id}] 窗口总数={summary[exp_id]['n_windows']} "
                  f"(train/val/test = {summary[exp_id]['train_windows']}/"
                  f"{summary[exp_id]['val_windows']}/{summary[exp_id]['test_windows']})")
            print(f"    类别窗口数(训练)={summary[exp_id]['train_class_counts']} "
                  f"权重={[round(w, 3) for w in summary[exp_id]['class_weights']]}")

    return {"records": len(records), "experiments": summary}
