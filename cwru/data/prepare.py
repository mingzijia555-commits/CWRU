# -*- coding: utf-8 -*-
"""prepare 流水线：元数据、A/B/C 固定划分、实验清单与归一化参数。"""
from __future__ import annotations

import csv
import os

from cwru.config import ARTIFACTS_DIR, EXPERIMENTS, SPLIT_SETS, WINDOW_PLAN_ORDER, ensure_dirs
from cwru.data.audit import audit_all
from cwru.data.dataset import get_experiment_arrays, manifest_path, run_key
from cwru.data.split import build_fixed_splits, save_splits, validate_splits


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
    """生成元数据 CSV、A/B/C 固定划分、以及「划分集 × 窗口方案 × 输入方案」实验清单。"""
    from cwru.config import SPLITS_DIR
    ensure_dirs()
    records = audit_all()

    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.csv")
    write_metadata_csv(records, meta_path)

    mappings = build_fixed_splits(records)
    save_splits(mappings, records, verbose=verbose)
    report = validate_splits(mappings, records)
    if not report["ok"]:
        raise RuntimeError("固定划分校验失败：\n" + "\n".join(report["issues"]))

    summary = {}
    for split_name in SPLIT_SETS:
        for plan in WINDOW_PLAN_ORDER:
            for exp_id, exp_cfg in EXPERIMENTS.items():
                split_of = mappings[split_name][exp_cfg["files"]]
                key = run_key(split_name, exp_id, plan)
                # 构建内存数组，并保存归一化参数等实验清单
                arrays = get_experiment_arrays(split_name, exp_id, exp_cfg, records,
                                               split_of, plan)
                summary[key] = {
                    "manifest": manifest_path(key),
                    "n_windows": int(len(arrays["y_cls"])),
                    "train_windows": int(sum(m["n_windows"] for m in arrays["files"]
                                             if m["split"] == "train")),
                    "val_windows": int(sum(m["n_windows"] for m in arrays["files"]
                                           if m["split"] == "val")),
                    "test_windows": int(sum(m["n_windows"] for m in arrays["files"]
                                            if m["split"] == "test")),
                    "class_weights": arrays["class_weights"],
                    "train_class_counts": arrays["train_class_counts"],
                    "norm": arrays["norm"],
                }
                if verbose:
                    print(f"[{key}] 窗口总数={summary[key]['n_windows']} "
                          f"(train/val/test = {summary[key]['train_windows']}/"
                          f"{summary[key]['val_windows']}/{summary[key]['test_windows']})")

    return {"records": len(records), "experiments": summary}
