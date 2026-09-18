# -*- coding: utf-8 -*-
"""按固定实验矩阵运行 CWRU 的 90 组实验。

实验数量 = 3 套划分 × 2 种窗口 × 5 种输入 × 3 种模型。
"""
from __future__ import annotations

import json
import os
import shutil

from cwru.config import (ARTIFACTS_DIR, EXPERIMENTS, EXPERIMENT_ORDER, MODELS,
                         MAX_EPOCHS, SPLIT_SETS, TRAIN_SEED, WINDOW_PLAN_ORDER,
                         SPLITS_DIR, ensure_dirs)
from cwru.data.audit import audit_all
from cwru.data.dataset import build_experiment_arrays
from cwru.data.split import load_all_splits
from cwru.evaluation.compare import run_compare
from cwru.evaluation.evaluate import evaluate_run
from cwru.training.trainer import train_model


RESULT_DIR = os.environ.get(
    "CWRU_RESULT_DIR", os.path.join(ARTIFACTS_DIR, "final_results"))


def write_experiment_config(records: list, path: str) -> None:
    config = {
        "n_data_files": len(records),
        "split_sets": SPLIT_SETS,
        "window_plans": WINDOW_PLAN_ORDER,
        "experiments": {name: EXPERIMENTS[name] for name in EXPERIMENT_ORDER},
        "models": MODELS,
        "n_runs": len(SPLIT_SETS) * len(WINDOW_PLAN_ORDER)
                  * len(EXPERIMENT_ORDER) * len(MODELS),
        "train_seed": TRAIN_SEED,
        "max_epochs": MAX_EPOCHS,
        "optimizer": "Adam",
        "regression_loss": "MSE",
        "lr_scheduler": "none",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def copy_split_snapshots(path: str) -> None:
    """把本轮使用的固定划分一并保存，方便之后解释结果。"""
    shutil.copytree(SPLITS_DIR, os.path.join(path, "splits"), dirs_exist_ok=True)


def main() -> None:
    ensure_dirs()
    if os.path.exists(RESULT_DIR) and os.listdir(RESULT_DIR):
        raise FileExistsError(
            f"结果目录已经存在：{RESULT_DIR}\n"
            "为避免覆盖已有结果，请先确认后再清理该目录。")
    os.makedirs(RESULT_DIR, exist_ok=True)

    records = audit_all()
    write_experiment_config(records, os.path.join(RESULT_DIR, "experiment_config.json"))
    copy_split_snapshots(RESULT_DIR)

    for split_name in SPLIT_SETS:
        split_files = load_all_splits(split_name)
        for window_plan in WINDOW_PLAN_ORDER:
            for experiment in EXPERIMENT_ORDER:
                exp_cfg = EXPERIMENTS[experiment]
                arrays = build_experiment_arrays(
                    experiment, exp_cfg, records, split_files[exp_cfg["files"]], window_plan)
                for model_name in MODELS:
                    out_dir = os.path.join(RESULT_DIR, split_name, window_plan,
                                           experiment, model_name)
                    print(f"[{split_name}/{window_plan}/{experiment}/{model_name}]")
                    train_model(experiment, exp_cfg, arrays, model_name,
                                out_dir=out_dir, epochs=MAX_EPOCHS,
                                seed=TRAIN_SEED, split_name=split_name,
                                window_plan=window_plan)
                    evaluate_run(out_dir, experiment, exp_cfg, model_name, arrays,
                                 split_name=split_name, window_plan=window_plan)

    run_compare(RESULT_DIR)
    print(f"90 组实验完成：{RESULT_DIR}")


if __name__ == "__main__":
    main()
