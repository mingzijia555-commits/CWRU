# -*- coding: utf-8 -*-
"""evaluate：加载最佳权重，在测试集上输出完整指标与图表。"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch

from cwru.config import EXPERIMENTS, METRICS_DIR, MODELS, RUNS_DIR, ensure_dirs
from cwru.data.audit import audit_all
from cwru.data.dataset import get_experiment_arrays
from cwru.data.split import load_split
from cwru.evaluation import plots
from cwru.evaluation.metrics import (confusion_matrix_np, file_level_predictions,
                                     macro_f1_from, mil28_breakdown, or_clock_breakdown,
                                     per_class_prf, summarize_window_metrics)
from cwru.models.models import build_model


@torch.no_grad()
def _predict_arrays(model: torch.nn.Module, arrays: dict, device: torch.device,
                    batch_size: int = 256) -> dict:
    model.eval()
    X = torch.from_numpy(arrays["X"])
    probs_list, dia_list = [], []
    for i in range(0, len(X), batch_size):
        xb = X[i: i + batch_size].to(device)
        out = model(xb)
        probs_list.append(torch.softmax(out["logits"], dim=1).cpu().numpy())
        dia_list.append(out["diameter"].cpu().numpy())
    return {"probs": np.concatenate(probs_list), "diameter": np.concatenate(dia_list)}


def load_best_model(experiment: str, model_name: str, in_channels: int,
                    device: torch.device) -> tuple[torch.nn.Module, dict]:
    best_path = os.path.join(RUNS_DIR, experiment, model_name, "best_inference.pt")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model = build_model(model_name, in_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate_run(experiment: str, model_name: str, verbose: bool = True) -> dict:
    """评估单个实验+模型，写 metrics JSON/CSV 与图表，返回汇总指标。"""
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_cfg = EXPERIMENTS[experiment]

    records = audit_all()
    split_of = load_split(exp_cfg["files"])
    arrays = get_experiment_arrays(experiment, exp_cfg, records, split_of)

    model, ckpt = load_best_model(experiment, model_name, arrays["X"].shape[1], device)
    preds = _predict_arrays(model, arrays, device)

    # 测试集子集
    file_split = np.array([m["split"] for m in arrays["files"]])
    sel = np.isin(arrays["file_idx"], np.where(file_split == "test")[0])
    test = {
        "y_cls": arrays["y_cls"][sel],
        "y_reg": arrays["y_reg"][sel],
        "reg_mask": arrays["reg_mask"][sel],
        "file_idx": arrays["file_idx"][sel],
        "probs": preds["probs"][sel],
        "diameter": preds["diameter"][sel],
    }
    y_pred = test["probs"].argmax(axis=1)

    win = summarize_window_metrics(test["y_cls"], y_pred, test["probs"],
                                   test["y_reg"], test["reg_mask"], test["diameter"])

    # 文件级
    file_rows = file_level_predictions(test["y_cls"], test["probs"], test["y_reg"],
                                       test["reg_mask"], test["file_idx"], arrays["files"])["files"]
    f_true = np.array([r["true_cls"] for r in file_rows])
    f_pred = np.array([r["pred_cls"] for r in file_rows])
    f_cm = confusion_matrix_np(f_true, f_pred)
    fault_rows = [r for r in file_rows if r["fault"] != "Normal"]
    f_reg_true = np.array([r["diameter_mil"] / 7.0 for r in fault_rows])
    f_reg_pred = np.array([r["pred_diameter_mil"] / 7.0 for r in fault_rows])
    err = f_reg_pred - f_reg_true
    file_level = {
        "n_files": len(file_rows),
        "accuracy": float((f_true == f_pred).mean()),
        "macro_f1": macro_f1_from(f_cm),
        "confusion_matrix": f_cm.tolist(),
        "per_class": per_class_prf(f_cm),
        "regression": {
            "mae_mil": float(np.abs(err).mean() * 7.0),
            "rmse_mil": float(np.sqrt((err ** 2).mean()) * 7.0),
            "n_fault_files": len(fault_rows),
        },
        "rows": file_rows,
    }

    result = {
        "experiment": experiment,
        "display": exp_cfg["display"],
        "model": model_name,
        "checkpoint": {"epoch": ckpt.get("epoch"), "val_loss": ckpt.get("val_loss")},
        "window_level": win,
        "file_level": file_level,
        "or_clock": or_clock_breakdown(file_rows, test["y_cls"], y_pred, test["y_reg"],
                                       test["diameter"], test["reg_mask"], test["file_idx"],
                                       arrays["files"]),
        "mil28": mil28_breakdown(test["y_cls"], y_pred, test["y_reg"], test["diameter"],
                                 test["reg_mask"], test["file_idx"], arrays["files"]),
    }

    # JSON
    os.makedirs(METRICS_DIR, exist_ok=True)
    json_path = os.path.join(METRICS_DIR, f"metrics_{experiment}_{model_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # CSV：文件级逐文件结果
    csv_path = os.path.join(METRICS_DIR, f"files_{experiment}_{model_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(file_rows[0].keys()))
        writer.writeheader()
        writer.writerows(file_rows)

    # 历史曲线
    hist_path = os.path.join(RUNS_DIR, experiment, model_name, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        # 训练后期补充验证 F1
        plots.plot_loss_curves(history, experiment, model_name)
        plots.plot_val_curves(history, experiment, model_name)
    plots.plot_confusion_matrix(win["confusion_matrix"], experiment, model_name, "window")
    plots.plot_confusion_matrix(f_cm.tolist(), experiment, model_name, "file")
    plots.plot_per_class(win["per_class"], experiment, model_name)
    if win["regression"].get("per_diameter"):
        plots.plot_diameter_errors(win["regression"]["per_diameter"], experiment, model_name)
    plots.plot_or_clock(result["or_clock"], experiment, model_name)

    if verbose:
        print(f"[{experiment}/{model_name}] 测试集: 窗口 Acc={win['accuracy']:.4f} "
              f"Macro-F1={win['macro_f1']:.4f} MAE={win['regression'].get('mae_mil', float('nan')):.2f}mil "
              f"| 文件 Acc={file_level['accuracy']:.4f} Macro-F1={file_level['macro_f1']:.4f}")

    return result


def evaluate_all() -> list[dict]:
    results = []
    for exp in EXPERIMENTS:
        for m in MODELS:
            results.append(evaluate_run(exp, m))
    return results
