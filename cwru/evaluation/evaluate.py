# -*- coding: utf-8 -*-
"""evaluate：加载最佳权重，在测试集上输出汇总指标、逐文件结果与图表。

输出约定（批次内单组实验目录）：
- metrics.json：仅汇总指标，不嵌入逐文件明细；
- files.csv：仅逐测试文件预测记录；
- figures/：本组损失曲线、验证曲线、混淆矩阵等。
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch

from cwru.config import CLASSES, MODELS, WINDOW_PLANS
from cwru.data.dataset import get_experiment_arrays
from cwru.evaluation import plots
from cwru.evaluation.metrics import (confusion_matrix_np, file_level_predictions,
                                     macro_f1_from, mil28_breakdown, or_clock_breakdown,
                                     per_class_prf, regression_errors, summarize_window_metrics)
from cwru.models.models import build_model

CSV_FIELDS = ["filename", "split", "end", "fault", "true_class", "pred_class",
              "p_Normal", "p_IR", "p_OR", "p_B", "true_diameter_mil", "pred_diameter_mil",
              "load", "or_clock", "n_windows", "correct"]


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


def load_best_model(ckpt_dir: str, model_name: str, in_channels: int,
                    device: torch.device) -> tuple[torch.nn.Module, dict]:
    """从指定实验目录加载最佳推理权重。"""
    best_path = os.path.join(ckpt_dir, "best_inference.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"未找到推理权重：{best_path}\n请检查划分集/窗口方案/输入方案/模型是否与实际训练一致")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model = build_model(model_name, in_channels).to(device)
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"权重与当前实验通道数不匹配（当前 in_channels={in_channels}，权重目录 {ckpt_dir}）。"
            f"请确认输入方案（dual101 为双通道，其余为单通道）与权重来源一致。原始错误：{exc}") from exc
    model.eval()
    return model, ckpt


def write_files_csv(path: str, rows: list[dict]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "filename": r["filename"], "split": r["split"], "end": r["end"],
                "fault": r["fault"], "true_class": CLASSES[r["true_cls"]],
                "pred_class": CLASSES[r["pred_cls"]],
                "p_Normal": f"{r['pred_prob'][0]:.6f}", "p_IR": f"{r['pred_prob'][1]:.6f}",
                "p_OR": f"{r['pred_prob'][2]:.6f}", "p_B": f"{r['pred_prob'][3]:.6f}",
                "true_diameter_mil": r["diameter_mil"],
                "pred_diameter_mil": f"{r['pred_diameter_mil']:.4f}",
                "load": r["load"], "or_clock": r["or_clock"] if r["or_clock"] is not None else "",
                "n_windows": r["n_windows"],
                "correct": int(r["true_cls"] == r["pred_cls"]),
            })
    return path


def evaluate_run(out_dir: str, experiment: str, exp_cfg: dict, model_name: str,
                 arrays: dict, split_name: str = "", window_plan: str = "",
                 verbose: bool = True) -> dict:
    """评估单组（划分×窗口×输入×模型），写批次内 metrics.json / files.csv / figures。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_best_model(out_dir, model_name, arrays["X"].shape[1], device)
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

    # 文件级：直径由模型窗口预测均值聚合（问题一修正）
    file_rows = file_level_predictions(test["y_cls"], test["probs"], test["y_reg"],
                                       test["diameter"], test["reg_mask"],
                                       test["file_idx"], arrays["files"])["files"]
    f_true = np.array([r["true_cls"] for r in file_rows])
    f_pred = np.array([r["pred_cls"] for r in file_rows])
    f_cm = confusion_matrix_np(f_true, f_pred)
    fault_rows = [r for r in file_rows if r["fault"] != "Normal"]
    f_reg = (regression_errors(
        np.array([r["diameter_mil"] / 7.0 for r in fault_rows], dtype=np.float32),
        np.array([r["pred_diameter_mil"] / 7.0 for r in fault_rows], dtype=np.float32),
    ) if fault_rows else {})
    file_level = {
        "n_files": len(file_rows),
        "n_fault_files": len(fault_rows),
        "accuracy": float((f_true == f_pred).mean()),
        "macro_f1": macro_f1_from(f_cm),
        "confusion_matrix": f_cm.tolist(),
        "per_class": per_class_prf(f_cm),
        "regression": {
            "mae_mil": f_reg.get("mae_mil", float("nan")),
            "rmse_mil": f_reg.get("rmse_mil", float("nan")),
            "n_fault_files": len(fault_rows),
        },
    }

    or_clock = or_clock_breakdown(test["y_cls"], y_pred, test["y_reg"], test["diameter"],
                                  test["reg_mask"], test["file_idx"], arrays["files"])
    mil28 = mil28_breakdown(test["y_cls"], y_pred, test["y_reg"], test["diameter"],
                            test["reg_mask"], test["file_idx"], arrays["files"])
    val_loss = ckpt.get("val_loss")

    result = {
        "split_set": split_name,
        "window_plan": window_plan,
        "window_display": WINDOW_PLANS.get(window_plan, {}).get("display", ""),
        "experiment": experiment,
        "display": exp_cfg["display"],
        "model": model_name,
        "checkpoint": {"epoch": ckpt.get("epoch"), "val_loss": val_loss,
                       "stride": int(arrays.get("stride", 0))},
        "n_test_windows": win["n_windows"],
        "n_test_files": file_level["n_files"],
        "window_level": win,
        "file_level": file_level,
        "or_clock": or_clock,
        "mil28": mil28,
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    csv_path = write_files_csv(os.path.join(out_dir, "files.csv"), file_rows)

    fig_dir = os.path.join(out_dir, "figures")
    hist_path = os.path.join(out_dir, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        plots.plot_loss_curves(history, fig_dir, experiment, model_name, split_name, window_plan)
        plots.plot_val_curves(history, fig_dir, experiment, model_name, split_name, window_plan)
    plots.plot_confusion_matrix(win["confusion_matrix"], fig_dir, experiment, model_name,
                                "window", split_name, window_plan)
    plots.plot_confusion_matrix(f_cm.tolist(), fig_dir, experiment, model_name,
                                "file", split_name, window_plan)
    plots.plot_per_class(win["per_class"], fig_dir, experiment, model_name, split_name, window_plan)
    if win["regression"].get("per_diameter"):
        plots.plot_diameter_errors(win["regression"]["per_diameter"], fig_dir,
                                   experiment, model_name, split_name, window_plan)
    plots.plot_or_clock(or_clock, fig_dir, experiment, model_name, split_name, window_plan)

    if verbose:
        print(f"[{split_name}/{window_plan}/{experiment}/{model_name}] "
              f"窗口 Acc={win['accuracy']:.4f} Macro-F1={win['macro_f1']:.4f} "
              f"MAE={win['regression'].get('mae_mil', float('nan')):.2f}mil "
              f"| 文件 Acc={file_level['accuracy']:.4f} Macro-F1={file_level['macro_f1']:.4f} "
              f"MAE={file_level['regression']['mae_mil']:.2f}mil")

    return {
        "split": split_name,
        "window_plan": window_plan,
        "experiment": experiment,
        "model": model_name,
        "best_epoch": ckpt.get("epoch"),
        "val_loss": val_loss,
        "n_test_windows": win["n_windows"],
        "n_test_files": file_level["n_files"],
        "window_accuracy": win["accuracy"],
        "window_macro_f1": win["macro_f1"],
        "window_mae_mil": win["regression"].get("mae_mil"),
        "window_rmse_mil": win["regression"].get("rmse_mil"),
        "file_accuracy": file_level["accuracy"],
        "file_macro_f1": file_level["macro_f1"],
        "file_mae_mil": file_level["regression"]["mae_mil"],
        "file_rmse_mil": file_level["regression"]["rmse_mil"],
        "mil28_accuracy": mil28.get("accuracy"),
        "mil28_mae_mil": mil28.get("mae_mil"),
        "metrics_json": json_path,
        "files_csv": csv_path,
        "figures_dir": fig_dir,
    }


def evaluate_matrix(split_name: str, records, split_of: dict[str, dict[str, str]],
                    verbose: bool = True,
                    batch_dir: str | None = None, only_plan: str | None = None,
                    only_experiment: str | None = None, only_model: str | None = None,
                    on_result=None) -> list[dict]:
    """按 (窗口方案 × 输入方案 × 模型) 评估某一套划分的全部组合。

    split_of 必须同时提供 ``{"101": mapping, "109": mapping}``，因为不同输入
    方案使用不同文件集合；不能用一份映射覆盖两类实验。

    batch_dir 给出时写入该批次；为空则写 artifacts/runs 下的兼容目录。
    on_result 回调用于逐组更新 README。
    """
    from cwru.config import EXPERIMENT_ORDER, EXPERIMENTS, RUNS_DIR

    plans = [only_plan] if only_plan else list(WINDOW_PLANS)
    exps = [only_experiment] if only_experiment else EXPERIMENT_ORDER
    models = [only_model] if only_model else MODELS

    results = []
    for plan in plans:
        for exp in exps:
            exp_cfg = EXPERIMENTS[exp]
            arrays = get_experiment_arrays(split_name, exp, exp_cfg, records,
                                           split_of[exp_cfg["files"]], plan)
            for model in models:
                out_dir = (os.path.join(batch_dir, split_name, plan, exp, model)
                           if batch_dir else os.path.join(RUNS_DIR,
                                                          f"{split_name}_{plan}_{exp}_{model}"))
                res = evaluate_run(out_dir, exp, exp_cfg, model, arrays,
                                   split_name=split_name, window_plan=plan, verbose=verbose)
                results.append(res)
                if on_result:
                    on_result(res)
    return results
