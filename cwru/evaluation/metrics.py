# -*- coding: utf-8 -*-
"""指标计算：窗口级与文件级，分类与回归，含 28 mil 与外圈钟点细分。

全部基于 numpy 实现，不依赖 sklearn。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

REG_SCALE = 7.0  # mil / 7


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[str(c)] = {"precision": float(precision), "recall": float(recall), "f1": float(f1),
                       "support": int(cm[c, :].sum())}
    return out


def macro_f1_from(cm: np.ndarray) -> float:
    prf = per_class_prf(cm)
    return float(np.mean([prf[k]["f1"] for k in prf]))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def regression_errors(y_reg_true: np.ndarray, y_reg_pred: np.ndarray) -> dict:
    """输入为缩放单位（mil/7）。返回缩放与 mil 两种单位的 MAE/RMSE。"""
    err = y_reg_pred - y_reg_true
    abs_err = np.abs(err)
    out = {
        "mae_scaled": float(abs_err.mean()),
        "rmse_scaled": float(np.sqrt((err ** 2).mean())),
        "mae_mil": float(abs_err.mean() * REG_SCALE),
        "rmse_mil": float(np.sqrt((err ** 2).mean()) * REG_SCALE),
    }
    out["per_diameter"] = {}
    for d in sorted(set(y_reg_true.tolist())):
        sel = y_reg_true == d
        if sel.sum() == 0:
            continue
        e = err[sel]
        out["per_diameter"][f"{d * REG_SCALE:.0f}mil"] = {
            "n": int(sel.sum()),
            "mae_scaled": float(np.abs(e).mean()),
            "rmse_scaled": float(np.sqrt((e ** 2).mean())),
            "mae_mil": float(np.abs(e).mean() * REG_SCALE),
            "rmse_mil": float(np.sqrt((e ** 2).mean()) * REG_SCALE),
        }
    return out


def file_level_predictions(y_cls: np.ndarray, probs: np.ndarray, y_reg: np.ndarray,
                           y_reg_pred: np.ndarray, reg_mask: np.ndarray,
                           file_idx: np.ndarray, files: list[dict]) -> dict:
    """按文件聚合窗口预测：类别取平均概率最大者，直径取模型窗口预测均值。

    文件级连续直径 = 该文件全部故障窗口的模型直径预测算术平均（缩放单位 × REG_SCALE）；
    Normal 文件无故障窗口，直径记 0，且不参与故障直径 MAE/RMSE。
    """
    by_file = defaultdict(lambda: {"idx": []})
    for i, fi in enumerate(file_idx):
        by_file[int(fi)]["idx"].append(i)

    rows = []
    for meta in files:
        fi = meta["index"]
        if fi not in by_file:
            continue
        idx = np.array(by_file[fi]["idx"], dtype=int)
        mean_prob = probs[idx].mean(axis=0)
        pred_cls = int(mean_prob.argmax())
        fault = meta["fault"]
        true_cls = {"Normal": 0, "IR": 1, "OR": 2, "B": 3}[fault]
        fault_windows = reg_mask[idx]
        if fault_windows.any():
            pred_dia = float(y_reg_pred[idx][fault_windows].mean()) * REG_SCALE
        else:
            pred_dia = 0.0
        rows.append({
            "filename": meta["filename"], "split": meta["split"],
            "end": meta["end"], "fault": fault, "diameter_mil": meta["diameter_mil"],
            "load": meta["load"], "or_clock": meta["or_clock"], "n_windows": int(len(idx)),
            "true_cls": true_cls, "pred_cls": pred_cls,
            "pred_prob": [float(p) for p in mean_prob],
            "pred_diameter_mil": pred_dia,
        })
    return {"files": rows}


def summarize_window_metrics(y_cls: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                             y_reg: np.ndarray, reg_mask: np.ndarray,
                             y_reg_pred: np.ndarray) -> dict:
    cm = confusion_matrix_np(y_cls, y_pred)
    out = {
        "n_windows": int(len(y_cls)),
        "accuracy": accuracy(y_cls, y_pred),
        "macro_f1": macro_f1_from(cm),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_prf(cm),
        "regression": regression_errors(y_reg[reg_mask], y_reg_pred[reg_mask]) if reg_mask.any() else {},
    }
    return out


def or_clock_breakdown(y_cls: np.ndarray, y_pred: np.ndarray,
                       y_reg: np.ndarray, y_reg_pred: np.ndarray, reg_mask: np.ndarray,
                       file_idx: np.ndarray, files: list[dict]) -> dict:
    """外圈 3/6/12 点钟位置的窗口级细分表现。"""
    clock_of_file = {m["index"]: m["or_clock"] for m in files if m["fault"] == "OR"}
    out = {}
    for clock in (3, 6, 12):
        sel = np.array([clock_of_file.get(int(fi)) == clock for fi in file_idx])
        sel = sel & (y_cls == 2)  # 仅 OR 窗口
        if not sel.any():
            out[f"{clock}点钟"] = {"n_windows": 0}
            continue
        acc = float((y_pred[sel] == 2).mean())
        e = np.abs(y_reg_pred[sel] - y_reg[sel]) * REG_SCALE
        out[f"{clock}点钟"] = {
            "n_windows": int(sel.sum()),
            "or_recall": acc,
            "mae_mil": float(e.mean()),
            "rmse_mil": float(np.sqrt((e ** 2).mean())),
        }
    return out


def mil28_breakdown(y_cls: np.ndarray, y_pred: np.ndarray, y_reg: np.ndarray,
                    y_reg_pred: np.ndarray, reg_mask: np.ndarray,
                    file_idx: np.ndarray, files: list[dict]) -> dict:
    """28 mil（B028/IR028）窗口级表现。"""
    dia_of_file = {m["index"]: m["diameter_mil"] for m in files}
    sel = np.array([dia_of_file.get(int(fi), 0.0) >= 28.0 for fi in file_idx])
    if not sel.any():
        return {"n_windows": 0}
    sub_cls_t, sub_cls_p = y_cls[sel], y_pred[sel]
    cm = confusion_matrix_np(sub_cls_t, sub_cls_p)
    res = {
        "n_windows": int(sel.sum()),
        "accuracy": accuracy(sub_cls_t, sub_cls_p),
        "macro_f1": macro_f1_from(cm),
        "confusion_matrix": cm.tolist(),
    }
    reg_sel = sel & reg_mask
    if reg_sel.any():
        e = np.abs(y_reg_pred[reg_sel] - y_reg[reg_sel]) * REG_SCALE
        res["mae_mil"] = float(e.mean())
        res["rmse_mil"] = float(np.sqrt((e ** 2).mean()))
    return res
