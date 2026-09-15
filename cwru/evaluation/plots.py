# -*- coding: utf-8 -*-
"""自动图表生成（matplotlib Agg，中文字体 Microsoft YaHei）。"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cwru.config import CLASSES, FIGURES_DIR

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, name: str) -> str:
    path = os.path.join(FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_loss_curves(history: list[dict], experiment: str, model: str) -> str:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, keys, title in (
        (axes[0], ["train_loss", "val_loss"], "总损失"),
        (axes[1], ["train_cls_loss", "val_cls_loss"], "分类损失"),
        (axes[2], ["train_reg_loss", "val_reg_loss"], "回归损失"),
    ):
        for key in keys:
            ax.plot(ep, [h[key] for h in history], label=key)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"{experiment} / {model} 损失曲线")
    return _save(fig, f"loss_curves_{experiment}_{model}.png")


def plot_val_curves(history: list[dict], experiment: str, model: str) -> str:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, [h["val_acc"] for h in history], label="val Accuracy")
    axes[0].plot(ep, [h.get("val_f1", np.nan) for h in history], label="val Macro-F1")
    axes[0].set_title("验证分类指标")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(ep, [h["val_mae"] for h in history], label="val MAE(缩放)")
    axes[1].set_title("验证回归 MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].plot(ep, [h["val_rmse"] for h in history], label="val RMSE(缩放)")
    axes[2].set_title("验证回归 RMSE")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.suptitle(f"{experiment} / {model} 验证指标曲线")
    return _save(fig, f"val_curves_{experiment}_{model}.png")


def plot_confusion_matrix(cm: list[list[int]], experiment: str, model: str,
                          level: str = "window") -> str:
    arr = np.asarray(cm, dtype=float)
    norm = arr / np.clip(arr.sum(axis=1, keepdims=True), 1e-9, None)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{int(arr[i, j])}\n{norm[i, j]:.2%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"{experiment} / {model} 混淆矩阵（{level}级）")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, f"confusion_{level}_{experiment}_{model}.png")


def plot_per_class(prf: dict, experiment: str, model: str) -> str:
    x = np.arange(len(CLASSES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, (metric, off) in enumerate((("precision", -width), ("recall", 0.0), ("f1", width))):
        vals = [prf[str(c)][metric] for c in range(len(CLASSES))]
        ax.bar(x + off, vals, width, label=metric)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{experiment} / {model} 各类别 Precision / Recall / F1")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, f"perclass_{experiment}_{model}.png")


def plot_diameter_errors(per_diameter: dict, experiment: str, model: str) -> str:
    dias = sorted(per_diameter.keys())
    mae = [per_diameter[d]["mae_mil"] for d in dias]
    rmse = [per_diameter[d]["rmse_mil"] for d in dias]
    x = np.arange(len(dias))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.2, mae, 0.4, label="MAE (mil)")
    ax.bar(x + 0.2, rmse, 0.4, label="RMSE (mil)")
    ax.set_xticks(x, dias)
    ax.set_title(f"{experiment} / {model} 各直径档位回归误差")
    ax.set_ylabel("误差 (mil)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, f"diameter_{experiment}_{model}.png")


def plot_or_clock(or_stat: dict, experiment: str, model: str) -> str:
    clocks = [c for c in ("3点钟", "6点钟", "12点钟") if c in or_stat]
    recalls = [or_stat[c].get("or_recall", np.nan) for c in clocks]
    maes = [or_stat[c].get("mae_mil", np.nan) for c in clocks]
    x = np.arange(len(clocks))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, recalls, 0.5, color="#4C72B0")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("OR 识别召回率（按钟点）")
    axes[1].bar(x, maes, 0.5, color="#DD8452")
    axes[1].set_title("OR 直径 MAE (mil)（按钟点）")
    for ax in axes:
        ax.set_xticks(x, clocks)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{experiment} / {model} 外圈位置细分")
    return _save(fig, f"or_clock_{experiment}_{model}.png")


def plot_compare_classification(rows: list[dict]) -> str:
    """rows: [{experiment, model, accuracy, macro_f1}] 10 组。"""
    labels = [f"{r['experiment']}\n{r['model']}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar(x - 0.2, [r["accuracy"] for r in rows], 0.4, label="Accuracy")
    ax.bar(x + 0.2, [r["macro_f1"] for r in rows], 0.4, label="Macro-F1")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("10 组实验分类指标对比（测试集，窗口级）")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "compare_classification.png")


def plot_compare_regression(rows: list[dict]) -> str:
    labels = [f"{r['experiment']}\n{r['model']}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar(x - 0.2, [r["mae_mil"] for r in rows], 0.4, label="MAE (mil)")
    ax.bar(x + 0.2, [r["rmse_mil"] for r in rows], 0.4, label="RMSE (mil)")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_title("10 组实验回归指标对比（测试集，窗口级）")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "compare_regression.png")


def plot_compare_window_file(rows: list[dict]) -> str:
    """rows: [{experiment, model, accuracy, file_accuracy, macro_f1, file_macro_f1}]"""
    labels = [f"{r['experiment']}\n{r['model']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    axes[0].bar(x - 0.2, [r["accuracy"] for r in rows], 0.4, label="窗口级")
    axes[0].bar(x + 0.2, [r["file_accuracy"] for r in rows], 0.4, label="文件级")
    axes[0].set_title("Accuracy：窗口级 vs 文件级")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(x - 0.2, [r["macro_f1"] for r in rows], 0.4, label="窗口级")
    axes[1].bar(x + 0.2, [r["file_macro_f1"] for r in rows], 0.4, label="文件级")
    axes[1].set_title("Macro-F1：窗口级 vs 文件级")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=8)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    return _save(fig, "compare_window_file.png")
