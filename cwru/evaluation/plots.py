# -*- coding: utf-8 -*-
"""自动图表生成（matplotlib Agg，中文字体 Microsoft YaHei）。

所有绘图函数接受 fig_dir 输出目录，由调用方决定写入批次目录还是临时目录。
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cwru.config import CLASSES

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, fig_dir: str, name: str) -> str:
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _title(experiment: str, model: str, split_name: str = "", window_plan: str = "") -> str:
    tag = " / ".join(t for t in (split_name, window_plan, experiment, model) if t)
    return tag


def plot_loss_curves(history: list[dict], fig_dir: str, experiment: str, model: str,
                     split_name: str = "", window_plan: str = "") -> str:
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
    fig.suptitle(f"{_title(experiment, model, split_name, window_plan)} 损失曲线")
    return _save(fig, fig_dir, "loss_curves.png")


def plot_val_curves(history: list[dict], fig_dir: str, experiment: str, model: str,
                    split_name: str = "", window_plan: str = "") -> str:
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
    fig.suptitle(f"{_title(experiment, model, split_name, window_plan)} 验证指标曲线")
    return _save(fig, fig_dir, "val_curves.png")


def plot_confusion_matrix(cm: list[list[int]], fig_dir: str, experiment: str, model: str,
                          level: str = "window", split_name: str = "", window_plan: str = "") -> str:
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
    ax.set_title(f"{_title(experiment, model, split_name, window_plan)} 混淆矩阵（{level}级）")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, fig_dir, f"confusion_{level}.png")


def plot_per_class(prf: dict, fig_dir: str, experiment: str, model: str,
                   split_name: str = "", window_plan: str = "") -> str:
    x = np.arange(len(CLASSES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for metric, off in (("precision", -width), ("recall", 0.0), ("f1", width)):
        vals = [prf[str(c)][metric] for c in range(len(CLASSES))]
        ax.bar(x + off, vals, width, label=metric)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{_title(experiment, model, split_name, window_plan)} 各类别 Precision / Recall / F1")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, fig_dir, "perclass.png")


def plot_diameter_errors(per_diameter: dict, fig_dir: str, experiment: str, model: str,
                         split_name: str = "", window_plan: str = "") -> str:
    dias = sorted(per_diameter.keys())
    mae = [per_diameter[d]["mae_mil"] for d in dias]
    rmse = [per_diameter[d]["rmse_mil"] for d in dias]
    x = np.arange(len(dias))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.2, mae, 0.4, label="MAE (mil)")
    ax.bar(x + 0.2, rmse, 0.4, label="RMSE (mil)")
    ax.set_xticks(x, dias)
    ax.set_title(f"{_title(experiment, model, split_name, window_plan)} 各直径档位回归误差")
    ax.set_ylabel("误差 (mil)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, fig_dir, "diameter.png")


def plot_or_clock(or_stat: dict, fig_dir: str, experiment: str, model: str,
                  split_name: str = "", window_plan: str = "") -> str:
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
    fig.suptitle(f"{_title(experiment, model, split_name, window_plan)} 外圈位置细分")
    return _save(fig, fig_dir, "or_clock.png")


def _grouped_bar(fig_dir: str, name: str, title: str, ylabel: str,
                 labels: list[str], series: list[tuple[str, list[float]]],
                 ylim: tuple[float, float] | None = None, figsize=(14, 5.0)) -> str:
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=figsize)
    for k, (label, vals) in enumerate(series):
        ax.bar(x + (k - (len(series) - 1) / 2) * width, vals, width, label=label)
    ax.set_xticks(x, labels, fontsize=6, rotation=90)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, fig_dir, name)


def _config_keys(rows: list[dict], group_by: tuple[str, ...]) -> list[tuple]:
    seen, out = set(), []
    for r in rows:
        key = tuple(r.get(k) for k in group_by)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _lookup(rows: list[dict], group_by: tuple[str, ...], key: tuple, metric: str):
    for r in rows:
        if tuple(r.get(k) for k in group_by) == key:
            v = r.get(metric)
            return float(v) if v is not None else float("nan")
    return float("nan")


def _label(key: tuple) -> str:
    return "|".join(str(k) for k in key)


def plot_split_comparison(rows: list[dict], fig_dir: str) -> list[str]:
    """对比 A/B/C 三套划分：按 (窗口, 输入, 模型) 分组。"""
    group_by = ("window_plan", "experiment", "model")
    keys = _config_keys(rows, group_by)
    labels = [_label(k) for k in keys]
    paths = []
    for metric, name, title, ylim in (
        ("file_accuracy", "split_comparison_accuracy.png", "三套划分对比：文件级 Accuracy", (0, 1.05)),
        ("file_macro_f1", "split_comparison_f1.png", "三套划分对比：文件级 Macro-F1", (0, 1.05)),
        ("file_mae_mil", "split_comparison_regression.png", "三套划分对比：文件级直径 MAE", None),
    ):
        series = [(f"划分{s}", [_lookup(rows, ("split",) + group_by, (s,) + k, metric) for k in keys])
                  for s in ("A", "B", "C")]
        paths.append(_grouped_bar(fig_dir, name, title, metric, labels, series, ylim))
    return paths


def plot_model_comparison(rows: list[dict], fig_dir: str) -> list[str]:
    """对比 Cnn1d / CnnGru / CnnLstm：按 (划分, 窗口, 输入) 分组。"""
    group_by = ("split", "window_plan", "experiment")
    keys = _config_keys(rows, group_by)
    labels = [_label(k) for k in keys]
    paths = []
    for metric, name, title, ylim in (
        ("window_accuracy", "model_comparison_classification.png", "模型对比：窗口级 Accuracy", (0, 1.05)),
        ("window_macro_f1", "model_comparison_f1.png", "模型对比：窗口级 Macro-F1", (0, 1.05)),
        ("window_mae_mil", "model_comparison_regression.png", "模型对比：窗口级直径 MAE", None),
    ):
        series = [(m, [_lookup(rows, ("model",) + group_by, (m,) + k, metric) for k in keys])
                  for m in ("Cnn1d", "CnnGru", "CnnLstm")]
        paths.append(_grouped_bar(fig_dir, name, title, metric, labels, series, ylim))
    return paths


def plot_overlap_comparison(rows: list[dict], fig_dir: str) -> list[str]:
    """对比无重叠与 50% 重叠：按 (划分, 输入, 模型) 分组。"""
    group_by = ("split", "experiment", "model")
    keys = _config_keys(rows, group_by)
    labels = [_label(k) for k in keys]
    paths = []
    for metric, name, title, ylim in (
        ("window_accuracy", "overlap_comparison_accuracy.png", "重叠对比：窗口级 Accuracy", (0, 1.05)),
        ("window_macro_f1", "overlap_comparison_f1.png", "重叠对比：窗口级 Macro-F1", (0, 1.05)),
        ("window_mae_mil", "overlap_comparison_regression.png", "重叠对比：窗口级直径 MAE", None),
    ):
        series = [(plan, [_lookup(rows, ("window_plan",) + group_by, (plan,) + k, metric) for k in keys])
                  for plan in ("no_overlap", "overlap50")]
        paths.append(_grouped_bar(fig_dir, name, title, metric, labels, series, ylim))
    return paths

