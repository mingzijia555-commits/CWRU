# -*- coding: utf-8 -*-
"""自动图表生成（matplotlib Agg，中文字体 Microsoft YaHei）。

所有绘图函数接受 fig_dir 输出目录，由调用方决定写入批次目录还是临时目录。
"""
from __future__ import annotations

import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cwru.config import CLASSES

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, fig_dir: str, name: str, tight: bool = True) -> str:
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    if tight:
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


def plot_final_overview(rows: list[dict], conclusions: dict, fig_dir: str) -> str:
    """生成适合汇报的一页式最终结果总览。

    图中统一使用窗口级指标：Macro-F1 越高越好，直径 MAE 越低越好。
    分组柱的误差线表示该分组内全部实验结果的总体标准差。
    """
    model_order = ["Cnn1d", "CnnGru", "CnnLstm"]
    model_colors = {"Cnn1d": "#4C78A8", "CnnGru": "#F58518", "CnnLstm": "#54A24B"}
    exp_order = ["dual101", "109DEFE", "101DEFE", "109DE", "101DE"]
    exp_labels = {
        "dual101": "101文件\nDE+FE双通道",
        "109DEFE": "109文件\nDE/FE单通道",
        "101DEFE": "101文件\nDE/FE单通道",
        "109DE": "109文件\n全DE",
        "101DE": "101文件\n全DE",
    }
    split_order = ["A", "B", "C"]
    plan_order = ["no_overlap", "overlap50"]
    plan_labels = {"no_overlap": "无重叠", "overlap50": "50%重叠"}

    def values(metric: str, field: str, value: str) -> list[float]:
        return [float(r[metric]) for r in rows
                if r.get(field) == value and r.get(metric) is not None]

    def mean_std(vals: list[float]) -> tuple[float, float]:
        return statistics.fmean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0

    def grouped(metric: str, field: str, order: list[str]) -> tuple[list[float], list[float]]:
        stats = [mean_std(values(metric, field, item)) for item in order]
        return [s[0] for s in stats], [s[1] for s in stats]

    fig = plt.figure(figsize=(16, 11), facecolor="#F5F7FA")
    grid = fig.add_gridspec(2, 3, left=0.06, right=0.97, bottom=0.10, top=0.76,
                           wspace=0.28, hspace=0.42)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(3)]
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", alpha=0.20, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    # 1-2. 三种模型的总体表现（每个模型 30 组）。
    model_f1, model_f1_std = grouped("window_macro_f1", "model", model_order)
    model_mae, model_mae_std = grouped("window_mae_mil", "model", model_order)
    x = np.arange(len(model_order))
    colors = [model_colors[m] for m in model_order]
    axes[0].bar(x, np.asarray(model_f1) * 100, yerr=np.asarray(model_f1_std) * 100,
                capsize=4, color=colors, width=0.62)
    axes[0].set_xticks(x, model_order)
    axes[0].set_ylim(88, 101)
    axes[0].set_ylabel("Macro-F1（%）")
    axes[0].set_title("① 模型分类表现（30组/模型）", loc="left", fontweight="bold")
    for i, val in enumerate(model_f1):
        axes[0].text(i, 88.5, f"{val:.2%}", ha="center", va="bottom", fontweight="bold")

    axes[1].bar(x, model_mae, yerr=model_mae_std, capsize=4, color=colors, width=0.62)
    axes[1].set_xticks(x, model_order)
    axes[1].set_ylabel("直径 MAE（mil，越低越好）")
    axes[1].set_title("② 模型回归表现（30组/模型）", loc="left", fontweight="bold")
    for i, val in enumerate(model_mae):
        axes[1].text(i, 0.05, f"{val:.3f}", ha="center", va="bottom",
                     color="white", fontweight="bold")

    # 3. 五种输入配置的分类/回归折中。
    exp_f1, _ = grouped("window_macro_f1", "experiment", exp_order)
    exp_mae, _ = grouped("window_mae_mil", "experiment", exp_order)
    exp_colors = ["#8E6C8A", "#2A9D8F", "#76B7B2", "#E9C46A", "#F4A261"]
    axes[2].scatter(np.asarray(exp_f1) * 100, exp_mae, s=130, c=exp_colors,
                    edgecolor="white", linewidth=1.5, zorder=3)
    for i, label in enumerate(exp_labels[e] for e in exp_order):
        offset = (7, -5) if i != 1 else (7, 5)
        axes[2].annotate(label,
                         (exp_f1[i] * 100, exp_mae[i]), xytext=offset,
                         textcoords="offset points", fontsize=8)
    axes[2].set_xlim(min(exp_f1) * 100 - 0.15, max(exp_f1) * 100 + 0.65)
    axes[2].set_xlabel("Macro-F1（%，越右越好）")
    axes[2].set_ylabel("直径 MAE（mil，越低越好）")
    axes[2].set_title("③ 输入配置的综合权衡", loc="left", fontweight="bold")
    axes[2].grid(axis="both", alpha=0.20)

    # 4. 三种文件划分揭示泛化难度。
    split_f1, split_f1_std = grouped("window_macro_f1", "split", split_order)
    split_colors = ["#59A14F", "#4E79A7", "#E15759"]
    axes[3].bar(x, np.asarray(split_f1) * 100, yerr=np.asarray(split_f1_std) * 100,
                capsize=4, color=split_colors, width=0.62)
    axes[3].set_xticks(x, [f"划分 {s}" for s in split_order])
    axes[3].set_ylim(86, 101)
    axes[3].set_ylabel("Macro-F1（%）")
    axes[3].set_title("④ 不同测试划分的泛化表现", loc="left", fontweight="bold")
    for i, val in enumerate(split_f1):
        axes[3].text(i, 86.5, f"{val:.2%}", ha="center", va="bottom", fontweight="bold")

    # 5-6. 窗口重叠策略影响（每种方案 45 组）。
    plan_f1, plan_f1_std = grouped("window_macro_f1", "window_plan", plan_order)
    plan_mae, plan_mae_std = grouped("window_mae_mil", "window_plan", plan_order)
    xp = np.arange(len(plan_order))
    plan_colors = ["#9C9C9C", "#B279A2"]
    axes[4].bar(xp, np.asarray(plan_f1) * 100, yerr=np.asarray(plan_f1_std) * 100,
                capsize=4, color=plan_colors, width=0.58)
    axes[4].set_xticks(xp, [plan_labels[p] for p in plan_order])
    axes[4].set_ylim(90, 100)
    axes[4].set_ylabel("Macro-F1（%）")
    axes[4].set_title("⑤ 窗口策略：分类", loc="left", fontweight="bold")
    for i, val in enumerate(plan_f1):
        axes[4].text(i, 90.4, f"{val:.2%}", ha="center", va="bottom", fontweight="bold")

    axes[5].bar(xp, plan_mae, yerr=plan_mae_std, capsize=4,
                color=plan_colors, width=0.58)
    axes[5].set_xticks(xp, [plan_labels[p] for p in plan_order])
    axes[5].set_ylabel("直径 MAE（mil，越低越好）")
    axes[5].set_title("⑥ 窗口策略：回归", loc="left", fontweight="bold")
    for i, val in enumerate(plan_mae):
        axes[5].text(i, 0.04, f"{val:.3f}", ha="center", va="bottom",
                     color="white", fontweight="bold")

    def readable_config(key: str) -> str:
        plan, exp, model = key.split("/")
        compact_exp = {
            "dual101": "101文件双通道",
            "109DEFE": "109文件DE/FE",
            "101DEFE": "101文件DE/FE",
            "109DE": "109文件全DE",
            "101DE": "101文件全DE",
        }[exp]
        return f"{plan_labels[plan]} + {compact_exp} + {model}"

    best_cls = readable_config(conclusions["best_classification"])
    best_cls_f1 = conclusions["best_classification_macro_f1"]
    best_reg = readable_config(conclusions["best_regression"])
    best_reg_mae = conclusions["best_regression_mae_mil"]
    fig.text(0.06, 0.955, "CWRU 轴承故障诊断 · 90组实验最终结果总览",
             fontsize=22, fontweight="bold", color="#243447")
    fig.text(0.06, 0.915,
             "3种文件划分 × 2种窗口方案 × 5种输入配置 × 3种模型；统一展示窗口级测试指标",
             fontsize=11, color="#5B6573")
    card_texts = [
        ("完整实验", f"{len(rows)} 组", "所有计划组合均有结果"),
        ("最佳分类（跨A/B/C平均）", f"Macro-F1  {best_cls_f1:.2%}", best_cls),
        ("最佳回归（跨A/B/C平均）", f"MAE  {best_reg_mae:.3f} mil", best_reg),
    ]
    card_x = [0.06, 0.37, 0.68]
    for x0, (label, value, note) in zip(card_x, card_texts):
        fig.text(x0, 0.865, label, fontsize=10, color="#5B6573")
        fig.text(x0, 0.835, value, fontsize=17, fontweight="bold", color="#243447")
        fig.text(x0, 0.812, note, fontsize=9, color="#5B6573")

    fig.text(0.06, 0.035,
             "读图结论：分类上 CnnGru 平均最好；直径回归上 CnnLstm 最好；50%重叠略有增益；划分C明显更难。"
             " 误差线为组内总体标准差。",
             fontsize=10.5, color="#243447")
    return _save(fig, fig_dir, "final_results_overview.png", tight=False)
