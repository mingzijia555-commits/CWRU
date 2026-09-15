# -*- coding: utf-8 -*-
"""predict：加载最佳权重，对指定 MAT 文件做终端推理（支持划分集与窗口方案）。"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from cwru.config import CLASSES, EXPERIMENTS, WINDOW_PLANS
from cwru.data.audit import load_record
from cwru.data.dataset import manifest_path, run_key
from cwru.data.signals import load_channel, make_windows
from cwru.evaluation.evaluate import load_best_model


def resolve_channels(exp_cfg: dict, record, channel_arg: str | None) -> list[str]:
    """确定推理通道。DEFE 实验必须显式指定 DE 或 FE。"""
    scheme = exp_cfg["channels"]
    if scheme == "dual":
        return ["DE", "FE"]
    if scheme == "defe":
        if channel_arg not in ("DE", "FE"):
            raise ValueError("该实验为 DEFE 单通道方案，请用 --channel DE 或 --channel FE 显式指定测量端")
        return [channel_arg]
    return ["DE"]


def resolve_ckpt_dir(experiment: str, model: str, split_name: str, window_plan: str,
                     ckpt_dir: str | None) -> str:
    """定位权重目录：优先显式指定，其次最新批次，最后 artifacts/runs 兼容路径。"""
    if ckpt_dir:
        return ckpt_dir
    from cwru.config import FULL_RUNS_DIR, RUNS_DIR
    if os.path.isdir(FULL_RUNS_DIR):
        batches = sorted(d for d in os.listdir(FULL_RUNS_DIR)
                         if os.path.isdir(os.path.join(FULL_RUNS_DIR, d))
                         and not d.startswith("legacy"))
        for b in reversed(batches):
            cand = os.path.join(FULL_RUNS_DIR, b, split_name, window_plan, experiment, model)
            if os.path.exists(os.path.join(cand, "best_inference.pt")):
                return cand
    return os.path.join(RUNS_DIR, f"{split_name}_{window_plan}_{experiment}_{model}")


def predict_file(experiment: str, model_name: str, mat_path: str,
                 channel: str | None = None, split_name: str = "A",
                 window_plan: str = "no_overlap", ckpt_dir: str | None = None) -> dict:
    """对单个 MAT 文件推理，返回结构化结果并打印终端摘要。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_cfg = EXPERIMENTS[experiment]
    plan = WINDOW_PLANS[window_plan]
    window_len, stride = plan["window_len"], plan["stride"]

    record = load_record(os.path.basename(mat_path), os.path.dirname(mat_path) or ".")
    roles = resolve_channels(exp_cfg, record, channel)
    channels = [load_channel(record, role) for role in roles]
    n_win = min((len(c) - window_len) // stride + 1 for c in channels)
    end = (n_win - 1) * stride + window_len
    chans = [make_windows(c[:end], window_len, stride) for c in channels]
    X = np.stack(chans, axis=1).astype(np.float32)  # [n, C, W]

    # 用训练期归一化参数（从实验清单读取，键含划分集与窗口方案）
    key = run_key(split_name, experiment, window_plan)
    mpath = manifest_path(key)
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"缺少实验清单 {mpath}，请先运行 prepare 或对应批次训练")
    with open(mpath, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    mean = np.array([n["mean"] for n in manifest["norm"]], dtype=np.float32)[None, :, None]
    std = np.array([n["std"] for n in manifest["norm"]], dtype=np.float32)[None, :, None]
    X = (X - mean) / std

    ckpt_dir = resolve_ckpt_dir(experiment, model_name, split_name, window_plan, ckpt_dir)
    model, ckpt = load_best_model(ckpt_dir, model_name, X.shape[1], device)
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        out = model(xb)
        probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
        dia = out["diameter"].cpu().numpy()

    pred_cls = int(probs.mean(axis=0).argmax())
    mean_prob = probs.mean(axis=0)
    raw_dia = float(dia.mean()) * 7.0
    final_dia = 0.0 if pred_cls == 0 else raw_dia

    per_window_classes = probs.argmax(axis=1)
    class_share = {CLASSES[c]: float((per_window_classes == c).mean()) for c in range(4)}

    print("=" * 64)
    print(f"划分/窗口 : {split_name} / {window_plan} ({window_len}/{stride})")
    print(f"实验/模型 : {experiment} / {model_name}")
    print(f"权重目录  : {ckpt_dir}")
    print(f"文件      : {os.path.basename(mat_path)}")
    print(f"通道      : {'+'.join(roles)}  "
          f"({'、'.join(record.var_de if r == 'DE' else record.var_fe for r in roles)})")
    print(f"窗口数    : {n_win}")
    print(f"预测类别  : {CLASSES[pred_cls]}")
    for c in range(4):
        print(f"  P({CLASSES[c]:6s}) = {mean_prob[c]:.4f}")
    print(f"回归原始直径: {raw_dia:.2f} mil")
    print(f"最终显示直径: {final_dia:.2f} mil"
          + ("（预测为 Normal，直径置 0）" if pred_cls == 0 else ""))
    print("窗口统计摘要:")
    print(f"  窗口类别占比: { {k: round(v, 3) for k, v in class_share.items()} }")
    print(f"  直径预测 mean={float(dia.mean()) * 7.0:.2f} mil, "
          f"std={float(dia.std()) * 7.0:.2f} mil, median={float(np.median(dia)) * 7.0:.2f} mil")
    print("=" * 64)

    return {
        "split_set": split_name, "window_plan": window_plan,
        "experiment": experiment, "model": model_name, "file": mat_path,
        "channels": roles, "n_windows": n_win, "pred_class": CLASSES[pred_cls],
        "probs": mean_prob.tolist(), "raw_diameter_mil": raw_dia,
        "final_diameter_mil": final_dia, "class_share": class_share,
    }
