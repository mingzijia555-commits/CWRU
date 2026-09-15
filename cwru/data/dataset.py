# -*- coding: utf-8 -*-
"""实验数组构建与 Dataset。

每个「划分集 × 窗口方案 × 输入方案」组合的数据布局在 prepare 阶段固化：
- 先按文件清单划分 train/val/test，再在各集合内部切窗口；
- 窗口长度固定 1024，步长由窗口方案决定（无重叠 1024 / 50% 重叠 512）；
- 归一化参数仅由该组合训练集窗口计算；
- 实验数组按需在内存中构建，不写入窗口缓存。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from cwru.config import (CLASSES, MANIFESTS_DIR, REG_SCALE, SIGNAL_SR,
                         WINDOW_LEN, WINDOW_PLANS)
from cwru.data.audit import FileRecord
from cwru.data.signals import load_channel, make_windows


def channel_roles_for(exp_cfg: dict, record: FileRecord) -> list[str]:
    """返回该文件在本实验中使用的通道角色序列（决定数组通道顺序）。"""
    scheme = exp_cfg["channels"]
    if scheme == "dual":
        return ["DE", "FE"]
    if scheme == "defe":
        if record.end == "Fan":
            return ["FE"]
        return ["DE"]  # 驱动端故障与 Normal 均取 DE
    if scheme == "de":
        return ["DE"]
    raise ValueError(f"未知通道方案: {scheme}")


def run_key(split_name: str, exp_id: str, window_plan: str) -> str:
    """实验清单的复合键：划分集 + 输入方案 + 窗口方案。"""
    return f"{split_name}_{exp_id}_{window_plan}"


def build_experiment_arrays(exp_id: str, exp_cfg: dict, records: list[FileRecord],
                            split_of: dict[str, str],
                            window_plan: str = "no_overlap") -> dict:
    """构建实验级窗口数组并计算归一化/类别权重。

    返回 dict(X, y_cls, y_reg, reg_mask, file_idx, files, norm, class_weights, counts)
    """
    plan = WINDOW_PLANS[window_plan]
    window_len, stride = plan["window_len"], plan["stride"]
    n_channels = 2 if exp_cfg["channels"] == "dual" else 1

    xs: list[np.ndarray] = []
    ys_cls: list[np.ndarray] = []
    ys_reg: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    file_ids: list[np.ndarray] = []
    files_meta: list[dict] = []

    used = [r for r in records if r.filename in split_of]
    assert len(used) == len(split_of), (len(used), len(split_of))

    for fi, rec in enumerate(sorted(used, key=lambda r: r.filename)):
        roles = channel_roles_for(exp_cfg, rec)
        channels = [load_channel(rec, role) for role in roles]
        n_win = min((len(c) - window_len) // stride + 1 for c in channels)
        if n_win <= 0:
            raise ValueError(f"{rec.filename}: 信号过短，无法切窗")
        end = (n_win - 1) * stride + window_len
        chans = [make_windows(c[:end], window_len, stride) for c in channels]
        x = np.stack(chans, axis=1)                       # [n_win, n_channels, 1024]

        cls_idx = CLASSES.index(rec.fault)
        y_cls = np.full(n_win, cls_idx, dtype=np.int64)
        y_reg = np.full(n_win, rec.diameter_mil / REG_SCALE, dtype=np.float32)
        mask = np.ones(n_win, dtype=bool)                 # Normal 不参与回归
        if rec.fault == "Normal":
            mask[:] = False
        y_reg[~mask] = 0.0

        xs.append(x)
        ys_cls.append(y_cls)
        ys_reg.append(y_reg)
        masks.append(mask)
        file_ids.append(np.full(n_win, fi, dtype=np.int32))
        files_meta.append({
            "index": fi,
            "filename": rec.filename,
            "split": split_of[rec.filename],
            "fault": rec.fault,
            "end": rec.end,
            "diameter_mil": rec.diameter_mil,
            "load": rec.load,
            "or_clock": rec.or_clock,
            "channels": roles,
            "n_windows": int(n_win),
        })

    X = np.concatenate(xs, axis=0)
    y_cls = np.concatenate(ys_cls, axis=0)
    y_reg = np.concatenate(ys_reg, axis=0)
    reg_mask = np.concatenate(masks, axis=0)
    file_idx = np.concatenate(file_ids, axis=0)

    # 归一化参数：仅训练集窗口
    train_file_idx = {m["index"] for m in files_meta if m["split"] == "train"}
    tr_sel = np.isin(file_idx, list(train_file_idx))
    norm = []
    for c in range(n_channels):
        vals = X[tr_sel, c, :]
        mean = float(vals.mean(dtype=np.float64))
        std = float(vals.std(dtype=np.float64))
        if std < 1e-8:
            std = 1.0
        norm.append({"mean": mean, "std": std})
    X = (X - np.array([n["mean"] for n in norm], dtype=np.float32)[None, :, None]) \
        / np.array([n["std"] for n in norm], dtype=np.float32)[None, :, None]
    X = X.astype(np.float32)

    # 类别权重：1/sqrt(训练窗口数)，归一化到均值 1（仅训练集）
    train_cls = y_cls[tr_sel]
    counts = [int((train_cls == c).sum()) for c in range(len(CLASSES))]
    w = np.array([1.0 / np.sqrt(max(n, 1)) for n in counts], dtype=np.float64)
    w = w / w.mean()
    class_weights = w.astype(np.float32).tolist()

    return {
        "X": X,
        "y_cls": y_cls,
        "y_reg": y_reg,
        "reg_mask": reg_mask,
        "file_idx": file_idx,
        "files": files_meta,
        "norm": norm,
        "class_weights": class_weights,
        "train_class_counts": counts,
        "sr": SIGNAL_SR,
        "window_len": window_len,
        "stride": stride,
    }


def manifest_path(key: str) -> str:
    return os.path.join(MANIFESTS_DIR, f"experiment_{key}.json")


def get_experiment_arrays(split_name: str, exp_id: str, exp_cfg: dict,
                          records: list[FileRecord], split_of: dict[str, str],
                          window_plan: str = "no_overlap") -> dict:
    """直接构建实验数组，并更新对应实验清单。"""
    key = run_key(split_name, exp_id, window_plan)
    arrays = build_experiment_arrays(exp_id, exp_cfg, records, split_of, window_plan)
    save_experiment_manifest(key, split_name, exp_id, exp_cfg, window_plan, arrays, split_of)
    return arrays


def save_experiment_manifest(key: str, split_name: str, exp_id: str, exp_cfg: dict,
                             window_plan: str, arrays: dict, split_of: dict[str, str]) -> str:
    manifest = {
        "key": key,
        "split_set": split_name,
        "experiment": exp_id,
        "window_plan": window_plan,
        "config": exp_cfg,
        "files": arrays["files"],
        "norm": arrays["norm"],
        "class_weights": arrays["class_weights"],
        "train_class_counts": arrays["train_class_counts"],
        "sr": arrays["sr"],
        "window_len": arrays["window_len"],
        "stride": arrays["stride"],
        "split_file": f"splits/{split_name}/split_{exp_cfg['files']}.json",
        "split": split_of,
    }
    path = manifest_path(key)
    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


class CwruDataset:
    """基于已构建数组的 Torch 数据集（数据已归一化）。"""

    def __init__(self, arrays: dict, split: str | None = None):
        if split is None:
            sel = np.arange(len(arrays["y_cls"]))
        else:
            file_split = np.array([m["split"] for m in arrays["files"]])
            sel = np.isin(arrays["file_idx"], np.where(file_split == split)[0])
        self.X = arrays["X"][sel]
        self.y_cls = arrays["y_cls"][sel]
        self.y_reg = arrays["y_reg"][sel]
        self.reg_mask = arrays["reg_mask"][sel]
        self.file_idx = arrays["file_idx"][sel]

    def __len__(self) -> int:
        return len(self.y_cls)

    def __getitem__(self, i: int):
        return (torch.from_numpy(self.X[i]), int(self.y_cls[i]),
                float(self.y_reg[i]), bool(self.reg_mask[i]))


def subset_arrays(arrays: dict, split: str) -> dict:
    """按划分名抽取子数组（评估/指标计算使用）。"""
    file_split = np.array([m["split"] for m in arrays["files"]])
    sel = np.isin(arrays["file_idx"], np.where(file_split == split)[0])
    return {
        "X": arrays["X"][sel],
        "y_cls": arrays["y_cls"][sel],
        "y_reg": arrays["y_reg"][sel],
        "reg_mask": arrays["reg_mask"][sel],
        "file_idx": arrays["file_idx"][sel],
    }
