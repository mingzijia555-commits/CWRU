# -*- coding: utf-8 -*-
"""实验数组构建与 Dataset。

先按文件清单划分 train/val/test，再在各集合内部切窗口：
- 窗口长度固定 1024，步长由窗口方案决定（无重叠 1024 / 50% 重叠 512）；
- 归一化参数仅由该组合训练集窗口计算；
- 实验数组按需在内存中构建，不写入缓存或额外清单。
"""
from __future__ import annotations

import numpy as np
import torch

from cwru.config import (CLASSES, REG_SCALE, SIGNAL_SR,
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
        # 明确指定回归标签为 float32，避免 DataLoader 将 Python float
        # 默认拼接成 float64，进而与模型的 float32 输出发生 dtype 冲突。
        return (torch.from_numpy(self.X[i]), int(self.y_cls[i]),
                torch.tensor(self.y_reg[i], dtype=torch.float32),
                bool(self.reg_mask[i]))
