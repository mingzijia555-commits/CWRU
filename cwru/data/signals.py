# -*- coding: utf-8 -*-
"""信号加载与重采样。"""
from __future__ import annotations

import os

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

from cwru.config import DATA_DIR, NORMAL_SR, SIGNAL_SR
from cwru.data.audit import FileRecord


def load_channel(record: FileRecord, role: str, base_dir: str = DATA_DIR) -> np.ndarray:
    """按记录读取指定通道（DE/FE），Normal 从 48 kHz 抗混叠降采样到 12 kHz。

    resample_poly(up=1, down=4) 为整数倍抽取并内置抗混叠 FIR。
    """
    if role not in ("DE", "FE"):
        raise ValueError(f"未知通道角色: {role}")
    var = record.var_de if role == "DE" else record.var_fe
    if not var:
        raise ValueError(f"{record.filename}: 无 {role} 通道")
    mat = sio.loadmat(os.path.join(base_dir, record.filename))
    signal = np.asarray(mat[var], dtype=np.float64).ravel()
    if record.end == "Normal":
        assert NORMAL_SR == 4 * SIGNAL_SR
        signal = resample_poly(signal, 1, 4)
    return signal


def make_windows(signal: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    """滑窗切片：形状 [n_windows, window_len]，尾部不足一个窗口的部分丢弃。

    stride == window_len 为无重叠；stride = window_len // 2 为 50% 重叠。
    重叠只发生在同一信号内部，跨集合的窗口隔离由文件级划分保证。
    """
    if stride <= 0 or stride > window_len:
        raise ValueError(f"非法步长 stride={stride}（需 0 < stride <= window_len={window_len}）")
    n = (len(signal) - window_len) // stride + 1
    if n <= 0:
        raise ValueError(f"信号长度 {len(signal)} 不足一个窗口 {window_len}")
    starts = np.arange(n, dtype=np.int64) * stride
    out = np.stack([signal[s: s + window_len] for s in starts])
    return out.astype(np.float32)
