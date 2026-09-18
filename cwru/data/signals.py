# -*- coding: utf-8 -*-
"""信号加载与重采样。"""
from __future__ import annotations

import os

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

from cwru.config import DATA_DIR
from cwru.data.audit import FileRecord


def load_channel(record: FileRecord, role: str, base_dir: str = DATA_DIR) -> np.ndarray:
    """按记录读取指定通道（DE/FE），Normal 从 48 kHz 抗混叠降采样到 12 kHz。

    resample_poly(up=1, down=4) 为整数倍抽取并内置抗混叠 FIR。
    """
    var = record.var_de if role == "DE" else record.var_fe
    mat = sio.loadmat(os.path.join(base_dir, record.filename))
    signal = np.asarray(mat[var], dtype=np.float64).ravel()
    if record.end == "Normal":
        signal = resample_poly(signal, 1, 4)
    return signal


def make_windows(signal: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    """滑窗切片：形状 [n_windows, window_len]，尾部不足一个窗口的部分丢弃。

    stride == window_len 为无重叠；stride = window_len // 2 为 50% 重叠。
    重叠只发生在同一信号内部，跨集合的窗口隔离由文件级划分保证。
    """
    n = (len(signal) - window_len) // stride + 1
    starts = np.arange(n, dtype=np.int64) * stride
    out = np.stack([signal[s: s + window_len] for s in starts])
    return out.astype(np.float32)
