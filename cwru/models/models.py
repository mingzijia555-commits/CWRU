# -*- coding: utf-8 -*-
"""模型定义：Cnn1d 与 CnnGru，均输出四分类 logits + 一个连续故障直径。"""
from __future__ import annotations

import torch
import torch.nn as nn

N_CLASSES = 4
FEATURE_DIM = 128


def _cnn_trunk(in_channels: int) -> nn.Sequential:
    """三个 Conv1d 模块：通道 32/64/128，卷积核 7/5/3，BN+ReLU+MaxPool。"""
    return nn.Sequential(
        nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
        nn.BatchNorm1d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
    )


def _heads(feature_dim: int = FEATURE_DIM) -> nn.ModuleDict:
    return nn.ModuleDict({
        "cls": nn.Linear(feature_dim, N_CLASSES),
        "reg": nn.Linear(feature_dim, 1),
    })


class Cnn1d(nn.Module):
    """纯 CNN：自适应平均池化得到共享特征。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.trunk = _cnn_trunk(in_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.heads = _heads()

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return self.pool(h).squeeze(-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.extract(x)
        return {"logits": self.heads["cls"](f), "diameter": self.heads["reg"](f).squeeze(-1)}


class CnnGru(nn.Module):
    """相同 CNN 提取局部特征 + 单层双向 GRU（单方向隐藏 64），拼接双向状态作为共享特征。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.trunk = _cnn_trunk(in_channels)
        self.gru = nn.GRU(input_size=FEATURE_DIM, hidden_size=64, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.heads = _heads(feature_dim=128)  # 64 * 2 双向拼接

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)                     # [B, 128, L]
        seq = h.transpose(1, 2)               # [B, L, 128]
        _, hn = self.gru(seq)                 # hn: [2, B, 64]
        return torch.cat([hn[0], hn[1]], dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.extract(x)
        return {"logits": self.heads["cls"](f), "diameter": self.heads["reg"](f).squeeze(-1)}


def build_model(name: str, in_channels: int = 1) -> nn.Module:
    if name == "Cnn1d":
        return Cnn1d(in_channels)
    if name == "CnnGru":
        return CnnGru(in_channels)
    raise ValueError(f"未知模型: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
