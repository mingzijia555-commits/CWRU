# -*- coding: utf-8 -*-
"""项目全局配置：路径、常量、实验定义。"""
from __future__ import annotations

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "CaseWesternReserveUniversityData")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MANIFESTS_DIR = os.path.join(ARTIFACTS_DIR, "manifests")
RUNS_DIR = os.path.join(ARTIFACTS_DIR, "runs")
METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
CACHE_DIR = os.path.join(ARTIFACTS_DIR, "cache")
FULL_RUNS_DIR = os.path.join(ARTIFACTS_DIR, "full_runs")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")

# 信号与窗口
SIGNAL_SR = 12000          # 统一采样率
NORMAL_SR = 48000          # Normal 原始采样率
WINDOW_LEN = 1024

# 窗口方案：两种步长，窗口长度固定 1024
WINDOW_PLANS = {
    "no_overlap": {"window_len": WINDOW_LEN, "stride": WINDOW_LEN,
                   "display": "无重叠 1024/1024"},
    "overlap50":  {"window_len": WINDOW_LEN, "stride": WINDOW_LEN // 2,
                   "display": "50%重叠 1024/512"},
}
WINDOW_PLAN_ORDER = ["no_overlap", "overlap50"]

# 任务
SEED = 42
TRAIN_SEED = 42            # 正式 90 组统一训练种子（衡量划分敏感性，非初始化方差）
CLASSES = ["Normal", "IR", "OR", "B"]          # 分类顺序
DIAMETERS_MIL = [7.0, 14.0, 21.0, 28.0]
REG_SCALE = 7.0            # 回归标签 = diameter_mil / 7
OR_CLOCKS = [3, 6, 12]

# 训练超参数
BATCH_SIZE = 128
MAX_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 4            # ReduceLROnPlateau 耐心值
LR_FACTOR = 0.5
ES_PATIENCE = 12           # EarlyStopping 耐心值

MODELS = ["Cnn1d", "CnnGru", "CnnLstm"]

# 实验定义：files=文件集合，channels=通道方案
#   dual  : [DE, FE] 双通道（仅 101 文件）
#   defe  : 单通道，驱动端故障取 DE，风扇端故障取 FE，正常取 DE
#   de    : 单通道，全部取 DE
EXPERIMENTS = {
    "dual101": {"files": "101", "channels": "dual", "display": "双端(101文件,DE+FE)"},
    "109DEFE": {"files": "109", "channels": "defe", "display": "109文件 DE/FE 单通道"},
    "101DEFE": {"files": "101", "channels": "defe", "display": "101文件 DE/FE 单通道"},
    "109DE":   {"files": "109", "channels": "de",   "display": "109文件 全DE"},
    "101DE":   {"files": "101", "channels": "de",   "display": "101文件 全DE"},
}
EXPERIMENT_ORDER = ["dual101", "109DEFE", "101DEFE", "109DE", "101DE"]

# 文件划分
SPLIT_SETS = ["A", "B", "C"]
SPLIT_NAMES = ["train", "val", "test"]

# 数量硬约束
GROUP4_TARGETS = (2, 1, 1)         # Normal / B028 / IR028 各 4 文件
SPLIT_TARGETS_101 = (71, 15, 15)   # 97 故障 + 4 Normal
SPLIT_TARGETS_109 = (75, 17, 17)   # 101 公共 + B028 + IR028

# 正式实验总数
N_FORMAL_RUNS = len(SPLIT_SETS) * len(WINDOW_PLAN_ORDER) * len(EXPERIMENT_ORDER) * len(MODELS)


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, MANIFESTS_DIR, RUNS_DIR, METRICS_DIR, FIGURES_DIR,
              CACHE_DIR, FULL_RUNS_DIR, SPLITS_DIR):
        os.makedirs(d, exist_ok=True)
