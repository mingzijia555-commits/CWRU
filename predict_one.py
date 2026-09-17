# -*- coding: utf-8 -*-
"""单文件预测演示：修改下面的文件路径和实验配置后直接运行。"""
from __future__ import annotations

from cwru.inference.predict import predict_file


MAT_FILE = "CaseWesternReserveUniversityData/normal_0_97.mat"
EXPERIMENT = "101DE"
MODEL = "Cnn1d"
SPLIT = "A"
WINDOW = "no_overlap"
CHANNEL = None


if __name__ == "__main__":
    predict_file(EXPERIMENT, MODEL, MAT_FILE, CHANNEL, SPLIT, WINDOW)
