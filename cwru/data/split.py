# -*- coding: utf-8 -*-
"""读取已经确定的 A/B/C 文件划分。

划分名单保存在项目根目录的 splits/A、splits/B、splits/C 中。
训练、验证和测试先按完整 MAT 文件划分，再在各集合内部切窗口。
划分模块只读取固定的 split_101.json / split_109.json，不生成额外的
metadata.csv 或 balance.csv 汇总表；类别和条件信息在运行时由 MAT 文件审计得到。
"""
from __future__ import annotations

import json
import os

from cwru.config import SPLIT_NAMES, SPLITS_DIR


def load_split(file_set: str, split_name: str) -> dict[str, str]:
    """读取一个文件集合（101 或 109）的一套固定划分。"""
    path = os.path.join(SPLITS_DIR, split_name, f"split_{file_set}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"固定划分不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    unknown = set(mapping.values()) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"{path} 中出现未知划分名称：{sorted(unknown)}")
    counts = {name: sum(value == name for value in mapping.values())
              for name in SPLIT_NAMES}
    expected = {"101": (71, 15, 15), "109": (75, 17, 17)}[file_set]
    actual = tuple(counts[name] for name in SPLIT_NAMES)
    if actual != expected:
        raise ValueError(f"{path} 数量为 {actual}，应为 {expected}")
    return mapping


def load_all_splits(split_name: str) -> dict[str, dict[str, str]]:
    """读取同一套 A/B/C 划分中的 101 和 109 文件名单。"""
    splits = {
        "101": load_split("101", split_name),
        "109": load_split("109", split_name),
    }
    common = set(splits["101"]) - set(splits["109"])
    if common:
        raise ValueError(f"{split_name} 的 101 文件不全属于 109 文件集合")
    for filename, part in splits["101"].items():
        if splits["109"].get(filename) != part:
            raise ValueError(f"{split_name} 中 {filename} 的 101/109 划分不一致")
    return splits
