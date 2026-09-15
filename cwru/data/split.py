# -*- coding: utf-8 -*-
"""文件级划分：先划分文件再切窗口，固定种子 42，101/109 清单公共部分一致。"""
from __future__ import annotations

import json
import os
import random

from cwru.config import MANIFESTS_DIR, SPLIT_NAMES, SPLIT_TARGETS
from cwru.data.audit import FileRecord


def build_split(records: list[FileRecord]) -> dict[str, dict[str, str]]:
    """生成 101 与 109 两个文件集合的划分清单。

    返回 {"101": {filename: split}, "109": {...}}。
    公共文件（97 个故障 + 4 个 Normal）在两个清单中归属完全一致；
    B028/IR028 仅出现在 109 清单中，各自按 2/1/1。
    """
    rng = random.Random(42)

    faults_101 = sorted((r for r in records if r.fault != "Normal" and r.in_set_101),
                        key=lambda r: (r.end, r.fault, r.diameter_mil, r.load, r.or_clock or 0))
    normals = sorted((r for r in records if r.fault == "Normal"), key=lambda r: r.load)
    b028 = sorted((r for r in records if r.fault == "B" and r.diameter_mil >= 28.0),
                  key=lambda r: r.load)
    ir028 = sorted((r for r in records if r.fault == "IR" and r.diameter_mil >= 28.0),
                   key=lambda r: r.load)

    assert len(faults_101) == 97 and len(normals) == 4 and len(b028) == 4 and len(ir028) == 4, \
        (len(faults_101), len(normals), len(b028), len(ir028))

    split101: dict[str, str] = {}
    split109: dict[str, str] = {}

    # 97 个故障文件 → 69/14/14：按 (end, fault, diameter, or_clock) 分组，
    # 组内负载顺序打散后逐文件填充剩余缺口，总数精确且负载分布均衡
    groups: dict[tuple, list[FileRecord]] = {}
    for r in faults_101:
        key = (r.end, r.fault, r.diameter_mil, r.or_clock)
        groups.setdefault(key, []).append(r)

    remaining = list(SPLIT_TARGETS["fault_common"])

    def pick_split() -> str:
        # 剩余缺口最大的划分优先；平手按 train > val > test
        idx = max(range(3), key=lambda i: (remaining[i], -i))
        return SPLIT_NAMES[idx]

    ordered: list[FileRecord] = []
    for key in sorted(groups):
        grp = list(groups[key])
        rng.shuffle(grp)
        ordered.extend(grp)

    for r in ordered:
        s = pick_split()
        split101[r.filename] = s
        remaining[SPLIT_NAMES.index(s)] -= 1
    assert all(v == 0 for v in remaining), remaining

    # Normal / B028 / IR028 各自 2/1/1，负载顺序打散
    for grp_records, key_name, sink in ((normals, "normal", split101),
                                        (b028, "b028", split109),
                                        (ir028, "ir028", split109)):
        tgt = SPLIT_TARGETS[key_name]
        order = list(range(4))
        rng.shuffle(order)
        for rank, idx in enumerate(order):
            rec = grp_records[idx]
            if rank < tgt[0]:
                s = SPLIT_NAMES[0]
            elif rank < tgt[0] + tgt[1]:
                s = SPLIT_NAMES[1]
            else:
                s = SPLIT_NAMES[2]
            sink[rec.filename] = s

    # 109 = 101 公共部分 + B028/IR028（保证公共归属一致）
    split109.update(split101)

    return {"101": split101, "109": split109}


def save_splits(splits: dict[str, dict[str, str]]) -> None:
    for set_id, mapping in splits.items():
        path = os.path.join(MANIFESTS_DIR, f"split_{set_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_split(set_id: str) -> dict[str, str]:
    path = os.path.join(MANIFESTS_DIR, f"split_{set_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
