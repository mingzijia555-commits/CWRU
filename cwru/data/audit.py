# -*- coding: utf-8 -*-
"""数据审计：文件名解析、MAT 内部变量识别与特殊映射。"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict

import scipy.io as sio

from cwru.config import DATA_DIR

# 特殊变量映射（按项目计划固定）
SPECIAL_VARS = {
    "normal_2_99.mat": {"DE": "X099_DE_time", "FE": "X099_FE_time"},
    "12k_Fan_End_IR014_1_276.mat": {"DE": "X275_DE_time", "FE": "X275_FE_time"},
}

FAULT_CODE_TO_CLASS = {"B": "B", "IR": "IR", "OR": "OR"}


@dataclass
class FileRecord:
    filename: str
    end: str                 # "Drive" / "Fan" / "Normal"
    fault: str               # "Normal" / "IR" / "OR" / "B"
    diameter_mil: float      # 0 表示正常
    load: int                # 0-3
    or_clock: int | None     # 外圈位置钟点，仅 OR
    file_id: int
    var_de: str = ""
    var_fe: str = ""
    n_de: int = 0
    n_fe: int = 0
    in_set_101: bool = True
    in_set_109: bool = True
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_filename(filename: str) -> dict:
    """解析 CWRU 命名惯例：

    12k_Drive_End_B007_0_118.mat / 12k_Fan_End_OR007@3_2_146.mat / normal_0_97.mat
    """
    stem = os.path.splitext(filename)[0]
    if stem.startswith("normal"):
        m = re.match(r"normal_(\d+)_(\d+)$", stem)
        if not m:
            raise ValueError(f"无法解析正常数据文件名: {filename}")
        return {"end": "Normal", "fault": "Normal", "diameter_mil": 0.0,
                "load": int(m.group(1)), "or_clock": None, "file_id": int(m.group(2))}

    m = re.match(r"12k_(Drive|Fan)_End_([A-Z]+)(\d+)(?:@(\d+))?_(\d+)_(\d+)$", stem)
    if not m:
        raise ValueError(f"无法解析故障数据文件名: {filename}")
    end, fault, dia, clock, load, fid = m.groups()
    return {
        "end": end,
        "fault": FAULT_CODE_TO_CLASS[fault],
        "diameter_mil": float(dia),
        "or_clock": int(clock) if clock else None,
        "load": int(load),
        "file_id": int(fid),
    }


def identify_variables(filename: str, mat: dict) -> dict:
    """确定一个 MAT 文件使用的 DE/FE 变量名。

    规则（按计划）：
    1. 特殊映射表优先；
    2. 28 mil 文件（变量名随机）从内容中识别唯一 *_DE_time；
    3. 其余文件取唯一的 X###_DE_time / X###_FE_time。
    """
    keys = [k for k in mat if not k.startswith("__")]
    if filename in SPECIAL_VARS:
        mapping = SPECIAL_VARS[filename]
        for role, var in mapping.items():
            if var not in keys:
                raise KeyError(f"{filename}: 特殊映射变量 {var} 不存在")
        return {"DE": mapping["DE"], "FE": mapping["FE"]}

    de = [k for k in keys if k.endswith("_DE_time")]
    fe = [k for k in keys if k.endswith("_FE_time")]
    if len(de) == 1:
        return {"DE": de[0], "FE": fe[0] if len(fe) == 1 else ""}
    if len(de) > 1:
        raise ValueError(f"{filename}: 存在多个 *_DE_time 变量 {de}，需人工指定映射")
    # 无标准命名（28 mil 随机变量名文件）：从内容识别
    arrays = {k: v for k, v in mat.items()
              if not k.startswith("__") and hasattr(v, "shape") and v.ndim == 2
              and 1 in v.shape and v.size > 1000 and v.dtype.kind == "f"}
    if len(arrays) != 1:
        raise ValueError(f"{filename}: 无法从内容中识别唯一信号通道，候选={list(arrays)}")
    (var,) = arrays.keys()
    return {"DE": var, "FE": ""}


def load_record(filename: str, base_dir: str = DATA_DIR) -> FileRecord:
    """读取单个 MAT 文件并生成完整记录（含变量名与信号长度）。"""
    path = os.path.join(base_dir, filename)
    mat = sio.loadmat(path)
    info = parse_filename(filename)
    vars_ = identify_variables(filename, mat)

    n_de = int(mat[vars_["DE"]].shape[0]) if vars_["DE"] else 0
    n_fe = int(mat[vars_["FE"]].shape[0]) if vars_["FE"] else 0

    is_28mil = info["diameter_mil"] >= 28.0
    rec = FileRecord(
        filename=filename,
        end=info["end"],
        fault=info["fault"],
        diameter_mil=info["diameter_mil"],
        load=info["load"],
        or_clock=info["or_clock"],
        file_id=info["file_id"],
        var_de=vars_["DE"],
        var_fe=vars_["FE"],
        n_de=n_de,
        n_fe=n_fe,
        in_set_101=not is_28mil,
        in_set_109=True,
        extras={"n_samples_at_48k": n_de if info["end"] == "Normal" else None},
    )
    return rec


def audit_all(base_dir: str = DATA_DIR, verbose: bool = False) -> list[FileRecord]:
    """扫描数据目录，返回全部文件记录。"""
    files = sorted(f for f in os.listdir(base_dir) if f.lower().endswith(".mat"))
    records = [load_record(f, base_dir) for f in files]
    if verbose:
        for r in records:
            print(f"{r.filename:40s} end={r.end:5s} fault={r.fault:6s} dia={r.diameter_mil:4.0f} "
                  f"load={r.load} clock={r.or_clock} DE={r.var_de}({r.n_de}) FE={r.var_fe}({r.n_fe})")
    return records
