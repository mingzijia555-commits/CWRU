# -*- coding: utf-8 -*-
"""A/B/C 三套固定文件划分的定义、校验与读写。

划分不在运行时随机生成。每套划分的验证集和测试集文件名在本模块中固定，
训练集由对应文件集合扣除验证集和测试集后得到。这样代码、JSON 快照与正式
批次使用的是同一套可审计名单。
"""
from __future__ import annotations

import csv
import json
import os

from cwru.config import (GROUP4_TARGETS, SPLIT_NAMES, SPLIT_SETS,
                         SPLIT_TARGETS_101, SPLIT_TARGETS_109, SPLITS_DIR)
from cwru.data.audit import FileRecord


# 101 公共集合：每套固定 15 个验证文件和 15 个测试文件；其余 71 个为训练集。
_FIXED_101 = {
    "A": {
        "val": (
            "12k_Drive_End_B007_0_118.mat",
            "12k_Drive_End_B021_1_223.mat",
            "12k_Drive_End_IR014_2_171.mat",
            "12k_Drive_End_OR007@6_0_130.mat",
            "12k_Drive_End_OR007@6_2_132.mat",
            "12k_Drive_End_OR014@6_3_200.mat",
            "12k_Drive_End_OR021@12_0_258.mat",
            "12k_Fan_End_B007_2_284.mat",
            "12k_Fan_End_B021_1_291.mat",
            "12k_Fan_End_IR014_1_276.mat",
            "12k_Fan_End_IR021_3_273.mat",
            "12k_Fan_End_OR007@3_3_301.mat",
            "12k_Fan_End_OR007@6_2_296.mat",
            "12k_Fan_End_OR021@3_3_318.mat",
            "normal_0_97.mat",
        ),
        "test": (
            "12k_Drive_End_B014_2_187.mat",
            "12k_Drive_End_IR007_0_105.mat",
            "12k_Drive_End_IR021_2_211.mat",
            "12k_Drive_End_OR007@12_1_158.mat",
            "12k_Drive_End_OR007@3_2_146.mat",
            "12k_Drive_End_OR007@6_3_133.mat",
            "12k_Drive_End_OR021@3_2_248.mat",
            "12k_Drive_End_OR021@6_1_235.mat",
            "12k_Fan_End_B014_0_286.mat",
            "12k_Fan_End_B014_1_287.mat",
            "12k_Fan_End_B014_2_288.mat",
            "12k_Fan_End_IR007_1_279.mat",
            "12k_Fan_End_OR007@12_3_307.mat",
            "12k_Fan_End_OR014@3_0_310.mat",
            "normal_2_99.mat",
        ),
    },
    "B": {
        "val": (
            "12k_Drive_End_B014_2_187.mat",
            "12k_Drive_End_OR007@12_1_158.mat",
            "12k_Drive_End_OR007@3_2_146.mat",
            "12k_Drive_End_OR007@6_3_133.mat",
            "12k_Drive_End_OR014@6_0_197.mat",
            "12k_Drive_End_OR021@12_1_259.mat",
            "12k_Drive_End_OR021@3_0_246.mat",
            "12k_Fan_End_B014_3_289.mat",
            "12k_Fan_End_B021_2_292.mat",
            "12k_Fan_End_IR007_3_281.mat",
            "12k_Fan_End_IR014_0_274.mat",
            "12k_Fan_End_IR021_2_272.mat",
            "12k_Fan_End_OR007@3_0_298.mat",
            "12k_Fan_End_OR014@3_0_310.mat",
            "normal_1_98.mat",
        ),
        "test": (
            "12k_Drive_End_B007_2_120.mat",
            "12k_Drive_End_B021_0_222.mat",
            "12k_Drive_End_IR007_3_108.mat",
            "12k_Drive_End_IR014_0_169.mat",
            "12k_Drive_End_IR021_1_210.mat",
            "12k_Drive_End_OR014@6_3_200.mat",
            "12k_Drive_End_OR021@3_1_247.mat",
            "12k_Drive_End_OR021@6_3_237.mat",
            "12k_Fan_End_B007_2_284.mat",
            "12k_Fan_End_B021_1_291.mat",
            "12k_Fan_End_IR014_1_276.mat",
            "12k_Fan_End_OR007@12_2_306.mat",
            "12k_Fan_End_OR007@6_1_295.mat",
            "12k_Fan_End_OR021@3_1_316.mat",
            "normal_3_100.mat",
        ),
    },
    "C": {
        "val": (
            "12k_Drive_End_B007_1_119.mat",
            "12k_Drive_End_IR007_3_108.mat",
            "12k_Drive_End_IR014_0_169.mat",
            "12k_Drive_End_OR007@12_2_159.mat",
            "12k_Drive_End_OR007@3_2_146.mat",
            "12k_Drive_End_OR014@6_1_198.mat",
            "12k_Drive_End_OR021@12_2_260.mat",
            "12k_Drive_End_OR021@3_1_247.mat",
            "12k_Fan_End_B014_1_287.mat",
            "12k_Fan_End_B021_3_293.mat",
            "12k_Fan_End_IR007_0_278.mat",
            "12k_Fan_End_OR007@12_1_305.mat",
            "12k_Fan_End_OR007@6_3_297.mat",
            "12k_Fan_End_OR021@3_2_317.mat",
            "normal_3_100.mat",
        ),
        "test": (
            "12k_Drive_End_B014_0_185.mat",
            "12k_Drive_End_B021_2_224.mat",
            "12k_Drive_End_IR007_1_106.mat",
            "12k_Drive_End_IR007_2_107.mat",
            "12k_Drive_End_IR021_3_212.mat",
            "12k_Drive_End_OR007@6_1_131.mat",
            "12k_Drive_End_OR021@6_0_234.mat",
            "12k_Fan_End_B007_0_282.mat",
            "12k_Fan_End_IR014_3_277.mat",
            "12k_Fan_End_IR021_3_273.mat",
            "12k_Fan_End_OR007@12_0_302.mat",
            "12k_Fan_End_OR007@3_1_299.mat",
            "12k_Fan_End_OR007@6_0_294.mat",
            "12k_Fan_End_OR014@3_2_311.mat",
            "normal_1_98.mat",
        ),
    },
}


# 109 集合独有的 28 mil 文件；未列入 val/test 的四个文件自动进入训练集。
_FIXED_109_EXTRA = {
    "A": {
        "val": ("12k_Drive_End_B028_3_3008.mat", "12k_Drive_End_IR028_0_3001.mat"),
        "test": ("12k_Drive_End_B028_2_3007.mat", "12k_Drive_End_IR028_2_3003.mat"),
    },
    "B": {
        "val": ("12k_Drive_End_B028_0_3005.mat", "12k_Drive_End_IR028_0_3001.mat"),
        "test": ("12k_Drive_End_B028_3_3008.mat", "12k_Drive_End_IR028_1_3002.mat"),
    },
    "C": {
        "val": ("12k_Drive_End_B028_1_3006.mat", "12k_Drive_End_IR028_2_3003.mat"),
        "test": ("12k_Drive_End_B028_0_3005.mat", "12k_Drive_End_IR028_3_3004.mat"),
    },
}


def _size_table(split_of: dict[str, str]) -> dict[str, int]:
    return {s: sum(1 for v in split_of.values() if v == s) for s in SPLIT_NAMES}


def _assign_fixed(files: set[str], val_files: tuple[str, ...],
                  test_files: tuple[str, ...], label: str) -> dict[str, str]:
    val_set, test_set = set(val_files), set(test_files)
    if len(val_set) != len(val_files) or len(test_set) != len(test_files):
        raise RuntimeError(f"{label}: 固定名单内部存在重复文件")
    if val_set & test_set:
        raise RuntimeError(f"{label}: 同一文件同时出现在 val 和 test")
    unknown = (val_set | test_set) - files
    if unknown:
        raise RuntimeError(f"{label}: 固定名单包含未知文件：{sorted(unknown)}")
    return {f: ("val" if f in val_set else "test" if f in test_set else "train")
            for f in sorted(files)}


def build_fixed_splits(records: list[FileRecord]) -> dict[str, dict[str, dict[str, str]]]:
    """根据写死的 val/test 名单构建 A/B/C；不使用随机数或候选搜索。"""
    files_101 = {r.filename for r in records if r.in_set_101}
    files_109 = {r.filename for r in records if r.in_set_109}
    extra_109 = files_109 - files_101
    result = {}
    for name in SPLIT_SETS:
        fixed = _FIXED_101[name]
        split_101 = _assign_fixed(files_101, fixed["val"], fixed["test"], f"{name}/101")

        extras = _FIXED_109_EXTRA[name]
        split_extra = _assign_fixed(extra_109, extras["val"], extras["test"], f"{name}/109-extra")
        split_109 = {**split_101, **split_extra}
        result[name] = {"101": split_101, "109": split_109}

    report = validate_splits(result, records)
    if not report["ok"]:
        raise RuntimeError("固定划分定义无效：\n" + "\n".join(report["issues"]))
    return result


def validate_splits(splits: dict[str, dict[str, dict[str, str]]],
                    records: list[FileRecord]) -> dict:
    """校验数量、集合关系、单套平衡、训练条件覆盖和跨划分重复。"""
    by_name = {r.filename: r for r in records}
    report: dict[str, object] = {"ok": True, "issues": [], "sets": {}}

    def issue(msg: str) -> None:
        report["ok"] = False
        report["issues"].append(msg)

    def condition_key(r: FileRecord) -> tuple:
        return r.end, r.fault, int(r.diameter_mil), r.or_clock

    for name in SPLIT_SETS:
        if name not in splits:
            issue(f"缺少划分 {name}")
            continue
        if "101" not in splits[name] or "109" not in splits[name]:
            issue(f"{name}: 缺少 101 或 109 文件集合")
            continue
        s101, s109 = splits[name]["101"], splits[name]["109"]

        if len(s101) != 101 or len(s109) != 109:
            issue(f"{name}: 文件数 101:{len(s101)}/109:{len(s109)} 不为 101/109")
        unknown = (set(s101) | set(s109)) - set(by_name)
        if unknown:
            issue(f"{name}: 包含未知文件 {sorted(unknown)}")
            continue
        if set(s101) - set(s109):
            issue(f"{name}: 101 文件不全是 109 的子集")
        if any(v not in SPLIT_NAMES for v in (*s101.values(), *s109.values())):
            issue(f"{name}: 出现 train/val/test 之外的划分名称")

        for set_id, mapping, targets in (("101", s101, SPLIT_TARGETS_101),
                                         ("109", s109, SPLIT_TARGETS_109)):
            sizes = _size_table(mapping)
            got = tuple(sizes[s] for s in SPLIT_NAMES)
            if got != targets:
                issue(f"{name}/{set_id}: 数量 {got} != {targets}")

        for filename, part in s101.items():
            if s109.get(filename) != part:
                issue(f"{name}: {filename} 在 101/109 中归属不一致")

        normal_counts = {s: 0 for s in SPLIT_NAMES}
        for filename, part in s101.items():
            if by_name[filename].fault == "Normal":
                normal_counts[part] += 1
        if tuple(normal_counts[s] for s in SPLIT_NAMES) != GROUP4_TARGETS:
            issue(f"{name}: Normal 分布 {normal_counts} != 2/1/1")

        for fault in ("B", "IR"):
            counts = {s: 0 for s in SPLIT_NAMES}
            for filename, part in s109.items():
                rec = by_name[filename]
                if rec.fault == fault and rec.diameter_mil >= 28.0:
                    counts[part] += 1
            if tuple(counts[s] for s in SPLIT_NAMES) != GROUP4_TARGETS:
                issue(f"{name}: {fault}028 分布 {counts} != 2/1/1")

        set_report = {}
        for set_id, mapping in (("101", s101), ("109", s109)):
            info = {}
            for part in SPLIT_NAMES:
                members = [by_name[f] for f, value in mapping.items() if value == part]
                ends = sorted({r.end for r in members})
                faults = sorted({r.fault for r in members})
                info[part] = {"n": len(members), "ends": ends, "faults": faults}
                if not {"Normal", "IR", "OR", "B"} <= set(faults):
                    issue(f"{name}/{set_id}/{part}: 故障类别覆盖不全（{faults}）")
                if not {"Drive", "Fan"} <= set(ends):
                    issue(f"{name}/{set_id}/{part}: 未同时包含 Drive 与 Fan（{ends}）")
            set_report[set_id] = info
        report["sets"][name] = set_report

        # 对 101 的 val/test 施加明确的最低平衡要求。
        train_records = [by_name[f] for f, value in s101.items() if value == "train"]
        train_conditions = {condition_key(r) for r in train_records}
        for part in ("val", "test"):
            members = [by_name[f] for f, value in s101.items() if value == part]
            fault_counts = {fault: sum(r.fault == fault for r in members)
                            for fault in ("Normal", "IR", "OR", "B")}
            if fault_counts["Normal"] != 1 or fault_counts["IR"] < 3 \
                    or fault_counts["B"] < 3 or fault_counts["OR"] < 5:
                issue(f"{name}/101/{part}: 类别分布不足 {fault_counts}")
            end_counts = {end: sum(r.end == end for r in members) for end in ("Drive", "Fan")}
            if min(end_counts.values()) < 5:
                issue(f"{name}/101/{part}: 端别分布不足 {end_counts}")
            load_counts = {load: sum(r.load == load for r in members) for load in range(4)}
            if min(load_counts.values()) < 2:
                issue(f"{name}/101/{part}: 负载分布不足 {load_counts}")
            diameter_counts = {d: sum(r.fault != "Normal" and int(r.diameter_mil) == d
                                      for r in members) for d in (7, 14, 21)}
            if min(diameter_counts.values()) < 3:
                issue(f"{name}/101/{part}: 直径分布不足 {diameter_counts}")
            clock_counts = {clock: sum(r.fault == "OR" and r.or_clock == clock
                                       for r in members) for clock in (3, 6, 12)}
            if min(clock_counts.values()) < 1:
                issue(f"{name}/101/{part}: OR 钟点分布不足 {clock_counts}")
            missing_conditions = sorted({condition_key(r) for r in members} - train_conditions,
                                        key=str)
            if missing_conditions:
                issue(f"{name}/101/{part}: 条件未在训练集覆盖 {missing_conditions}")

    test_sets_101 = [{f for f, part in splits[n]["101"].items() if part == "test"}
                     for n in SPLIT_SETS]
    val_sets_101 = [{f for f, part in splits[n]["101"].items() if part == "val"}
                    for n in SPLIT_SETS]
    test_sets_109 = [{f for f, part in splits[n]["109"].items() if part == "test"}
                     for n in SPLIT_SETS]
    repeats = {"test": {}, "val": {}}
    for i in range(3):
        for j in range(i + 1, 3):
            key = f"{SPLIT_SETS[i]}{SPLIT_SETS[j]}"
            repeats["test"][key] = sorted(test_sets_101[i] & test_sets_101[j])
            repeats["val"][key] = sorted(val_sets_101[i] & val_sets_101[j])
            if test_sets_109[i] & test_sets_109[j]:
                issue(f"{key}: 109 测试集存在重复文件 {sorted(test_sets_109[i] & test_sets_109[j])}")
    report["repeats"] = repeats
    report["test_union"] = len(set().union(*test_sets_101))
    report["val_union"] = len(set().union(*val_sets_101))
    report["test_union_109"] = len(set().union(*test_sets_109))
    return report


def _balance_rows(name: str, splits: dict[str, dict[str, str]],
                  records: list[FileRecord]) -> list[dict]:
    by_name = {r.filename: r for r in records}
    rows = []
    for set_id in ("101", "109"):
        mapping = splits[set_id]
        for part in SPLIT_NAMES:
            members = [by_name[f] for f, value in mapping.items() if value == part]
            row = {
                "split_set": name, "file_set": set_id, "split": part, "n_files": len(members),
                "Drive": sum(r.end == "Drive" for r in members),
                "Fan": sum(r.end == "Fan" for r in members),
                "Normal": sum(r.fault == "Normal" for r in members),
                "IR": sum(r.fault == "IR" for r in members),
                "OR": sum(r.fault == "OR" for r in members),
                "B": sum(r.fault == "B" for r in members),
            }
            for diameter in (7, 14, 21, 28):
                row[f"dia_{diameter}mil"] = sum(
                    r.fault != "Normal" and int(r.diameter_mil) == diameter for r in members)
            for load in range(4):
                row[f"load_{load}"] = sum(r.load == load for r in members)
            for clock in (3, 6, 12):
                row[f"or_clock_{clock}"] = sum(
                    r.fault == "OR" and r.or_clock == clock for r in members)
            rows.append(row)
    return rows


def save_splits(splits: dict[str, dict[str, dict[str, str]]],
                records: list[FileRecord], verbose: bool = False) -> dict[str, str]:
    """保存固定 JSON、balance.csv 与说明 README。"""
    report = validate_splits(splits, records)
    if not report["ok"]:
        raise RuntimeError("划分校验失败：\n" + "\n".join(report["issues"]))

    dirs = {}
    for name in SPLIT_SETS:
        out_dir = os.path.join(SPLITS_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        dirs[name] = out_dir
        bundle = splits[name]
        for set_id in ("101", "109"):
            path = os.path.join(out_dir, f"split_{set_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bundle[set_id], f, ensure_ascii=False, indent=2, sort_keys=True)

        rows = _balance_rows(name, bundle, records)
        with open(os.path.join(out_dir, "balance.csv"), "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(_split_readme(name, bundle, report, rows))
        if verbose:
            print(f"[split] 固定划分 {name} 已保存 -> {out_dir}")
    return dirs


def _split_readme(name: str, bundle: dict[str, dict[str, str]], report: dict,
                  rows: list[dict]) -> str:
    def fmt_counts(set_id: str) -> str:
        sizes = _size_table(bundle[set_id])
        return f"train/val/test = {sizes['train']}/{sizes['val']}/{sizes['test']}"

    lines = [
        f"# 固定划分 {name}", "",
        "- 划分来源：代码中写死的固定文件名单；不使用随机种子，不在运行时搜索或重抽。",
        "- 固定约束：验证/测试均覆盖四类、Drive/Fan、7/14/21 mil、四档负载和 OR 三个钟点。",
        "- 训练覆盖：验证/测试出现的每个 (端别, 故障, 直径, OR钟点) 条件都在训练集中出现。",
        "- 跨划分：109 测试集两两无重复；验证集只保留少量重复。", "",
        "## 数量硬约束", "",
        f"- 101 文件集合：{fmt_counts('101')}",
        f"- 109 文件集合：{fmt_counts('109')}",
        "- 101 与 109 的公共文件归属完全一致；B028 与 IR028 仅出现在 109。", "",
        "## 分布表", "",
        "| 文件集合 | 集合 | 文件数 | Drive | Fan | Normal | IR | OR | B |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['file_set']} | {row['split']} | {row['n_files']} | "
                     f"{row['Drive']} | {row['Fan']} | {row['Normal']} | {row['IR']} | "
                     f"{row['OR']} | {row['B']} |")
    lines += ["", "## 跨划分重复（101 集合）", ""]
    for level in ("test", "val"):
        rep = report["repeats"][level]
        union_n = report[f"{level}_union"]
        lines.append(f"- {level}：A∩B={len(rep['AB'])}、A∩C={len(rep['AC'])}、"
                     f"B∩C={len(rep['BC'])}；三套并集覆盖 {union_n} 个文件")
    lines += [
        f"- 109 测试集三套并集覆盖 {report['test_union_109']} 个文件，三套之间无重复。",
        "", "## 说明", "",
        "- Normal 和 28 mil 文件数量较少，因此每套仍固定采用 2/1/1。",
        "- 稀有条件优先留在训练集；验证集和测试集不承担训练中从未出现过的条件。",
        "- A/B/C 用于衡量文件划分敏感性，三套模型训练统一使用相同训练种子。", "",
        f"校验结果：{'通过' if report['ok'] else '未通过'}", "",
    ]
    return "\n".join(lines)


def load_split(set_id: str, split_name: str) -> dict[str, str]:
    """读取固定划分 JSON。"""
    path = os.path.join(SPLITS_DIR, split_name, f"split_{set_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"固定划分不存在：{path}\n请先运行 `python -m cwru split-init`")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
