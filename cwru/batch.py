# -*- coding: utf-8 -*-
"""批次上下文：正式结果目录、run_config、划分快照与逐组更新的 README。

完整运行统一写入 artifacts/full_runs/formal_run/，小规模验证写入
artifacts/tmp_check/full_run/。开始运行前会清空对应目录，避免混入旧结果。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess

from cwru.config import (BATCH_SIZE, ES_PATIENCE, EXPERIMENTS, EXPERIMENT_ORDER,
                         FULL_RUNS_DIR, LR, LR_FACTOR, LR_PATIENCE, MAX_EPOCHS,
                         MODELS, N_FORMAL_RUNS, PROJECT_ROOT, SPLIT_SETS,
                         SPLITS_DIR, TMP_CHECK_DIR, TRAIN_SEED, WEIGHT_DECAY, WINDOW_PLANS,
                         WINDOW_PLAN_ORDER)


def git_version() -> str:
    """返回当前代码版本（短哈希 + 是否有未提交改动）。"""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{rev}{'+dirty' if dirty else ''}"
    except Exception:
        return "unknown"


class BatchContext:
    def __init__(self, batch_dir: str, split_mappings: dict, records, plan: str = "full"):
        self.batch_dir = batch_dir
        self.batch_id = os.path.basename(batch_dir)
        self.split_mappings = split_mappings      # {"A": {"101":..., "109":...}, ...}
        self.records = records
        self.plan = plan
        self.started_at = _dt.datetime.now().isoformat(timespec="seconds")
        self.ended_at: str | None = None
        self.results: list[dict] = []
        self.conclusions: dict | None = None
        self.message: str = ""

    # ---------- 创建 ----------
    @classmethod
    def create(cls, split_mappings: dict, records, plan: str = "full") -> "BatchContext":
        batch_dir = (os.path.join(TMP_CHECK_DIR, "full_run") if plan == "check"
                     else os.path.join(FULL_RUNS_DIR, "formal_run"))
        if os.path.isdir(batch_dir):
            shutil.rmtree(batch_dir)
        os.makedirs(batch_dir)
        ctx = cls(batch_dir, split_mappings, records, plan=plan)
        ctx.snapshot_splits()
        ctx.write_run_config()
        ctx.init_readme()
        return ctx

    # ---------- 路径 ----------
    def run_dir(self, split_name: str, window_plan: str, exp: str, model: str) -> str:
        return os.path.join(self.batch_dir, split_name, window_plan, exp, model)

    @property
    def comparisons_dir(self) -> str:
        return os.path.join(self.batch_dir, "comparisons")

    @property
    def readme_path(self) -> str:
        return os.path.join(self.batch_dir, "README.md")

    # ---------- 划分快照 ----------
    def snapshot_splits(self) -> None:
        dst_root = os.path.join(self.batch_dir, "splits")
        os.makedirs(dst_root, exist_ok=True)
        for name in SPLIT_SETS:
            src = os.path.join(SPLITS_DIR, name)
            dst = os.path.join(dst_root, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

    # ---------- 配置 ----------
    def run_config(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "created_at": self.started_at,
            "code_version": git_version(),
            "plan": self.plan,
            "split_definition": "fixed_file_lists_v1",
            "train_seed": TRAIN_SEED,
            "split_sets": SPLIT_SETS,
            "window_plans": {k: WINDOW_PLANS[k] for k in WINDOW_PLAN_ORDER},
            "experiments": {k: EXPERIMENTS[k] for k in EXPERIMENT_ORDER},
            "models": MODELS,
            "n_expected_runs": N_FORMAL_RUNS,
            "training": {
                "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "lr": LR,
                "weight_decay": WEIGHT_DECAY, "lr_patience": LR_PATIENCE,
                "lr_factor": LR_FACTOR, "es_patience": ES_PATIENCE,
            },
            "n_data_files": len(self.records),
        }

    def write_run_config(self) -> str:
        path = os.path.join(self.batch_dir, "run_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.run_config(), f, ensure_ascii=False, indent=2)
        return path

    # ---------- 逐组更新 ----------
    def record_result(self, res: dict) -> None:
        self.results.append(res)
        self.write_readme()

    def set_message(self, text: str) -> None:
        self.message = text
        self.write_readme()

    def finish(self, conclusions: dict | None = None) -> None:
        self.ended_at = _dt.datetime.now().isoformat(timespec="seconds")
        self.conclusions = conclusions
        self.write_readme()

    # ---------- README ----------
    def init_readme(self) -> str:
        return self.write_readme()

    def write_readme(self) -> str:
        cfg = self.run_config()
        lines = [
            f"# CWRU 90 组扩展实验批次 `{self.batch_id}`",
            "",
            "## 批次概况",
            "",
            f"- 批次编号：`{self.batch_id}`",
            f"- 开始时间：{self.started_at}",
            f"- 结束时间：{self.ended_at or '（进行中）'}",
            f"- 项目代码版本：`{cfg['code_version']}`",
            f"- 原始数据文件数：{cfg['n_data_files']} 个 MAT（101 公共集合 + B028/IR028）",
            "- 文件划分：代码内固定文件名单 `fixed_file_lists_v1`（无随机生成）",
            f"- 训练种子 TRAIN_SEED = {TRAIN_SEED}（90 组统一）",
            f"- 预计实验数：{N_FORMAL_RUNS}，已完成：{len(self.results)}",
            "",
            "## 实验矩阵",
            "",
            "- 固定划分：A、B、C（每套 101 为 71/15/15，109 为 75/17/17）",
            "- 窗口方案：无重叠 1024/1024、50% 重叠 1024/512",
            "- 输入方案：" + "、".join(EXPERIMENT_ORDER),
            "- 模型：" + "、".join(MODELS),
            f"- 训练参数：batch={BATCH_SIZE}，max_epochs={MAX_EPOCHS}，lr={LR}，"
            f"weight_decay={WEIGHT_DECAY}，early_stopping={ES_PATIENCE}",
            "",
            "## 三套划分摘要（101 集合）",
            "",
            "| 划分 | train | val | test | 测试并集覆盖 | 验证并集覆盖 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        test_sets = [{f for f, s in m.get("101", {}).items() if s == "test"}
                     for m in (self.split_mappings.get(n, {}) for n in SPLIT_SETS)]
        val_sets = [{f for f, s in m.get("101", {}).items() if s == "val"}
                    for m in (self.split_mappings.get(n, {}) for n in SPLIT_SETS)]
        test_union = len(set().union(*test_sets)) if all(test_sets) else 0
        val_union = len(set().union(*val_sets)) if all(val_sets) else 0
        for name in SPLIT_SETS:
            m = self.split_mappings.get(name, {}).get("101", {})
            tr = sum(1 for v in m.values() if v == "train")
            va = sum(1 for v in m.values() if v == "val")
            te = sum(1 for v in m.values() if v == "test")
            lines.append(f"| {name} | {tr} | {va} | {te} | {test_union} | {val_union} |")
        lines.append(f"\n三套划分两两重复："
                     + "；".join(
                         f"{SPLIT_SETS[i]}∩{SPLIT_SETS[j]} 测试 {len(test_sets[i] & test_sets[j])}、"
                         f"验证 {len(val_sets[i] & val_sets[j])}"
                         for i in range(3) for j in range(i + 1, 3)))

        lines += ["", "## 90 组核心指标", "",
                  "| 划分 | 窗口 | 输入 | 模型 | 窗口Acc | 窗口F1 | 窗口MAE(mil) | "
                  "文件Acc | 文件F1 | 文件MAE(mil) | best_epoch |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]

        def fmt(v, nd=4):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"

        for r in self.results:
            lines.append(
                f"| {r['split']} | {r['window_plan']} | {r['experiment']} | {r['model']} | "
                f"{fmt(r.get('window_accuracy'))} | {fmt(r.get('window_macro_f1'))} | "
                f"{fmt(r.get('window_mae_mil'), 2)} | {fmt(r.get('file_accuracy'))} | "
                f"{fmt(r.get('file_macro_f1'))} | {fmt(r.get('file_mae_mil'), 2)} | "
                f"{r.get('best_epoch') if r.get('best_epoch') is not None else '—'} |")

        if self.conclusions:
            lines += self._conclusion_lines(self.conclusions)

        if self.message:
            lines += ["", "## 备注", "", self.message]

        lines += [
            "",
            "## 结果文件位置",
            "",
            "- 单组实验：`<划分>/<窗口方案>/<输入方案>/<模型>/`",
            "  - `best_inference.pt`：验证总损失最佳时的推理权重",
            "  - `history.json`：逐 epoch 训练/验证曲线数据",
            "  - `metrics.json`：本组汇总指标（不含逐文件明细）",
            "  - `files.csv`：测试集逐文件真实值、预测值与属性",
            "  - `figures/`：损失曲线、验证曲线、混淆矩阵、分类与回归细分图",
            "- 划分快照：`splits/A|B|C/`（split_101.json、split_109.json、balance.csv、README.md）",
            "- 汇总对比：`comparisons/metrics_all.csv`、`comparisons/mean_std_summary.csv`、"
            "`comparisons/*.png`、`comparisons/conclusions.json`",
            "- 批次配置：`run_config.json`",
            "",
        ]
        with open(self.readme_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return self.readme_path

    def _conclusion_lines(self, c: dict) -> list[str]:
        def fmt(v, nd=4):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"

        lines = [
            "",
            "## A/B/C 平均值与标准差",
            "",
            "| 窗口 | 输入 | 模型 | 窗口Acc 均值±std | 窗口F1 均值±std | 窗口MAE 均值±std | "
            "文件Acc 均值±std | 文件MAE 均值±std |",
            "|---|---|---|---|---|---|---|",
        ]
        summary = c.get("summary", {})
        for key in sorted(summary):
            e = summary[key]
            lines.append(
                f"| {e['window_plan']} | {e['experiment']} | {e['model']} | "
                f"{fmt(e.get('window_accuracy_mean'))}±{fmt(e.get('window_accuracy_std'), 3)} | "
                f"{fmt(e.get('window_macro_f1_mean'))}±{fmt(e.get('window_macro_f1_std'), 3)} | "
                f"{fmt(e.get('window_mae_mil_mean'), 2)}±{fmt(e.get('window_mae_mil_std'), 2)} | "
                f"{fmt(e.get('file_accuracy_mean'))}±{fmt(e.get('file_accuracy_std'), 3)} | "
                f"{fmt(e.get('file_mae_mil_mean'), 2)}±{fmt(e.get('file_mae_mil_std'), 2)} |")

        conc = c.get("conclusions", {})
        lines += [
            "",
            "## 关键结论",
            "",
            f"- 最佳分类配置（窗口 Macro-F1 均值最高）：`{conc.get('best_classification')}` "
            f"（{fmt(conc.get('best_classification_macro_f1'))}）",
            f"- 最佳回归配置（窗口直径 MAE 均值最低）：`{conc.get('best_regression')}` "
            f"（{fmt(conc.get('best_regression_mae_mil'), 2)} mil）",
        ]
        mm = conc.get("model_mean_macro_f1", {})
        mma = conc.get("model_mean_mae_mil", {})
        if mm:
            lines.append("- 模型对比（窗口 Macro-F1 / 直径 MAE 平均）："
                         + "；".join(f"{m} {fmt(mm.get(m))} / {fmt(mma.get(m), 2)}mil"
                                     for m in MODELS if mm.get(m) is not None))
        ov = conc.get("overlap_mean_macro_f1", {})
        oma = conc.get("overlap_mean_mae_mil", {})
        if ov:
            lines.append("- 重叠对比（窗口 Macro-F1 / 直径 MAE 平均）："
                         + "；".join(f"{p} {fmt(ov.get(p))} / {fmt(oma.get(p), 2)}mil"
                                     for p in WINDOW_PLAN_ORDER if ov.get(p) is not None))
        sm = conc.get("split_mean", {})
        if sm:
            lines.append("- 划分对比（窗口 Macro-F1 平均）："
                         + "；".join(f"{s} {fmt(sm[s].get('window_macro_f1'))}"
                                     for s in SPLIT_SETS if s in sm))
        em = conc.get("experiment_mean", {})
        if em:
            lines.append("- 输入方案对比（窗口 Macro-F1 / 直径 MAE 平均）："
                         + "；".join(f"{e} {fmt(em[e].get('window_macro_f1'))} / "
                                     f"{fmt(em[e].get('window_mae_mil'), 2)}mil"
                                     for e in EXPERIMENT_ORDER if e in em and em[e].get('window_macro_f1') is not None))
        return lines
