# -*- coding: utf-8 -*-
"""完整批次编排：初始化固定划分 -> 90 组训练 -> 评估 -> 汇总 -> 批次 README。

用法：
    python -m cwru split-init        # 仅生成 A/B/C 固定划分（不训练）
    python -m cwru full-run          # 写入 formal_run 并执行 90 组正式实验
    python -m cwru full-run --epochs 1 --limit 3   # 小型验证，用于检查流程
"""
from __future__ import annotations

import time

from cwru.batch import BatchContext
from cwru.config import (EXPERIMENT_ORDER, EXPERIMENTS, MAX_EPOCHS, MODELS, SPLIT_SETS,
                         TRAIN_SEED, WINDOW_PLAN_ORDER, ensure_dirs)
from cwru.data.audit import audit_all
from cwru.data.dataset import get_experiment_arrays
from cwru.data.split import build_fixed_splits, save_splits, validate_splits
from cwru.evaluation import compare as compare_mod
from cwru.evaluation.evaluate import evaluate_run
from cwru.training.trainer import train_model


def init_splits(records, verbose: bool = True) -> dict:
    """写入并返回代码中定义的 A/B/C 三套固定划分。"""
    mappings = build_fixed_splits(records)
    save_splits(mappings, records, verbose=verbose)

    report = validate_splits(mappings, records)
    if not report["ok"]:
        raise RuntimeError("固定划分校验失败：\n" + "\n".join(report["issues"]))
    if verbose:
        print(f"[split] 三套固定划分校验通过；测试并集覆盖 {report['test_union']} 个文件，"
              f"验证并集覆盖 {report['val_union']} 个文件")
    return mappings


def run_full(epochs: int | None = None, verbose: bool = True,
             splits_only: bool = False, limit: int | None = None) -> BatchContext | None:
    """执行完整批次：三套划分 × 两种窗口 × 五种输入 × 三种模型。"""
    ensure_dirs()
    records = audit_all()
    mappings = init_splits(records, verbose=verbose)
    if splits_only:
        return None

    ctx = BatchContext.create(mappings, records)
    print(f"[batch] 正式结果目录：{ctx.batch_dir}")

    total = len(SPLIT_SETS) * len(WINDOW_PLAN_ORDER) * len(EXPERIMENT_ORDER) * len(MODELS)
    done, failed = 0, 0
    hit_limit = False
    t0 = time.time()

    for split_name in SPLIT_SETS:
        split_101 = mappings[split_name]["101"]
        split_109 = mappings[split_name]["109"]
        for plan in WINDOW_PLAN_ORDER:
            for exp in EXPERIMENT_ORDER:
                exp_cfg = EXPERIMENTS[exp]
                split_of = split_101 if exp_cfg["files"] == "101" else split_109
                arrays = get_experiment_arrays(split_name, exp, exp_cfg, records,
                                               split_of, plan)
                for model in MODELS:
                    if limit is not None and done >= limit:
                        hit_limit = True
                        break
                    done += 1
                    tag = f"{split_name}/{plan}/{exp}/{model}"
                    print(f"\n=== [{done}/{total}] {tag} ===")
                    out_dir = ctx.run_dir(split_name, plan, exp, model)
                    try:
                        train_model(exp, exp_cfg, arrays, model, out_dir=out_dir,
                                    epochs=epochs if epochs is not None else MAX_EPOCHS,
                                    seed=TRAIN_SEED, verbose=verbose,
                                    split_name=split_name, window_plan=plan)
                        res = evaluate_run(out_dir, exp, exp_cfg, model, arrays,
                                           split_name=split_name, window_plan=plan,
                                           verbose=verbose)
                        ctx.record_result(res)
                    except Exception as exc:  # 单组失败不中断整批
                        failed += 1
                        print(f"[batch] {tag} 失败：{exc}")
                        ctx.record_failure(split_name, plan, exp, model, repr(exc))
                if hit_limit:
                    break
            if hit_limit:
                break
        if hit_limit:
            break

    if hit_limit:
        note = (f"验证运行：仅执行前 {limit} 组（--limit {limit}），"
                f"非正式完整批次，不用于最终结论。")
        print(f"[batch] 达到 limit={limit}，提前结束")
        ctx.set_message(note)

    print(f"\n[batch] 训练与评估完成：成功 {len(ctx.results)} 组，失败 {failed} 组，"
          f"用时 {time.time() - t0:.0f}s")

    if ctx.results:
        summary = compare_mod.run_compare(ctx.batch_dir, verbose=verbose)
        ctx.finish(conclusions=summary)
    else:
        ctx.finish()
    print(f"[batch] 批次 README：{ctx.readme_path}")
    return ctx
