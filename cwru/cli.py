# -*- coding: utf-8 -*-
"""统一终端入口：python -m cwru <command>"""
from __future__ import annotations

import argparse
import os
import sys

from cwru.config import (EXPERIMENTS, MODELS, RUNS_DIR, SPLIT_SETS, WINDOW_LEN,
                         WINDOW_PLAN_ORDER)
from cwru.data.audit import audit_all
from cwru.data.dataset import get_experiment_arrays
from cwru.data.prepare import run_prepare
from cwru.data.split import build_fixed_splits, load_split, save_splits, validate_splits


def _load_arrays(exp: str, split_name: str = "A", window_plan: str = "no_overlap") -> tuple[dict, dict]:
    exp_cfg = EXPERIMENTS[exp]
    split_of = load_split(exp_cfg["files"], split_name)
    arrays = get_experiment_arrays(split_name, exp, exp_cfg, audit_all(), split_of, window_plan)
    return exp_cfg, arrays


def _default_run_dir(split_name: str, window_plan: str, experiment: str, model: str) -> str:
    """单组命令共用的默认输出目录，避免训练和评估各用一套路由。"""
    return os.path.join(RUNS_DIR, f"{split_name}_{window_plan}_{experiment}_{model}")


def cmd_audit_data(args) -> None:
    records = audit_all(verbose=True)
    print(f"\n共 {len(records)} 个 MAT 文件")
    n101 = sum(1 for r in records if r.in_set_101)
    print(f"101 文件集合: {n101} 个（排除 28 mil），109 文件集合: {len(records)} 个")
    print(f"窗口长度 {WINDOW_LEN}；窗口方案: 无重叠 1024/1024、50% 重叠 1024/512")


def cmd_split_init(args) -> None:
    records = audit_all()
    mappings = build_fixed_splits(records)
    save_splits(mappings, records, verbose=True)
    report = validate_splits(mappings, records)
    print(f"\n三套固定划分写入完成，校验：{'通过' if report['ok'] else '未通过'}")
    if not report["ok"]:
        for it in report["issues"]:
            print("  -", it)
        raise SystemExit(1)
    print(f"测试并集覆盖 {report['test_union']} 个文件，验证并集覆盖 {report['val_union']} 个文件")
    print("目录：splits/A、splits/B、splits/C")


def cmd_prepare(args) -> None:
    summary = run_prepare()
    print(f"\nprepare 完成：{summary['records']} 个文件，"
          f"{len(summary['experiments'])} 个实验清单已生成")


def cmd_train(args) -> None:
    from cwru.training.trainer import train_model
    exp_cfg, arrays = _load_arrays(args.experiment, args.split, args.window)
    out_dir = args.out_dir or _default_run_dir(args.split, args.window,
                                                args.experiment, args.model)
    train_model(args.experiment, exp_cfg, arrays, args.model, out_dir=out_dir,
                resume=args.resume, split_name=args.split, window_plan=args.window)


def cmd_full_run(args) -> None:
    from cwru.fullrun import run_full
    ctx = run_full(epochs=args.epochs, limit=args.limit)
    if ctx is None:
        print("\n已完成固定划分生成（未训练）")
        return
    print(f"\n完整批次完成：{ctx.batch_dir}")


def cmd_evaluate(args) -> None:
    from cwru.evaluation.evaluate import evaluate_run
    exp_cfg, arrays = _load_arrays(args.experiment, args.split, args.window)
    out_dir = args.out_dir or _default_run_dir(args.split, args.window,
                                                args.experiment, args.model)
    evaluate_run(out_dir, args.experiment, exp_cfg, args.model, arrays,
                 split_name=args.split, window_plan=args.window)


def cmd_compare(args) -> None:
    from cwru.evaluation.compare import run_compare
    if not args.batch:
        raise SystemExit("compare 需要 --batch 指定批次目录")
    run_compare(args.batch)


def cmd_predict(args) -> None:
    from cwru.inference.predict import predict_file
    if not os.path.exists(args.file):
        raise SystemExit(f"文件不存在: {args.file}")
    predict_file(args.experiment, args.model, args.file, args.channel,
                 split_name=args.split, window_plan=args.window, ckpt_dir=args.ckpt_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cwru", description="CWRU 轴承故障诊断（分类+回归）")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit-data", help="检查文件、采样率、变量和特殊映射")
    a.set_defaults(func=cmd_audit_data)

    s = sub.add_parser("split-init", help="生成 A/B/C 三套平衡固定划分（不训练）")
    s.set_defaults(func=cmd_split_init)

    b = sub.add_parser("prepare", help="生成元数据、划分和实验清单与归一化参数")
    b.set_defaults(func=cmd_prepare)

    c = sub.add_parser("train", help="训练指定「划分×窗口×输入×模型」组合")
    c.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    c.add_argument("--model", required=True, choices=MODELS)
    c.add_argument("--split", default="A", choices=SPLIT_SETS)
    c.add_argument("--window", default="no_overlap", choices=WINDOW_PLAN_ORDER)
    c.add_argument("--out-dir", default=None, help="输出目录（默认为 artifacts/runs/<key>）")
    c.add_argument("--resume", action="store_true", help="从 last_training.ckpt 断点续训")
    c.set_defaults(func=cmd_train)

    d = sub.add_parser("full-run", help="新建批次并执行完整 90 组正式实验")
    d.add_argument("--epochs", type=int, default=None, help="覆盖最大 epoch（默认 80）")
    d.add_argument("--limit", type=int, default=None, help="仅执行前 N 组（流程验证用）")
    d.set_defaults(func=cmd_full_run)

    e = sub.add_parser("evaluate", help="对已训练组合生成指标与图表")
    e.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    e.add_argument("--model", required=True, choices=MODELS)
    e.add_argument("--split", default="A", choices=SPLIT_SETS)
    e.add_argument("--window", default="no_overlap", choices=WINDOW_PLAN_ORDER)
    e.add_argument("--out-dir", default=None)
    e.set_defaults(func=cmd_evaluate)

    f = sub.add_parser("compare", help="汇总指定批次的结果")
    f.add_argument("--batch", required=True, help="待汇总的批次目录（例如 artifacts/full_runs/formal_run）")
    f.set_defaults(func=cmd_compare)

    g = sub.add_parser("predict", help="加载权重并对指定 MAT 文件推理")
    g.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    g.add_argument("--model", required=True, choices=MODELS)
    g.add_argument("--file", required=True, help="MAT 文件路径")
    g.add_argument("--split", default="A", choices=SPLIT_SETS)
    g.add_argument("--window", default="no_overlap", choices=WINDOW_PLAN_ORDER)
    g.add_argument("--ckpt-dir", default=None,
                   help="权重目录（默认取批次内 A/no_overlap 之外的 artifacts/runs 兼容路径）")
    g.add_argument("--channel", choices=["DE", "FE"], default=None,
                   help="DEFE 单通道实验必须显式指定测量端")
    g.set_defaults(func=cmd_predict)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
