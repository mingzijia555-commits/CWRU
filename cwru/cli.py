# -*- coding: utf-8 -*-
"""统一终端入口：python -m cwru <command>"""
from __future__ import annotations

import argparse
import os
import sys

from cwru.config import (EXPERIMENTS, EXPERIMENT_ORDER, MODELS, WINDOW_LEN, ensure_dirs)
from cwru.data.audit import audit_all
from cwru.data.dataset import get_experiment_arrays
from cwru.data.prepare import run_prepare
from cwru.data.split import load_split


def _load_arrays(exp: str) -> tuple[dict, dict]:
    from cwru.config import EXPERIMENTS
    from cwru.data.audit import audit_all
    exp_cfg = EXPERIMENTS[exp]
    split_of = load_split(exp_cfg["files"])
    arrays = get_experiment_arrays(exp, exp_cfg, audit_all(), split_of)
    return exp_cfg, arrays


def cmd_audit_data(args) -> None:
    records = audit_all(verbose=True)
    print(f"\n共 {len(records)} 个 MAT 文件")
    n101 = sum(1 for r in records if r.in_set_101)
    print(f"101 文件集合: {n101} 个（排除 28 mil），109 文件集合: {len(records)} 个")
    print(f"窗口长度 {WINDOW_LEN}，步长等于窗口长度（不重叠）")


def cmd_prepare(args) -> None:
    summary = run_prepare()
    print(f"\nprepare 完成：{summary['records']} 个文件，{len(summary['experiments'])} 个实验清单已生成")


def cmd_train(args) -> None:
    from cwru.training.trainer import train_model
    exp_cfg, arrays = _load_arrays(args.experiment)
    train_model(args.experiment, exp_cfg, arrays, args.model, resume=args.resume)


def cmd_run_all(args) -> None:
    from cwru.training.trainer import train_model
    ensure_dirs()
    total = len(EXPERIMENT_ORDER) * len(MODELS)
    done = 0
    for exp in EXPERIMENT_ORDER:
        exp_cfg, arrays = _load_arrays(exp)
        for model in MODELS:
            done += 1
            print(f"\n=== [{done}/{total}] {exp} / {model} ===")
            train_model(exp, exp_cfg, arrays, model, resume=args.resume)
    print("\nrun-all 完成：10 次训练全部就绪")


def cmd_evaluate(args) -> None:
    from cwru.evaluation.evaluate import evaluate_all, evaluate_run
    if args.all:
        evaluate_all()
    else:
        if not args.experiment or not args.model:
            raise SystemExit("evaluate 需要 --all 或同时指定 --experiment 与 --model")
        evaluate_run(args.experiment, args.model)


def cmd_compare(args) -> None:
    from cwru.evaluation.compare import run_compare
    run_compare()


def cmd_predict(args) -> None:
    from cwru.inference.predict import predict_file
    if not os.path.exists(args.file):
        raise SystemExit(f"文件不存在: {args.file}")
    predict_file(args.experiment, args.model, args.file, args.channel)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cwru", description="CWRU 轴承故障诊断（分类+回归）")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit-data", help="检查文件、采样率、变量和特殊映射")
    a.set_defaults(func=cmd_audit_data)

    b = sub.add_parser("prepare", help="生成元数据、划分和归一化参数")
    b.set_defaults(func=cmd_prepare)

    c = sub.add_parser("train", help="训练指定实验和模型")
    c.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    c.add_argument("--model", required=True, choices=MODELS)
    c.add_argument("--resume", action="store_true", help="从 last_training.ckpt 断点续训")
    c.set_defaults(func=cmd_train)

    d = sub.add_parser("run-all", help="依次完成或恢复全部 10 次训练")
    d.add_argument("--resume", action="store_true", help="已有检查点的实验从断点恢复")
    d.set_defaults(func=cmd_run_all)

    e = sub.add_parser("evaluate", help="生成完整测试指标和图表")
    e.add_argument("--all", action="store_true", help="评估全部 10 组")
    e.add_argument("--experiment", choices=list(EXPERIMENTS.keys()))
    e.add_argument("--model", choices=MODELS)
    e.set_defaults(func=cmd_evaluate)

    f = sub.add_parser("compare", help="汇总比较 10 组实验")
    f.set_defaults(func=cmd_compare)

    g = sub.add_parser("predict", help="加载权重并对指定 MAT 文件推理")
    g.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    g.add_argument("--model", required=True, choices=MODELS)
    g.add_argument("--file", required=True, help="MAT 文件路径")
    g.add_argument("--channel", choices=["DE", "FE"], default=None,
                   help="DEFE 单通道实验必须显式指定测量端")
    g.set_defaults(func=cmd_predict)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
