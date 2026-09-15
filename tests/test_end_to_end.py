# -*- coding: utf-8 -*-
"""小规模端到端：真实数据上 1 个 epoch 训练 → 评估 → 绘图 → 终端推理。

所有临时产物写入 tmp_path，不触碰正式 artifacts 批次目录。
"""
import os

import pytest


@pytest.fixture(scope="module")
def arrays_101de():
    from cwru.cli import _load_arrays
    return _load_arrays("101DE", "A", "no_overlap")


def test_end_to_end_train_evaluate_predict(arrays_101de, tmp_path):
    from cwru.training.trainer import train_model
    from cwru.evaluation.evaluate import evaluate_run
    from cwru.config import DATA_DIR

    exp_cfg, arrays = arrays_101de
    out_dir = str(tmp_path / "A_no_overlap_101DE_Cnn1d")
    train_model("101DE", exp_cfg, arrays, "Cnn1d", out_dir=out_dir, epochs=1, seed=42,
                split_name="A", window_plan="no_overlap")
    result = evaluate_run(out_dir, "101DE", exp_cfg, "Cnn1d", arrays,
                          split_name="A", window_plan="no_overlap")
    assert result["n_test_windows"] > 1000
    assert result["n_test_files"] == 15
    # 101 集合不含 28 mil
    import json
    with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    assert m["mil28"]["n_windows"] == 0
    assert "rows" not in m["file_level"]          # 逐文件明细只在 files.csv
    assert os.path.exists(os.path.join(out_dir, "files.csv"))

    # 文件级回归指标不再恒为 0（问题一修正）
    assert m["file_level"]["regression"]["mae_mil"] > 0

    # 图表生成检查
    fig_dir = os.path.join(out_dir, "figures")
    for name in ("loss_curves.png", "confusion_window.png", "confusion_file.png",
                 "perclass.png", "or_clock.png"):
        assert os.path.exists(os.path.join(fig_dir, name)), name


def test_109_mil28_breakdown(tmp_path):
    from cwru.cli import _load_arrays
    from cwru.training.trainer import train_model
    from cwru.evaluation.evaluate import evaluate_run
    exp_cfg, arrays = _load_arrays("109DEFE", "A", "no_overlap")
    out_dir = str(tmp_path / "A_no_overlap_109DEFE_Cnn1d")
    train_model("109DEFE", exp_cfg, arrays, "Cnn1d", out_dir=out_dir, epochs=1, seed=42,
                split_name="A", window_plan="no_overlap")
    result = evaluate_run(out_dir, "109DEFE", exp_cfg, "Cnn1d", arrays,
                          split_name="A", window_plan="no_overlap")
    import json
    with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    assert m["mil28"]["n_windows"] > 0


def test_overlap_plan_end_to_end(tmp_path):
    """50% 重叠方案可正常训练与前向，窗口数较无重叠接近翻倍。"""
    from cwru.cli import _load_arrays
    from cwru.training.trainer import train_model
    exp_cfg, arrays = _load_arrays("101DE", "A", "overlap50")
    assert int(arrays["stride"]) == 512
    out_dir = str(tmp_path / "A_overlap50_101DE_CnnLstm")
    train_model("101DE", exp_cfg, arrays, "CnnLstm", out_dir=out_dir, epochs=1, seed=42,
                split_name="A", window_plan="overlap50")
    assert os.path.exists(os.path.join(out_dir, "best_inference.pt"))


def test_defe_requires_channel():
    from cwru.inference.predict import predict_file
    from cwru.config import DATA_DIR
    with pytest.raises(ValueError):
        predict_file("109DEFE", "Cnn1d", os.path.join(DATA_DIR, "normal_0_97.mat"), None)
