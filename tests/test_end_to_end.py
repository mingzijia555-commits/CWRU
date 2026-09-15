# -*- coding: utf-8 -*-
"""小规模端到端：真实数据上 1 个 epoch 训练 → 评估 → 绘图 → 终端推理。"""
import os

import pytest


@pytest.fixture(scope="module")
def arrays_101de():
    from cwru.cli import _load_arrays
    return _load_arrays("101DE")


def test_end_to_end_train_evaluate_predict(arrays_101de):
    from cwru.training.trainer import train_model
    from cwru.evaluation.evaluate import evaluate_run
    from cwru.config import DATA_DIR

    exp_cfg, arrays = arrays_101de
    train_model("101DE", exp_cfg, arrays, "Cnn1d", epochs=1, seed=42)
    result = evaluate_run("101DE", "Cnn1d")
    assert result["window_level"]["n_windows"] > 1000
    assert "confusion_matrix" in result["window_level"]
    # 101 集合不含 28 mil
    assert result["mil28"]["n_windows"] == 0

    # 109 实验包含 28 mil，端到端验证其细分指标
    from cwru.cli import _load_arrays
    exp_cfg109, arrays109 = _load_arrays("109DEFE")
    train_model("109DEFE", exp_cfg109, arrays109, "Cnn1d", epochs=1, seed=42)
    result109 = evaluate_run("109DEFE", "Cnn1d")
    assert result109["mil28"]["n_windows"] > 0

    # 图表生成检查
    from cwru.config import FIGURES_DIR
    for name in ("loss_curves_101DE_Cnn1d.png", "confusion_window_101DE_Cnn1d.png",
                 "perclass_101DE_Cnn1d.png", "or_clock_101DE_Cnn1d.png",
                 "diameter_109DEFE_Cnn1d.png"):
        assert os.path.exists(os.path.join(FIGURES_DIR, name)), name

    # 终端推理：任意测试 MAT 文件（101DE 为单通道 DE，无需 --channel）
    from cwru.inference.predict import predict_file
    res = predict_file("101DE", "Cnn1d", os.path.join(DATA_DIR, "normal_0_97.mat"))
    assert res["pred_class"] in ("Normal", "IR", "OR", "B")
    assert len(res["probs"]) == 4

    # DEFE 实验显式指定通道可正常推理
    res2 = predict_file("109DEFE", "Cnn1d", os.path.join(DATA_DIR, "12k_Fan_End_B007_0_282.mat"),
                        channel="FE")
    assert res2["channels"] == ["FE"]


def test_defe_requires_channel():
    from cwru.inference.predict import predict_file
    from cwru.config import DATA_DIR
    with pytest.raises(ValueError):
        predict_file("109DEFE", "Cnn1d", os.path.join(DATA_DIR, "normal_0_97.mat"), None)
