# -*- coding: utf-8 -*-
"""评估层测试：文件级连续直径聚合修正（问题一）与指标职责分离。

关键回归点：文件级直径必须由模型窗口预测取均值，而不是真实标签均值，
否则文件级 MAE/RMSE 会恒为 0。
"""
import numpy as np

from cwru.evaluation.metrics import file_level_predictions, regression_errors


def _toy_inputs():
    # 两个测试文件：file 0 为故障（真实 14 mil，3 窗口），file 1 为 Normal（2 窗口）
    files = [
        {"index": 0, "filename": "12k_Drive_End_B014_0_1.mat", "split": "test",
         "end": "Drive", "fault": "B", "diameter_mil": 14.0, "load": 0,
         "or_clock": None, "channels": ["DE"], "n_windows": 3},
        {"index": 1, "filename": "normal_0_97.mat", "split": "test",
         "end": "Normal", "fault": "Normal", "diameter_mil": 0.0, "load": 0,
         "or_clock": None, "channels": ["DE"], "n_windows": 2},
    ]
    y_cls = np.array([3, 3, 3, 0, 0])
    probs = np.array([
        [0.05, 0.05, 0.10, 0.80],
        [0.05, 0.05, 0.10, 0.80],
        [0.05, 0.05, 0.10, 0.80],
        [0.90, 0.03, 0.04, 0.03],
        [0.90, 0.03, 0.04, 0.03],
    ])
    # 真实标签：故障窗口 y_reg=14/7=2.0，Normal 被掩码为 0
    y_reg = np.array([2.0, 2.0, 2.0, 0.0, 0.0])
    # 模型直径预测（缩放单位）：文件 0 三个窗口预测均值 = (2.4+2.0+1.6)/3 = 2.0 → 14.0 mil
    y_reg_pred = np.array([2.4, 2.0, 1.6, 0.1, 0.1])
    reg_mask = np.array([True, True, True, False, False])
    file_idx = np.array([0, 0, 0, 1, 1])
    return y_cls, probs, y_reg, y_reg_pred, reg_mask, file_idx, files


def test_file_diameter_uses_model_predictions_not_labels():
    y_cls, probs, y_reg, y_reg_pred, reg_mask, file_idx, files = _toy_inputs()
    rows = file_level_predictions(y_cls, probs, y_reg, y_reg_pred, reg_mask,
                                  file_idx, files)["files"]
    fault_row = next(r for r in rows if r["fault"] != "Normal")
    # 手工均值：模型预测均值，而非真实标签均值
    manual = float(y_reg_pred[:3].mean()) * 7.0
    assert abs(fault_row["pred_diameter_mil"] - manual) < 1e-9
    assert abs(fault_row["pred_diameter_mil"] - 14.0) < 1e-9


def test_file_regression_error_not_zero():
    """分组后文件级 MAE/RMSE 不应恒为 0。"""
    y_cls, probs, y_reg, y_reg_pred, reg_mask, file_idx, files = _toy_inputs()
    # 让模型预测偏离真实值
    y_reg_pred = np.array([3.0, 3.0, 3.0, 0.1, 0.1])
    rows = file_level_predictions(y_cls, probs, y_reg, y_reg_pred, reg_mask,
                                  file_idx, files)["files"]
    fault_rows = [r for r in rows if r["fault"] != "Normal"]
    err = np.array([r["pred_diameter_mil"] / 7.0 - r["diameter_mil"] / 7.0
                    for r in fault_rows])
    mae = float(np.abs(err).mean() * 7.0)
    rmse = float(np.sqrt((err ** 2).mean()) * 7.0)
    assert mae > 0.0 and rmse > 0.0


def test_regression_errors_report_mil_units():
    result = regression_errors(np.array([2.0]), np.array([3.0]))
    assert result["mae_scaled"] == 1.0
    assert result["rmse_scaled"] == 1.0
    assert result["mae_mil"] == 7.0
    assert result["rmse_mil"] == 7.0


def test_normal_file_diameter_zero_and_excluded():
    y_cls, probs, y_reg, y_reg_pred, reg_mask, file_idx, files = _toy_inputs()
    rows = file_level_predictions(y_cls, probs, y_reg, y_reg_pred, reg_mask,
                                  file_idx, files)["files"]
    normal_row = next(r for r in rows if r["fault"] == "Normal")
    assert normal_row["pred_diameter_mil"] == 0.0


def test_evaluate_outputs_separated(tmp_path):
    """逐文件结果写入 files.csv，并使用可读的类别字段。"""
    import csv
    from cwru.evaluation.evaluate import write_files_csv
    path = str(tmp_path / "files.csv")
    write_files_csv(path, [{
        "filename": "a.mat", "split": "test", "end": "Fan", "fault": "IR",
        "true_cls": 1, "pred_cls": 1, "pred_prob": [0.05, 0.8, 0.1, 0.05],
        "diameter_mil": 7.0, "pred_diameter_mil": 7.4, "load": 2,
        "or_clock": None, "n_windows": 12,
    }])
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["pred_class"] == "IR"
    assert rows[0]["correct"] == "1"
