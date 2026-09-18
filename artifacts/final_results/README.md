# CWRU 正式实验结果

这里保存 90 组正式实验结果：3 套数据划分 × 2 种窗口方式 × 5 种输入实验 × 3 种模型。

- 每个实验文件夹中的 `best_inference.pt`：该组验证集效果最好的模型，可用于预测。
- `metrics.json`：测试集上的准确率、F1 等指标。
- `history.json`：训练过程中损失和准确率的变化。
- `files.csv`：该组使用的数据文件清单。
- `figures/`：训练曲线和混淆矩阵。
- `comparisons/`：把 90 组结果汇总后的表格和图。
- `experiment_config.json`：这批结果对应的实验组合和训练设置。

旧的 10 组阶段性结果仍保存在 `artifacts/full_runs/legacy_20260915_1353_original`，没有删除。
