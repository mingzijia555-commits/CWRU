# CWRU 90 组扩展实验批次 `full_run`

## 批次概况

- 批次编号：`full_run`
- 开始时间：2026-09-15T20:57:01
- 结束时间：2026-09-15T20:57:08
- 项目代码版本：`89731ed+dirty`
- 原始数据文件数：109 个 MAT（101 公共集合 + B028/IR028）
- 文件划分：代码内固定文件名单 `fixed_file_lists_v1`（无随机生成）
- 训练种子 TRAIN_SEED = 42（90 组统一）
- 预计实验数：90，已完成：1

## 实验矩阵

- 固定划分：A、B、C（每套 101 为 71/15/15，109 为 75/17/17）
- 窗口方案：无重叠 1024/1024、50% 重叠 1024/512
- 输入方案：dual101、109DEFE、101DEFE、109DE、101DE
- 模型：Cnn1d、CnnGru、CnnLstm
- 训练参数：batch=128，max_epochs=80，lr=0.001，weight_decay=0.0001，early_stopping=12

## 三套划分摘要（101 集合）

| 划分 | train | val | test | 测试并集覆盖 | 验证并集覆盖 |
|---|---:|---:|---:|---:|---:|
| A | 71 | 15 | 15 | 45 | 44 |
| B | 71 | 15 | 15 | 45 | 44 |
| C | 71 | 15 | 15 | 45 | 44 |

三套划分两两重复：A∩B 测试 0、验证 0；A∩C 测试 0、验证 0；B∩C 测试 0、验证 1

## 90 组核心指标

| 划分 | 窗口 | 输入 | 模型 | 窗口Acc | 窗口F1 | 窗口MAE(mil) | 文件Acc | 文件F1 | 文件MAE(mil) | best_epoch |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | no_overlap | dual101 | Cnn1d | 0.8351 | 0.8345 | 2.57 | 0.8000 | 0.8059 | 2.42 | 1 |

## 备注

验证运行：仅执行前 1 组（--limit 1），非正式完整批次，不用于最终结论。

## 结果文件位置

- 单组实验：`<划分>/<窗口方案>/<输入方案>/<模型>/`
  - `best_inference.pt`：验证总损失最佳时的推理权重
  - `history.json`：逐 epoch 训练/验证曲线数据
  - `metrics.json`：本组汇总指标（不含逐文件明细）
  - `files.csv`：测试集逐文件真实值、预测值与属性
  - `figures/`：损失曲线、验证曲线、混淆矩阵、分类与回归细分图
- 划分快照：`splits/A|B|C/`（split_101.json、split_109.json、balance.csv、README.md）
- 汇总对比：`comparisons/metrics_all.csv`、`comparisons/mean_std_summary.csv`、`comparisons/*.png`、`comparisons/conclusions.json`
- 批次配置：`run_config.json`
