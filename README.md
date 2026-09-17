# CWRU 轴承故障诊断

这是一个用于课程汇报的 PyTorch 小项目：根据 CWRU 轴承振动信号，完成四分类故障识别，并预测故障直径。

## 怎么运行

项目使用 Conda 环境 `AI2026Summer`。数据放在 `CaseWesternReserveUniversityData/`，固定的数据划分放在 `splits/`。

### 重新做完整实验

```bash
python run_experiments.py
```

程序按“3 套划分 × 2 种窗口 × 5 种输入 × 3 种模型”运行 90 组实验，并把结果写入 `artifacts/final_results/`。已有正式结果时程序会停止，避免误覆盖。

### 对一个 MAT 文件进行预测

打开 `predict_one.py`，修改文件路径、实验方案和模型名称，然后运行：

```bash
python predict_one.py
```

这是汇报时最容易演示的入口，不需要记命令行参数。

## 主要设置

- 输入方案：101/109 文件集合、DE/FE 单通道或双通道。
- 窗口方案：长度 1024，支持无重叠和 50% 重叠。
- 模型：`Cnn1d`、`CnnGru`、`CnnLstm`。
- 输出：四分类结果和故障直径回归结果。
- 训练：Adam，最多 80 个 epoch；验证集效果不再改善时提前停止，学习率保持为 0.001。

说明：`artifacts/final_results/` 中已经保存的 90 组结果是之前训练得到的历史结果，具体训练设置以其中的 `run_config.json` 为准；上面的新设置用于之后重新训练时的结果。

## 结果目录怎么读

`artifacts/final_results/` 是当前 90 组正式结果。每组目录里：

- `best_inference.pt`：该组验证集表现最好的模型；
- `metrics.json`：测试集指标；
- `history.json`：训练过程；
- `files.csv`：测试文件及预测值；
- `figures/`：训练曲线和混淆矩阵；
- `comparisons/`：90 组结果的汇总表和对比图。

旧的 10 组阶段性结果仍保存在 `artifacts/full_runs/legacy_20260915_1353_original/`，只作为过程记录，没有删除，也不计入 90 组正式汇总。

## 汇报时可以这样说明

先按文件划分训练和测试，再切分振动信号窗口；模型同时学习故障类别和故障直径。A/B/C 三套固定划分用于检查结果是否依赖某一批文件，最后用汇总表比较不同输入、窗口和模型的表现。
