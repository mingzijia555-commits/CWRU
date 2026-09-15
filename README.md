# CWRU 轴承故障诊断（分类 + 故障直径回归）

基于 Case Western Reserve University (CWRU) 轴承数据集的 PyTorch 项目：
四分类（Normal / IR / OR / B）+ 连续故障直径回归，共 5 种实验配置 × 2 种模型 = 10 次正式训练。

## 环境

- Python 3.11（conda 环境 `AI2026Summer`）
- torch 2.11.0+cu128（CUDA 可用即可，CPU 也能运行但更慢）
- scipy / numpy / matplotlib / pandas / pytest

```bash
pip install -r requirements.txt
```

数据：`CaseWesternReserveUniversityData/` 下 109 个 MAT 文件（勿改动，首次提交后视为只读）。

## 快速开始

```bash
# 1. 数据审计（文件、变量、特殊映射）
python -m cwru audit-data

# 2. 生成元数据、固定划分、归一化参数与实验清单
python -m cwru prepare

# 3. 训练（可单次或全部）
python -m cwru train --experiment 101DE --model Cnn1d
python -m cwru run-all --resume          # 依次完成/恢复全部 10 次训练

# 4. 评估与图表
python -m cwru evaluate --all
python -m cwru compare

# 5. 终端推理
python -m cwru predict --experiment 101DE --model Cnn1d --file CaseWesternReserveUniversityData/normal_0_97.mat
# DEFE 单通道实验必须显式指定测量端：
python -m cwru predict --experiment 109DEFE --model CnnGru --file CaseWesternReserveUniversityData/12k_Fan_End_B007_0_282.mat --channel FE
```

自动测试：

```bash
pytest
```

## 实验配置

| 实验 | 文件集合 | 通道方案 |
|---|---|---|
| `dual101` | 101 文件 | [DE, FE] 双通道 |
| `109DEFE` | 109 文件 | 驱动端故障取 DE，风扇端故障取 FE，正常取 DE |
| `101DEFE` | 101 文件 | 同上（排除 28 mil） |
| `109DE` | 109 文件 | 全部取 DE |
| `101DE` | 101 文件 | 全部取 DE |

## 数据处理规则

- Normal 原始 48 kHz，`scipy.signal.resample_poly(sig, 1, 4)` 降采样到 12 kHz；故障数据保持 12 kHz。
- 窗口长度 1024，步长 1024（不重叠），尾部不足丢弃。
- 先划分文件（固定清单，种子 42），再切窗口；101 集合 71/15/15，Normal/B028/IR028 各 2/1/1，109 集合 75/17/17。
- 归一化参数仅从各实验训练集计算；类别权重 = 1/√(训练窗口数)，归一化到均值 1。
- 特殊变量映射：`normal_2_99.mat` → X099；`12k_Fan_End_IR014_1_276.mat` → X275；28 mil 文件从内容识别唯一 `*_DE_time`。
- 窗口数量：101 方案 11,891；109 方案 12,833。

## 模型

- **Cnn1d**：3 组 Conv1d（通道 32/64/128，核 7/5/3）+ BN + ReLU + MaxPool，自适应平均池化 → 共享特征。
- **CnnGru**：相同 CNN + 单层双向 GRU（单方向隐藏 64），拼接双向状态 → 共享特征。
- 两个输出头：四分类 logits + 连续直径（标签 = mil / 7，显示时 ×7）。
- 训练：AdamW（lr 1e-3，wd 1e-4），batch 128，最多 80 epoch；总损失（分类 + Smooth L1 回归，Normal 不参与回归）作为早停（耐心 12）、ReduceLROnPlateau（耐心 4，因子 0.5）与 `best_inference.pt` 保存的唯一监控指标。

## 输出结构

```
artifacts/
├── metadata.csv                    # 109 个文件的完整审计表
├── manifests/                      # 固定划分与实验清单（含归一化参数、类别权重）
├── runs/<实验>/<模型>/              # best_inference.pt / last_training.ckpt / history.json
├── metrics/                        # 每次评估的 JSON/CSV 与 compare_all.csv
└── figures/                        # 损失曲线、混淆矩阵、对比图等报告用图片
```

## 已知约束

- 28 mil 数据保留参与训练与评估，代码单独输出其细分指标（`mil28`）。
- 不使用厂家、故障深度、负载、RPM、BA、外圈位置作为模型输入，仅用于划分与分析。
- 不开发图形界面；训练、评估、推理均通过终端完成。
