# CWRU 轴承故障诊断（分类 + 故障直径回归）

基于 Case Western Reserve University (CWRU) 轴承数据集的 PyTorch 项目：
四分类（Normal / IR / OR / B）+ 连续故障直径回归。

扩展实验矩阵为 **3 套固定划分 × 2 种窗口方案 × 5 种输入方案 × 3 种模型 = 90 组正式实验**，
每套划分得到一组结果，最终报告 A/B/C 的平均值与标准差。

## 环境

- Python 3.11（conda 环境 `AI2026Summer`）
- torch 2.11.0+cu128（CUDA 可用即可，CPU 也能运行但更慢）
- scipy / numpy / matplotlib / pandas

```bash
pip install -r requirements.txt
```

数据：`CaseWesternReserveUniversityData/` 下 109 个 MAT 文件（勿改动，首次提交后视为只读）。

## 快速开始

```bash
# 1. 数据审计（文件、变量、特殊映射）
python -m cwru audit-data

# 2. 将代码中写死的 A/B/C 三套固定划分写入 splits/ 目录
python -m cwru split-init

# 3. 生成元数据与全部实验清单（30 个清单）
python -m cwru prepare

# 4. 完整正式批次：写入固定目录 formal_run 并执行 90 组训练 + 评估 + 汇总
python -m cwru full-run

#    流程自检（不产生正式结论）：每组合 1 个 epoch，仅前 N 组
python -m cwru full-run --epochs 1 --limit 3

# 5. 训练单个组合 / 评估单个组合
python -m cwru train --experiment 101DE --model CnnLstm --split A --window overlap50
python -m cwru evaluate --experiment 101DE --model CnnLstm --split A --window overlap50

# 6. 重新汇总某个批次
python -m cwru compare --batch artifacts/full_runs/formal_run

# 7. 终端推理
python -m cwru predict --experiment 101DE --model Cnn1d --file CaseWesternReserveUniversityData/normal_0_97.mat
# DEFE 单通道实验必须显式指定测量端：
python -m cwru predict --experiment 109DEFE --model CnnGru --file CaseWesternReserveUniversityData/12k_Fan_End_B007_0_282.mat --channel FE --split A --window no_overlap
```

## 实验配置

### 输入方案（5 种）

| 实验 | 文件集合 | 通道方案 |
|---|---|---|
| `dual101` | 101 文件 | [DE, FE] 双通道 |
| `109DEFE` | 109 文件 | 驱动端故障取 DE，风扇端故障取 FE，正常取 DE |
| `101DEFE` | 101 文件 | 同上（排除 28 mil） |
| `109DE` | 109 文件 | 全部取 DE |
| `101DE` | 101 文件 | 全部取 DE |

### 窗口方案（2 种）

| 方案 | 窗口长度 | 步长 | 重叠 |
|---|---:|---:|---:|
| `no_overlap` | 1024 | 1024 | 0% |
| `overlap50` | 1024 | 512 | 50% |

50% 重叠使窗口数接近翻倍，但相邻窗口共享 512 个采样点，不代表独立信息翻倍。
重叠只发生在同一文件内部，训练/验证/测试的文件级隔离在切窗之前完成，因此不会产生集合间窗口泄漏。

### 模型（3 种）

- **Cnn1d**：3 组 Conv1d（通道 32/64/128，核 7/5/3）+ BN + ReLU + MaxPool，自适应平均池化 → 共享特征。
- **CnnGru**：相同 CNN 主干 + 单层双向 GRU（单方向隐藏 64），拼接双向状态 → 128 维共享特征。
- **CnnLstm**：与 CnnGru 结构对齐，仅把循环单元换成单层双向 LSTM（单方向隐藏 64）。
- 三个模型共享两个输出头：四分类 logits + 连续直径（标签 = mil / 7，显示时 ×7）。
- 训练条件统一：AdamW（lr 1e-3，wd 1e-4），batch 128，最多 80 epoch，TRAIN_SEED = 42；
  总损失（分类 + Smooth L1 回归，Normal 不参与回归）作为早停（耐心 12）、
  ReduceLROnPlateau（耐心 4，因子 0.5）与 `best_inference.pt` 保存的唯一监控指标。

## 固定文件划分

- 三套固定划分 A / B / C 的验证和测试文件名直接写在 `cwru/data/split.py` 中；
  不使用划分随机种子，也不在运行时生成候选或重抽。
- 数量硬约束：101 集合 71/15/15，109 集合 75/17/17；101/109 公共文件归属完全一致。
- 每套验证/测试均覆盖四类、Drive/Fan、7/14/21 mil、四档负载和 OR 三个钟点；
  每类至少保留合理文件数，且验证/测试出现的条件都在训练集中有对应条件。
- 三套 109 测试集完全不重复；验证集只保留少量重复。
- Normal 与 28 mil 文件各自仍按 2/1/1 分配，具体名单和分布见 `splits/A|B|C/`。

## 数据处理规则

- Normal 原始 48 kHz，`scipy.signal.resample_poly(sig, 1, 4)` 降采样到 12 kHz；故障数据保持 12 kHz。
- 先划分文件再切窗口；窗口长度 1024，步长由窗口方案决定（1024 或 512），尾部不足丢弃。
- 归一化参数仅从各组合训练集窗口计算；类别权重 = 1/√(训练窗口数)，归一化到均值 1。
- 特殊变量映射：`normal_2_99.mat` → X099；`12k_Fan_End_IR014_1_276.mat` → X275；28 mil 文件从内容识别唯一 `*_DE_time`。
- 无重叠窗口数：101 方案 11,891；109 方案 12,833。50% 重叠约为其两倍（23,730 / 25,612）。

## 评价指标

- **窗口级**：Accuracy、Macro-F1、混淆矩阵、各类别 P/R/F1、直径 MAE/RMSE 与分直径档位误差、OR 钟点细分、28 mil 细分。
- **文件级**：类别取窗口平均概率最大者；**连续直径取该文件全部故障窗口的模型预测均值**（不使用真实标签聚合），
  Normal 不参与故障直径 MAE/RMSE。
- `metrics.json` 只保存汇总指标；`files.csv` 只保存逐测试文件的真实值、预测值与属性，两者不重复。

## 输出结构

```
splits/
├── A|B|C/
│   ├── split_101.json / split_109.json   # 固定划分（正式训练权威来源）
│   ├── balance.csv                        # 各集合 Drive/Fan/类别/直径/负载/钟点分布
│   └── README.md                          # 生成方法、约束与无法完全平衡的稀有组合
artifacts/
├── metadata.csv                # 109 个文件的完整审计表
├── manifests/                  # 30 个实验清单（键 = 划分_输入_窗口）
├── runs/                       # 单组命令的兼容输出
└── full_runs/formal_run/        # 完整批次固定目录；再次完整运行会覆盖同名结果
    ├── README.md               # 逐组更新进度与结果，含 A/B/C 均值标准差与结论
    ├── run_config.json
    ├── splits/                 # 本批次使用的三套划分快照
    ├── <A|B|C>/<窗口>/<输入>/<模型>/
    │   ├── best_inference.pt / last_training.ckpt
    │   ├── history.json / metrics.json / files.csv
    │   └── figures/
    └── comparisons/            # metrics_all.csv、mean_std_summary.csv、对比图、conclusions.json
```

本地旧版 10 组结果归档在 `artifacts/full_runs/legacy_20260915_1353_original/`，
仅作过程记录，不进入 90 组正式统计；该目录已在 `.gitignore` 中排除，不会推送到远程。

## 已知约束

- 28 mil 数据保留参与训练与评估，代码单独输出其细分指标（`mil28`）。
- 不使用厂家、故障深度、负载、RPM、BA、外圈位置作为模型输入，仅用于划分与分析。
- 不开发图形界面；训练、评估、推理均通过终端完成。
- 本轮不针对过拟合增加 Dropout、数据增强或正则化，保留既有早停与学习率调度。
- A/B/C 的标准差反映对文件划分的敏感性，不代表多训练种子方差。
