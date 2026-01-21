# OULAD 数据集转换与 ContraTGT 运行指南

本指南说明如何将 OULAD 教育日志数据转换为 ContraTGT 格式，并在 ContraTGT 框架中运行。

## 目录结构

确保您的项目目录结构如下：

```
code/
├── convert_oulad.py          # 数据转换脚本
├── OULAD-main/
│   └── data/                 # OULAD 原始数据目录
│       ├── studentVle.csv
│       ├── studentInfo.csv
│       └── courses.csv
├── ContraTGT/                # ContraTGT 框架目录
│   ├── data/                 # 转换后的边数据将保存在此
│   ├── node_feature/         # 节点特征文件将保存在此
│   ├── main.py
│   ├── pretrain.py
│   └── utils.py
└── data/                     # 转换后的边数据输出目录
```

## 步骤 1: 数据转换

### 1.1 准备原始数据

确保 `OULAD-main/data/` 目录下包含以下文件：
- `studentVle.csv` - 学生与虚拟学习环境交互数据
- `studentInfo.csv` - 学生信息数据
- `courses.csv` - 课程信息数据

### 1.2 运行转换脚本

在项目根目录下运行：

```bash
python convert_oulad.py
```

### 1.3 转换输出

转换脚本会生成两个文件：

1. **边数据文件**: `data/ml_oulad.csv`
   - 格式：`id,u,i,ts,label,idx`
   - `u`: 学生节点ID（从1开始）
   - `i`: 课程节点ID（从1开始）
   - `ts`: 时间戳（整数，从1开始）
   - `label`: 边标签（1=通过，-1=失败/退学）
   - `idx`: 边索引

2. **节点特征文件**: `ContraTGT/node_feature/oulad.content`
   - 每行代表一个节点的特征向量（逗号分隔）
   - 特征包括：学生特征（年龄、教育背景、学分等）和课程特征（点击量、活跃学生数、课程长度等）

### 1.4 验证转换结果

转换完成后，脚本会输出数据统计信息：
- 学生节点数
- 课程节点数
- 总边数
- 特征维度
- 正负样本分布

## 步骤 2: 在 ContraTGT 中运行

### 2.1 确认数据集已注册

检查 `ContraTGT/utils.py` 文件，确保 `'oulad'` 已在数据集选择列表中：

```python
parser.add_argument('-d', '--data', type=str, help='data sources to use',
                    choices=['socialevolve_1m', 'wiki', 'slashdot', 'bitcoinotc','ubuntu','oulad'],
                    default='slashdot')
```

**注意**: 如果 `convert_oulad.py` 脚本已更新，`'oulad'` 应该已经包含在列表中。

### 2.2 预训练模型

进入 `ContraTGT` 目录并运行预训练：

```bash
cd ContraTGT
python pretrain.py -d oulad --bs 800 --ctx_sample 30 --tmp_sample 21 --seed 60
```

**参数说明**:
- `-d oulad`: 指定数据集为 oulad
- `--bs 800`: 批次大小
- `--ctx_sample 30`: 空间采样数量
- `--tmp_sample 21`: 时间采样数量
- `--seed 60`: 随机种子

预训练模型将保存到 `ContraTGT/pretrain_model/oulad.pth`

### 2.3 训练模型

运行主训练脚本：

```bash
python main.py -d oulad --bs 800 --ctx_sample 40 --tmp_sample 31 --seed 60
```

**参数说明**:
- `-d oulad`: 指定数据集为 oulad
- `--bs 800`: 批次大小
- `--ctx_sample 40`: 空间采样数量
- `--tmp_sample 31`: 时间采样数量
- `--seed 60`: 随机种子

训练好的模型将保存到 `ContraTGT/saved_models/oulad.pth`

### 2.4 调整参数（可选）

如果数据量很大或遇到内存问题，可以调整以下参数：

- **减小批次大小**: `--bs 400` 或 `--bs 200`
- **减小采样数量**: `--ctx_sample 20 --tmp_sample 15`
- **调整学习率**: `--lr 3e-4`（对于大数据集可能需要更小的学习率）

## 完整运行示例

```bash
# 1. 转换数据
python convert_oulad.py

# 2. 进入 ContraTGT 目录
cd ContraTGT

# 3. 预训练
python pretrain.py -d oulad --bs 800 --ctx_sample 30 --tmp_sample 21 --seed 60

# 4. 训练
python main.py -d oulad --bs 800 --ctx_sample 40 --tmp_sample 31 --seed 60
```

## 数据格式说明

### 边数据格式 (ml_oulad.csv)

```csv
id,u,i,ts,label,idx
1,1,1001,1,1,1
2,1,1001,5,1,2
3,2,1002,10,-1,3
...
```

- 所有节点索引从 **1** 开始（0 保留用于填充）
- 时间戳为整数，从 **1** 开始
- 标签：`1` = 通过，`-1` = 失败/退学

### 节点特征格式 (oulad.content)

每行是一个节点的特征向量，用逗号分隔：

```
0.0,1.0,0.0,0.5,1.0,0.0,0.0,0.0,1,0
0.0,0.0,1.0,0.3,0.0,2.5,1.2,30.0,0,1
...
```

特征包括：
- 学生特征：年龄组（one-hot，3维）、教育背景（one-hot，5维）、学分（归一化，1维）、残疾标识（1维）
- 占位符/统计量：3维（学生用占位符，课程用统计量）
- 节点类型标识：`[is_student, is_course]`（2维）
- 填充维度：1维（确保总维度16能被4整除，满足多头注意力要求）

**总特征维度：16**（3+5+1+1+3+2+1=16）

## 常见问题

### Q: 转换脚本报错找不到数据目录
**A**: 确保 `OULAD-main/data/` 目录存在，且包含所需的 CSV 文件。

### Q: 预训练或训练时内存不足
**A**: 尝试减小批次大小（`--bs`）和采样数量（`--ctx_sample`, `--tmp_sample`）。

### Q: 模型性能不佳
**A**: 可以尝试：
- 调整学习率（`--lr`）
- 增加训练轮数（`--n_epoch`）
- 调整 dropout 率（`--drop_out`）

### Q: 预训练时出现 AssertionError
**A**: 这通常是因为特征维度不能被多头注意力的头数（4）整除。转换脚本已自动处理此问题，通过添加填充维度使特征维度为16（能被4整除）。如果仍遇到此问题，请重新运行 `convert_oulad.py` 生成新的特征文件。

### Q: 如何查看训练进度
**A**: 训练过程中会输出每个 epoch 的损失、准确率、AP 和 AUC 指标。

## 注意事项

1. **节点索引**: 所有节点索引必须从 1 开始，0 保留用于填充操作
2. **时间戳**: 建议将时间戳离散化为整数以便索引
3. **数据量**: 如果 OULAD 数据集很大，可能需要调整批次大小和采样参数
4. **GPU**: 确保有可用的 GPU（默认使用 GPU 0），可通过 `--gpu` 参数指定

## 参考

- ContraTGT 原始 README: `ContraTGT/README.md`
- 数据转换脚本: `convert_oulad.py`

