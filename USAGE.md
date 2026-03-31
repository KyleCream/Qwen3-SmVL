# Qwen3-SmVL 训练使用文档

## 1. 环境准备

### 1.1 安装依赖

```bash
pip install -r requirements.txt
```

### 1.2 下载模型和数据

```bash
# 下载预训练模型
bash download_resource.sh

# 下载训练数据集（取消注释后执行）
modelscope download --dataset AI-ModelScope/the_cauldron --local_dir ./data/the_cauldron
```

下载完成后目录结构：

```
model/
  Qwen3-0.6B/
  SmolVLM2-256M-Video-Instruct/
data/
  the_cauldron/
    cocoqa/
    chartqa/
    ...
```

---

## 2. 模型架构

项目将 SmolVLM2 的视觉编码器 (SigLip-93M) 与 Qwen3-0.6B 中文语言模型融合，支持两种训练模式：

| 模式 | 说明 | 新增可训练参数 |
|------|------|---------------|
| 普通模式 | 只有主 Connector (768->1024) | ~12M |
| DeepStack 模式 | 主 Connector + 多层视觉特征注入 | ~12M + ~36M |

**DeepStack** 在视觉编码器的第 3/7/11 层捕获中间特征，通过独立的 Connector 投影后，在 LLM 的前 3 层以残差方式注入，提供更丰富的视觉信息。

---

## 3. 单阶段训练 (train.py)

冻结视觉编码器和语言模型，只训练 Connector。适合快速验证和资源有限的场景。

### 3.1 快速验证（cocoqa 数据集，200 步）

```bash
accelerate launch --num_processes 8 train.py ./cocoqa_train.yaml
```

单卡运行：

```bash
python train.py ./cocoqa_train.yaml
```

### 3.2 全量数据训练（60K 样本）

```bash
accelerate launch --num_processes 8 train.py ./full_train.yaml
```

### 3.3 配置说明

`cocoqa_train.yaml` / `full_train.yaml` 中的关键参数：

```yaml
# 数据选择："cocoqa" 为单数据集，"all" 为全部数据集（60K样本）
train_data: "cocoqa"

# DeepStack 开关
use_deepstack: true              # true 启用，false 使用普通模式
deepstack_layer_indexes: "3,7,11" # 视觉编码器提取层索引

# 训练超参
per_device_train_batch_size: 1   # 每卡 batch size
gradient_accumulation_steps: 4   # 梯度累积（有效 batch = 1 * 4 * num_gpus）
learning_rate: 0.0001            # 学习率
max_steps: 200                   # 最大训练步数（不设置则按 epoch）

# 精度
bf16: true                       # 使用 bfloat16 混合精度
```

---

## 4. 分阶段训练 (train_staged.py)

渐进式解冻训练，三个阶段逐步放开更多参数：

| 阶段 | 视觉编码器 | 语言模型 | Connector + DeepStack |
|------|-----------|---------|----------------------|
| Stage 1 | 冻结 | 冻结 | 训练 |
| Stage 2 | 训练 | 冻结 | 训练 |
| Stage 3 | 训练 | 训练 | 训练 |

### 4.1 完整三阶段训练

```bash
python train_staged.py ./staged_training.yaml
```

### 4.2 快速测试（5000 步）

```bash
python train_staged.py ./staged_training_test.yaml
```

### 4.3 从指定阶段恢复

如果 Stage 1 已完成，只需从 Stage 2 开始：

```yaml
training_stage: "all"
resume_from_stage: "stage2"
```

只运行某一个阶段：

```yaml
training_stage: "stage2"
```

### 4.4 配置说明

`staged_training.yaml` 中分阶段特有参数：

```yaml
# 运行哪些阶段："stage1" / "stage2" / "stage3" / "all"
training_stage: "all"

# 从哪个阶段恢复（跳过已完成的阶段）
resume_from_stage: null

# 各阶段独立的 epoch 和学习率
stage1_epochs: 1
stage1_lr: 0.0001    # Connector 学习率较高
stage2_epochs: 1
stage2_lr: 0.00005   # 视觉编码器用中等学习率
stage3_epochs: 1
stage3_lr: 0.00001   # 全量微调用低学习率
```

---

## 5. DeepStack 配置详解

在任意 YAML 配置中添加以下两行启用 DeepStack：

```yaml
use_deepstack: true
deepstack_layer_indexes: "3,7,11"
```

关闭 DeepStack（回退到普通模式）：

```yaml
use_deepstack: false
```

`deepstack_layer_indexes` 指定从视觉编码器（共 12 层，索引 0-11）的哪些层提取中间特征：

| 配置 | 提取层 | 注入 LLM 层 | 说明 |
|------|--------|------------|------|
| `"3,7,11"` | 低/中/高层 | 0, 1, 2 | 默认，均匀采样 |
| `"5,11"` | 中/高层 | 0, 1 | 减少参数量 |
| `"11"` | 仅最后层 | 0 | 最轻量 |

---

## 6. 推理

训练完成后使用 `inferance.py` 进行推理：

```bash
python inferance.py
```

需要修改 `inferance.py` 中的 `checkpoint_path` 指向训练输出目录。

---

## 7. 配置文件一览

| 文件 | 训练脚本 | 数据 | 用途 |
|------|---------|------|------|
| `cocoqa_train.yaml` | train.py | cocoqa (200步) | 快速验证 |
| `full_train.yaml` | train.py | 全部 (60K) | 正式训练 |
| `staged_training.yaml` | train_staged.py | 全部 (60K) | 分阶段正式训练 |
| `staged_training_test.yaml` | train_staged.py | cocoqa (5000步) | 分阶段快速测试 |

所有配置文件已默认启用 DeepStack。

---

## 8. 硬件需求

| 模式 | 最低显存 | 推荐 |
|------|---------|------|
| Stage 1（只训练 Connector） | ~16GB | 1x A100 40GB |
| Stage 2（训练视觉+Connector） | ~24GB | 1x A100 40GB |
| Stage 3（全量微调） | ~40GB | 1x A100 80GB 或多卡 |

使用 `gradient_checkpointing: true` 可降低约 30% 的显存占用，但会增加训练时间。
