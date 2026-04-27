# 指令微调 (SFT) 学习文档

> 通过有监督学习教会预训练LLM遵循人类指令并完成任务。

> 来源线索：本节内容根据原书中关于"instruction tuning / SFT"的相关章节整理、扩展与教学化改写。

## 1. 算法基础认知

### 一句话定义
指令微调 (SFT) 是用标注好的指令-回答数据对预训练LLM进行有监督微调的方法。

### 直觉类比
一个博学但不懂社交的人（预训练LLM），你给他看一系列"别人问什么-他怎么答"的范例后，他就学会了如何回应他人的提问。SFT 就是给 LLM 看这类"问答范例"。

### 历史背景
指令微调的概念在 2022 年左右随着 FLAN、T0 等研究被系统化。ChatGPT (2022) 是 SFT 大规模成功应用的代表——先经过 SFT 学习对话格式，再经过 RLHF 对齐偏好。如今 SFT 已成为开发LLM的标准后训练流程之一。

### 算法定位
- **类型**：监督学习 / 后训练阶段 / 微调方法
- **性质**：需要标注数据集，修改模型权重

### 前置知识
- 了解 LLM 预训练的基本概念
- 理解梯度下降和损失函数
- 了解 Transformer 架构（不必须深入）

## 2. 核心原理

### 核心思想
预训练LLM学会了"续写文本"，但不擅长"回答问题"。SFT 用指令-回答对训练模型，让模型从"续写模式"转变为"指令遵循模式"。本质上，SFT 是让模型学会从指令到回答的条件映射。

### 工作流程
1. 准备指令数据集：每条数据 = (指令/问题, 期望回答)
2. 将数据格式化为统一的对话模板（如 `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...`）
3. 用标准语言模型训练目标（下一个 token 预测）在这些格式化数据上微调模型
4. 损失仅计算在"回答"部分的 token 上（通常 mask 掉指令部分的 loss）

### 关键概念解释
- **指令数据集**：包含 (instruction, output) 对的数据，指令可以是问题、任务描述、要求等
- **对话模板**：将原始文本包装成统一的格式，如 ChatML、ShareGPT 格式
- **Loss Masking**：在 SFT 中，通常只对 assistant 的输出部分计算损失，用户指令部分不参与 loss 计算
- **多轮对话**：SFT 数据可以包含多轮交互，帮助模型学会上下文对话

### 直观解释
```
预训练LLM看到 "中国的首都是"  →  预测 "北京"
               (学会了续写，但不会回答问题)

SFT后LLM看到 "中国首都是哪里？"  →  回答 "中国的首都是北京。"
               (学会了理解指令并给出合适的回答)
```

## 3. 数学公式与推导

### 符号约定
| 符号 | 含义 |
|------|------|
| $\theta$ | 模型参数 |
| $x = (x_1, ..., x_n)$ | 指令部分 token 序列 |
| $y = (y_1, ..., y_m)$ | 回答部分 token 序列 |
| $\mathcal{D} = \{(x_i, y_i)\}$ | 指令数据集 |
| $L$ | 损失函数 |

### 问题形式化

SFT 的目标是最大化给定指令 $x$ 时生成回答 $y$ 的条件概率：

$$\max_\theta \sum_{(x,y) \in \mathcal{D}} \log p_\theta(y | x)$$

### 自回归分解

回答 $y$ 在自回归模型中的概率分解为：

$$p_\theta(y | x) = \prod_{t=1}^{m} p_\theta(y_t | x, y_{<t})$$

其中 $y_{<t}$ 表示前 $t-1$ 个回答 token。

### 损失函数

SFT 使用标准交叉熵损失（即负对数似然）：

$$\mathcal{L}(\theta) = -\frac{1}{|\mathcal{D}|} \sum_{(x,y) \in \mathcal{D}} \frac{1}{|y|} \sum_{t=1}^{|y|} \log p_\theta(y_t | x, y_{<t})$$

实际上，整个序列（指令+回答）都会被输入模型，但损失只在回答部分计算：

$$\mathcal{L}_{\text{mask}}(\theta) = -\frac{1}{|\mathcal{D}|} \sum_{(x,y) \in \mathcal{D}} \frac{1}{\sum_t m_t} \sum_{t} m_t \cdot \log p_\theta(z_t | z_{<t})$$

其中 $z = [x; y]$ 是完整序列，$m_t \in \{0, 1\}$ 表示该位置是否计算损失（回答部分为1，指令部分为0）。

## 4. 训练过程讲解

### 数据预处理
1. **收集/生成指令数据**：人工编写、从已有NLP任务转换、用更强模型生成
2. **数据清洗**：去掉太短/太长/质量差的数据
3. **格式化**：统一应用对话模板
4. **Token化**：将格式化文本转为 token IDs
5. **填充与截断**：填充到统一长度或按最大长度截断

### 参数初始化
- 从预训练 base 模型（或已有的指令模型）开始，不随机初始化
- 这是关键——SFT 是"微调"而非"从头训练"

### 迭代过程
1. 从数据集中采样一批 (instruction, response) 对
2. 格式化为完整对话序列并 tokenize
3. 前向传播计算 logits
4. 计算回答部分的交叉熵损失（mask掉指令部分）
5. 反向传播更新模型参数
6. 重复直到验证损失收敛

### 收敛条件
- 验证损失不再下降（通常在 1-3 个 epoch 后收敛）
- SFT 数据量通常较小（几千到几万条），过多 epoch 容易过拟合

### 超参数表
| 参数 | 作用 | 推荐范围 | 默认建议 |
|------|------|----------|----------|
| learning_rate | 学习率 | 1e-6 ~ 5e-5 | 2e-5 |
| batch_size | 批次大小 | 4 ~ 128 | 32 |
| epochs | 训练轮数 | 1 ~ 5 | 2-3 |
| max_seq_length | 最大序列长度 | 512 ~ 4096 | 2048 |
| warmup_ratio | 学习率预热比例 | 0.03 ~ 0.1 | 0.03 |
| weight_decay | 权重衰减 | 0.0 ~ 0.1 | 0.0 |

## 5. 应用场景

### 典型应用
1. **通用的LLM对话助手**：ChatGPT、Claude 等都经过 SFT。训练数据覆盖各种话题和任务类型。
2. **领域专用助手**：医疗问诊、法律咨询、代码助手。结合领域指令数据微调。
3. **格式遵循任务**：信息提取、JSON生成、翻译。SFT 教会模型遵循特定的输出格式。
4. **推理模型的基础**：SFT 是开发推理模型的第一步——先用推理格式的数据做 SFT建立基础，再用RL强化。

### 适用数据特征
- 拥有高质量、多样化的指令-回答数据
- 任务目标是改进模型的指令遵循能力
- 基座模型已经过充分预训练

### 不适用场景
- 完全没有标注数据的场景（可用 zero-shot / few-shot 提示替代）
- 需要探索性学习（SFT 是模仿学习，不产生超越数据的策略）
- 基座模型太弱（SFT 无法弥补预训练不足）

## 6. 优缺点分析

### 优点
| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 简单直接 | 标准监督学习，工具链成熟 | 有可用数据 |
| 效果明显 | 能显著改善模型的指令遵循和对话能力 | 数据质量高且多样化 |
| 成本可控 | SFT 数据量通常几千到几万条，训练时间短 | 有预训练好的基座模型 |
| 可控性强 | 可以通过数据内容精确控制行为方向 | 数据集设计合理 |

### 缺点
| 缺点 | 说明 | 缓解思路 |
|------|------|----------|
| 依赖数据质量 | 数据里的偏见、错误会被模型学习 | 严格的数据质量控制和多样性检查 |
| 可能过拟合 | 小数据集容易记住而非泛化 | 早停、正则化、数据增强 |
| 无法超越数据 | 只能模仿训练数据中的行为模式 | 结合RL进一步优化 |
| 格式敏感 | 不同对话模板效果可能差异大 | 使用业界标准模板并实验对比 |

### 与同类方法对比
| 方法 | 数据需求 | 训练复杂度 | 对基础能力的改进 | 对偏好的改进 |
|------|----------|------------|------------------|-------------|
| SFT | 指令-回答对 | 低 | 强 | 弱 |
| RLHF/DPO | 偏好对比数据 | 中-高 | 弱 | 强 |
| 持续预训练 | 大量领域文本 | 高 | 增强领域知识 | 无 |

## 7. 调库实现

```python
"""
指令微调 (SFT) 的调库实现
使用 HuggingFace Transformers + TRL (Transformer Reinforcement Learning)
"""

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import json


# ===== 1. 准备数据 =====
def prepare_sft_dataset(data_file: str, tokenizer, max_length: int = 2048):
    """
    加载并处理 SFT 数据

    数据格式 (JSONL):
    {"instruction": "...", "output": "..."}
    """

    def format_and_tokenize(example):
        # 对话模板格式化
        formatted = (
            f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        )

        # Tokenize
        tokenized = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # SFT 的关键：创建 labels
        # 指令部分的 labels 设为 -100（在 loss 中被忽略）
        input_ids = tokenized["input_ids"]

        # 找到 assistant 部分开始的位置，只保留回答部分的 labels
        # 简化处理：将序列复制为 labels
        labels = input_ids.copy()

        # 将指令部分的 labels 设为 -100
        # 找到 "<|im_start|>assistant" 的位置
        assistant_start_token = tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )
        for i in range(len(input_ids) - len(assistant_start_token) + 1):
            if input_ids[i:i + len(assistant_start_token)] == assistant_start_token:
                labels[:i + len(assistant_start_token)] = [-100] * (i + len(assistant_start_token))
                break

        tokenized["labels"] = labels
        return tokenized

    import json
    data_list = []
    with open(data_file, 'r') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))

    dataset = Dataset.from_list(data_list)
    tokenized_dataset = dataset.map(
        format_and_tokenize,
        remove_columns=dataset.column_names,
    )
    return tokenized_dataset


# ===== 2. 配置 LoRA 微调（节省显存） =====
def setup_lora_model(model):
    """配置 LoRA 适配器"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                    # LoRA 秩
        lora_alpha=32,           # LoRA 缩放因子
        lora_dropout=0.05,       # LoRA dropout
        target_modules=[         # 在哪些模块上应用 LoRA
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ===== 3. 训练配置 =====
def train_sft(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    data_file: str = "sft_data.jsonl",
    output_dir: str = "./sft_model",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 可选：使用 LoRA 节省显存
    model = setup_lora_model(model)

    # 准备数据集
    dataset = prepare_sft_dataset(data_file, tokenizer)

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 不是 MLM，是因果语言模型
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # 梯度累积
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,                  # 使用 bfloat16
        gradient_checkpointing=True, # 节省显存
        report_to="none",
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


# ===== 4. 创建示例数据并测试 =====
# 创建简单的示例 SFT 数据
sample_data = [
    {"instruction": "1+1等于多少？", "output": "1+1等于2。"},
    {"instruction": "请用一句话介绍中国。", "output": "中国是一个拥有悠久历史和丰富文化的东方大国。"},
    {"instruction": "翻译：Hello, how are you?", "output": "你好，你怎么样？"},
]

with open("sample_sft_data.jsonl", "w") as f:
    for item in sample_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("示例 SFT 数据已写入 sample_sft_data.jsonl")
print("运行 train_sft(data_file='sample_sft_data.jsonl') 即可开始训练")

# 注意：实际运行时取消注释下面的代码
# model, tokenizer = train_sft(data_file="sample_sft_data.jsonl")
```

## 8. 手工代码实现

```python
"""
指令微调 (SFT) 的手工实现
从零实现 SFT 的核心训练循环
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class SFTTrainer:
    """
    手工实现 SFT 训练器

    核心：标准的监督微调，损失只在 assistant 回答部分计算
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,  # 需要提供 encode/decode
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)

    def format_prompt(
        self,
        instruction: str,
        response: str = None,
    ) -> str:
        """
        格式化指令为对话模板

        这种格式化是 SFT 成功的关键——统一的格式让模型学会
        区分"用户说了什么"和"我应该回答什么"。
        """
        # 使用 ChatML 风格的模板
        formatted = f"<|im_start|>user\n{instruction}<|im_end|>\n"

        if response is not None:
            formatted += f"<|im_start|>assistant\n{response}<|im_end|>"
        else:
            formatted += "<|im_start|>assistant\n"

        return formatted

    def create_training_batch(
        self,
        batch_data: List[Tuple[str, str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建训练批次：将(指令, 回答)对转为 input_ids + labels + attention_mask

        注意：labels 中指令部分设为 -100（ignore_index），
        这样 CrossEntropyLoss 只计算回答部分的损失。
        """
        all_input_ids = []
        all_labels = []
        all_attention_masks = []

        for instruction, response in batch_data:
            full_text = self.format_prompt(instruction, response)
            # 指令部分的文本（无回答）
            instruction_only = self.format_prompt(instruction)

            # Tokenize 完整序列
            full_ids = self.tokenizer.encode(full_text)
            # Tokenize 仅指令部分
            inst_ids = self.tokenizer.encode(instruction_only)

            input_ids = full_ids.copy()
            # 指令部分的 token 不计入损失
            labels = [-100] * len(inst_ids) + full_ids[len(inst_ids):]
            attention_mask = [1] * len(input_ids)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attention_mask)

        # Padding 到相同长度
        max_len = max(len(ids) for ids in all_input_ids)

        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []

        for input_ids, labels, mask in zip(
            all_input_ids, all_labels, all_attention_masks
        ):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(
                input_ids + [self.tokenizer.pad_token_id] * pad_len
            )
            padded_labels.append(labels + [-100] * pad_len)
            padded_attention_masks.append(mask + [0] * pad_len)

        return (
            torch.tensor(padded_input_ids, device=self.device),
            torch.tensor(padded_labels, device=self.device),
            torch.tensor(padded_attention_masks, device=self.device),
        )

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        单步训练：前向传播 + 损失计算 + 反向传播

        SFT 和预训练的唯一区别在于 loss masking——
        预训练对所有 token 计算 loss，SFT 只对回答部分计算。
        """
        self.model.train()

        # 前向传播
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # 形状: (batch, seq_len, vocab_size)

        # 计算损失（仅计算 labels != -100 的位置）
        # Shift logits 和 labels 对齐（因为预测的是下一个 token）
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止训练不稳定
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()

    def train(
        self,
        train_data: List[Tuple[str, str]],
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 8,
    ) -> List[float]:
        """完整的 SFT 训练循环"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        losses = []

        for epoch in range(num_epochs):
            epoch_losses = []

            # 每个 epoch 打乱数据
            indices = np.random.permutation(len(train_data))
            for i in range(0, len(train_data), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = [train_data[idx] for idx in batch_indices]

                input_ids, labels, attention_mask = self.create_training_batch(batch)
                loss = self.train_step(
                    input_ids, labels, attention_mask, optimizer
                )
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

        return losses


# ===== (简化) 使用示例 =====
# 假设有一个模拟的 tokenizer 和 model
class SimpleTokenizer:
    """极简 tokenizer 用于教学"""
    def __init__(self):
        self.word2id = {
            "<|im_start|>": 0, "<|im_end|>": 1,
            "user": 2, "assistant": 3, "\n": 4,
            "1+1": 5, "等于多少": 6, "等于": 7, "2": 8, "。": 9,
        }
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.pad_token_id = 99
        self.eos_token_id = 100

    def encode(self, text):
        # 极简实现：按已知词匹配
        ids = []
        remaining = text
        while remaining:
            matched = False
            for word, wid in sorted(
                self.word2id.items(), key=lambda x: -len(x[0])
            ):
                if remaining.startswith(word):
                    ids.append(wid)
                    remaining = remaining[len(word):]
                    matched = True
                    break
            if not matched:
                ids.append(99)  # 未知
                remaining = remaining[1:]
        return ids

    def decode(self, ids):
        return " ".join(self.id2word.get(i, "?") for i in ids)


class SimpleModel(nn.Module):
    """极简模型用于教学"""
    def __init__(self, vocab_size=100, embed_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        logits = self.lm_head(x)
        # 返回一个类似 HuggingFace 输出格式的对象
        class Output:
            pass
        output = Output()
        output.logits = logits
        return output


# 测试手工实现
train_data = [
    ("1+1等于多少？", "1+1等于2。"),
    ("今天天气怎么样？", "今天天气晴朗。"),
    ("翻译hello", "你好。"),
    ("解释AI是什么", "AI是人工智能的缩写。"),
]

tokenizer = SimpleTokenizer()
model = SimpleModel(vocab_size=200, embed_dim=32)

trainer = SFTTrainer(model, tokenizer, device="cpu")
losses = trainer.train(train_data, num_epochs=5, batch_size=2)

print(f"\n训练完成！各 epoch 损失: {losses}")
```

## 9. 可视化与结果理解

```python
"""
SFT 训练过程的损失曲线可视化
"""

import matplotlib.pyplot as plt
import numpy as np

# ---- 模拟 SFT 训练损失 ----
epochs = np.arange(1, 6)
# 典型的 SFT 损失下降曲线（模拟数据）
train_loss = [3.45, 2.18, 1.65, 1.38, 1.22]
val_loss =   [3.52, 2.35, 1.78, 1.52, 1.48]  # 第5个epoch验证损失开始上升

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ---- 左图：损失曲线 ----
ax1.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=8, label="训练损失")
ax1.plot(epochs, val_loss, 'r-s', linewidth=2, markersize=8, label="验证损失")
ax1.axvline(x=3, color='green', linestyle='--', alpha=0.7,
            label="建议早停点 (Epoch 3)")
ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
ax1.set_ylabel("Cross-Entropy Loss", fontsize=12, fontweight="bold")
ax1.set_title("SFT 训练损失曲线\n3个Epoch后验证损失不再降低，建议早停", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# ---- 右图：数据量 vs 效果 ----
data_sizes = [100, 500, 1000, 2000, 5000, 10000]
# 不同数据量下的模拟评测分数
benchmark_scores = [45, 62, 71, 78, 83, 86]
ax2.plot(data_sizes, benchmark_scores, 'g-D', linewidth=2, markersize=10)
ax2.set_xlabel("SFT 训练数据量", fontsize=12, fontweight="bold")
ax2.set_ylabel("基准评测分数", fontsize=12, fontweight="bold")
ax2.set_title("数据量对 SFT 效果的边际收益\n超过2000条后每千条提升小于3%", fontsize=13, fontweight="bold")
ax2.grid(alpha=0.3)

# 标注"甜蜜区间"
ax2.axvspan(1000, 3000, alpha=0.15, color='green')
ax2.annotate("性价比最优区间",
             xy=(2000, 78), fontsize=10, fontweight="bold",
             color="green",
             ha="center")
# 标注"边际递减区"
ax2.axvspan(5000, 10000, alpha=0.1, color='orange')
ax2.annotate("边际递减",
             xy=(7500, 84), fontsize=10, color="orange",
             ha="center")

plt.tight_layout()
plt.show()

print("""
图表解读：
左图：
- 前两个epoch损失急剧下降，说明模型快速学会了基本的指令遵循
- 第3个epoch后验证损失不再下降，继续训练只会过拟合训练数据
- SFT 通常1-3个epoch就足够——它是微调，不是从头训练

右图：
- 数据量从100到1000带来25%+的评测分数提升——早期数据质量回报极高
- 1000-3000条数据是"甜蜜区间"：成本可控且效果足够
- 超过5000条后边际收益递减严重，不如提升数据质量
""")
```

## 10. 模型评估

```python
"""
SFT 模型的评估方法
"""

from typing import List, Dict
import re


class SFTEvaluator:
    """评估 SFT 后的模型性能"""

    @staticmethod
    def instruction_following_accuracy(
        predictions: List[str],
        expected_keywords: List[List[str]],
    ) -> Dict:
        """
        评估指令遵循的准确率

        例如：指令"用中文回答"→ 回答中应包含中文字符
        指令"用JSON格式"→ 回答应是可以解析的JSON
        """
        results = []
        for pred, keywords in zip(predictions, expected_keywords):
            # 检查是否所有关键词/模式都在回答中
            matches = [kw.lower() in pred.lower() for kw in keywords]
            accuracy = sum(matches) / len(matches) if matches else 0.0
            results.append({
                "prediction": pred[:100],
                "keywords": keywords,
                "matches": matches,
                "accuracy": accuracy,
            })

        overall = {
            "total_samples": len(predictions),
            "fully_followed": sum(1 for r in results if r["accuracy"] == 1.0),
            "avg_keyword_accuracy": np.mean([r["accuracy"] for r in results]),
            "samples": results,
        }
        return overall

    @staticmethod
    def format_compliance_rate(predictions: List[str], required_format: str) -> float:
        """检查输出格式的符合率（如JSON格式）"""
        if required_format == "json":
            import json
            success = 0
            for pred in predictions:
                try:
                    # 尝试提取并解析JSON
                    json_match = re.search(r'\{.*\}', pred, re.DOTALL)
                    if json_match:
                        json.loads(json_match.group())
                        success += 1
                except:
                    pass
            return success / len(predictions)
        return 0.0

    @staticmethod
    def response_length_stats(predictions: List[str]) -> Dict:
        """回答长度的统计（太短→不完整，太长→冗余）"""
        lengths = [len(pred) for pred in predictions]
        return {
            "min": min(lengths),
            "max": max(lengths),
            "mean": np.mean(lengths),
            "median": np.median(lengths),
            "std": np.std(lengths),
        }


# ===== 使用示例 =====
predictions = [
    "1+1等于2。这是基本的加法运算。",
    "中国是一个拥有丰富文化和悠久历史的东方大国。",
    "你好，今天天气怎么样？",
]

# 评估例1：关键词检查（是否遵循了指令中的要求）
evaluator = SFTEvaluator()
result_kw = evaluator.instruction_following_accuracy(
    predictions,
    expected_keywords=[["2", "加法"], ["中国", "文化"], ["你好"]],
)
print(f"完全遵循率: {result_kw['fully_followed']}/{result_kw['total_samples']}")
print(f"平均关键词匹配率: {result_kw['avg_keyword_accuracy']:.1%}")

# 评估例2：长度统计
length_stats = evaluator.response_length_stats(predictions)
print(f"回答长度: 平均{length_stats['mean']:.1f}字符 (范围 {length_stats['min']}-{length_stats['max']})")

print("""
SFT 评估注意事项：
1. 指令遵循是核心指标——是否按要求格式、语言、长度回答
2. 不能仅依赖BLEU/ROUGE——这些指标不擅长评估开放性对话
3. 实际项目中常用"LLM-as-judge"——用更强的LLM评判SFT模型回答质量
""")
```

## 11. 常见问题与易错点

### 数据层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 数据格式不一致 | 模型时而按一种格式回答时而又换另一种 | 训练数据用了多种互不兼容的对话模板 | 统一所有数据到一种模板格式 |
| 数据分布偏差 | 模型在某一类任务上变好但在其他任务上退化 | 数据集中某类任务占比过高 | 平衡各类任务的采样比例 |

### 模型层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 灾难性遗忘 | SFT后模型丢失了预训练获得的通用知识 | 学习率太大/epoch太多 | 降低学习率到1e-6，减少epoch到1-2 |
| 过拟合指令模板 | 只能回复训练数据里见过的指令形式 | 数据多样性不够 | 增加数据的表述多样性（同义改写） |
| 未正确处理 loss mask | 训练不稳定或效果不明显 | 指令部分也参与了loss计算 | 确认labels中指令部分为-100 |

### 调参层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 学习率偏大 | 模型输出变得"发疯"或退化到无意义文本 | 学习率超出了微调的安全范围 | SFT学习率应在1e-6到5e-5之间，从低开始找最佳值 |

## 12. 学习总结

### 核心思想回顾
SFT 通过有监督方式教会预训练模型遵循指令。它不改变模型的底层架构，只是在预训练权重的起点上，用指令-回答数据调整模型，使其从"续写文本"转变为"回答问题"。SFT 是现代 LLM 后训练流程中不可或缺的第一步。

### 关键公式
$$\mathcal{L}_{\text{SFT}} = -\frac{1}{N} \sum_{i} \frac{1}{|y_i|} \sum_{t} m_t \cdot \log p_\theta(y_{i,t} | x_i, y_{i,<t})$$

其中 $m_t$ 是 mask，仅在回答部分为1。

### 与前序/相关算法的联系
- 建立在预训练的基座模型之上
- SFT 为 RLHF/DPO 提供初始策略
- 推理模型的 CoT SFT 是强化学习推理训练的前置步骤

### 后续学习方向
- RLHF / DPO：进一步让模型对齐人类偏好
- 推理 SFT：用含推理链的数据训练推理模型
- 数据合成：用强模型自动生成高质量 SFT 数据

## 13. 练习题与思考题

### 基础题

**题1**：SFT 和预训练在训练目标上有什么相同和不同之处？

**参考答案**：
相同：都使用"下一个 token 预测"（next-token prediction）作为训练目标，都使用交叉熵损失函数。
不同：(1) SFT 只对回答部分计算损失，忽略指令部分的 loss；(2) 预训练需要海量数据（TB级），SFT 只需几千到几万条数据；(3) 预训练使用通用文本，SFT 使用精心标注的指令-回答对。

**题2**：为什么 SFT 不能替代 RLHF？两者各解决什么问题？

**参考答案**：
SFT 是模仿学习（学习"怎么做"），RLHF 是偏好优化（学习"什么是好"）。SFT 让模型学会遵循指令，但无法告诉模型在所有可能的回答中哪一个更受人类偏好。RLHF 通过人类偏好比较信号，让模型不仅能回答问题，还能用更受欢迎的方式回答。

### 进阶题

**题3**：训练推理模型时，SFT 阶段的数据格式与通用对话 SFT 有什么区别？

**参考答案**：
推理模型的 SFT 数据通常包含两个显式分块：
```
<|im_start|>assistant
thinking
详细的推理链……
response
最终答案
<|im_end|>
```
而通用对话 SFT 只需要回答内容即可。推理 SFT 的关键区别在于显式划分"思考过程"和"最终回答"，这教会模型在回答前先进行系统性推理。

### 开放思考题

**题4**：SFT 训练出的模型行为完全由训练数据决定。如果让你设计一个 SFT 数据集来训练一个"诚实的 AI 助手"（不会不懂装懂），你会如何设计数据的构成？

**参考答案**：
一个诚实的 AI 助手数据集应考虑：
1. **"我不知道"示例**（~10-15%）：包含模型不可能知道答案的问题（如"2050年股市会涨吗？"），标准回答为诚实地表示不知道
2. **"边界说明"示例**（~5-10%）：当回答基于截止日期的知识时，明确说明知识截至时间
3. **"拒绝不合理请求"示例**（~5%）：包含有伦理风险的请求及礼貌拒绝的回应
4. **"纠正错误前提"示例**（~5%）：当问题的前提是错误的时候，先纠正前提再回答
5. **正常问答示例**（~65-70%）：涵盖各类知识领域的准确回答

关键设计原则：诚实数据的占比必须足够高（至少15-20%），否则模型在实际使用中会倾向于"编造"而非坦承不知道。

## 14. 学习路径建议

### 前置算法
- LLM 预训练原理（下一个 token 预测）
- 梯度下降与损失函数基础

### 平行算法
- RLHF / DPO：与 SFT 互补的后训练方法
- 持续预训练：在领域数据上继续预训练而非SFT

### 进阶算法
- 推理模型 SFT（带 CoT 的指令微调）
- 数据合成与蒸馏（用强模型生成SFT数据）
- 多模态指令微调

### 推荐资源
1. **论文**：Wei et al., "Finetuned Language Models Are Zero-Shot Learners" (FLAN, ICLR 2022) — 系统研究指令微调的奠基论文
2. **论文**：Ouyang et al., "Training language models to follow instructions with human feedback" (2022) — InstructGPT, SFT+RLHF的代表
3. **工具**：HuggingFace TRL 库文档 — 实际SFT训练的工程工具
