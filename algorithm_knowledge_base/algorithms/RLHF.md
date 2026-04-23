# RLHF（人类反馈强化学习）学习文档

> 通过人类反馈微调大语言模型，使其生成更安全、更有帮助的回答

---

## 1. 算法基础认知

**一句话定义**：RLHF（Reinforcement Learning from Human Feedback，人类反馈强化学习）是一种使用人类反馈作为奖励信号来微调预训练语言模型的技术，让模型的输出更符合人类偏好和价值观。

**直觉类比**：RLHF就像一位老师在教导学生——学生（模型）先通过大量阅读学会了基本知识（预训练），然后老师对学生生成的回答给出反馈（这个回答好，那个回答不好），学生根据反馈不断改进，逐渐学会生成更符合人类期望的回答。

**历史背景**：2017年，OpenAI和DeepMind首次将RLHF应用于语言模型训练。2022年，InstructGPT和ChatGPT使用RLHF显著提升了模型的有用性和安全性，RLHF因此成为大模型对齐（Alignment）的主流技术。

**算法定位**：
- 类型：NLP → 模型对齐 → 强化学习
- 输出：微调后的语言模型
- 模型类型：PPO + 奖励模型

**前置知识**：
- [必备]：强化学习基础（PPO、Policy Gradient）
- [必备]：语言模型基础（Transformer、GPT）
- [扩展]：奖励模型、对比学习

---

## 2. 核心原理

### 2.1 核心思想

RLHF的核心创新是**用人类反馈作为奖励信号，而非简单的损失函数**：
1. 训练奖励模型（Reward Model）：学习人类偏好
2. 使用PPO微调语言模型：基于奖励模型优化

核心思想可以概括为：**让模型学习"什么是好的回答"，而不仅是"什么是正确的回答"**。

### 2.2 工作流程

```
预训练模型 → 有监督微调(SFT) → 训练奖励模型(RM) → PPO强化学习 → 对齐后的模型
```

### 2.3 关键概念

- **Reward Model（奖励模型）**：学习人类偏好的二分类模型
- **PPO（Proximal Policy Optimization）**：稳定策略更新的强化学习算法
- **SFT（有监督微调）**：使用人类标注数据微调
- **KL散度约束**：限制新策略与原始策略的差异

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $\pi_{\theta}$ | 策略模型 |
| $r_{\phi}$ | 奖励模型 |
| $D$ | 人类反馈数据 |
| $\beta$ | KL惩罚系数 |

### 3.2 奖励模型目标

$$\max_{\phi} \mathbb{E}_{(x, y_1, y_2) \sim D}[\log \sigma(r_{\phi}(x, y_1) - r_{\phi}(x, y_2))]$$

其中人类偏好 $y_1 > y_2$。

### 3.3 PPO目标

$$L^{PPO}(\theta) = \mathbb{E}_t [\min(r_t(\theta) \hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)] - \beta D_{KL}(\pi_{\theta} || \pi_{ref})$$

---

## 4. 训练过程

### 4.1 数据准备

```python
# 人类反馈数据格式
preference_data = [
    {
        "prompt": "写一首关于春天的诗",
        "response_a": "春风拂面...",
        "response_b": "春天是万物复苏的季节...",
        "label": "a更好"  # 人类选择
    },
]
```

### 4.2 奖励模型训练

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.scorer = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask)
        # 取最后一个token的表示
        hidden = outputs.last_hidden_state[:, -1]
        return self.scorer(hidden)
```

### 4.3 PPO训练

```python
def train_ppo(policy_model, reward_model, prompt_data):
    # 1. 获取响应
    responses = policy_model.generate(prompt_data)
    
    # 2. 计算奖励
    rewards = reward_model(prompt_data, responses)
    
    # 3. PPO更新
    loss = ppo_loss(policy_model, responses, rewards)
    loss.backward()
    optimizer.step()
```

---

## 5. 应用场景

### 5.1 典型应用

**ChatGPT/InstructGPT**：让模型遵循指令

**对话系统**：生成更有帮助的回答

**代码生成**：生成更安全、可读的代码

**内容过滤**：减少有害内容输出

### 5.2 适用场景

- 需要模型输出符合人类偏好
- 安全性要求高的应用
- 需要持续改进的AI系统

---

## 6. 优缺点

### 6.1 优点

1. **对齐人类价值观**：输出更安全、有帮助
2. **可扩展**：只需人类反馈数据
3. **效果显著**：ChatGPT成功的关键

### 6.2 缺点

1. **训练复杂**：多阶段训练流程
2. **数据成本**：需要大量人类标注
3. **奖励模型偏差**：可能引入新的偏见

---

## 7. 调库实现

### 7.1 完整代码

```python
"""
RLHF 简化实现（使用TRL库）
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, PPOTrainer, RewardTrainer
from trl.core import set_seed

# 1. 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. 有监督微调
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    max_seq_length=512,
)
trainer.train()

# 3. 训练奖励模型
reward_trainer = RewardTrainer(
    model=model,
    train_dataset=reward_dataset,
    tokenizer=tokenizer,
)
reward_trainer.train()

# 4. PPO微调
ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_model=reward_model,
    train_dataset=prompt_dataset,
)
ppo_trainer.train()
```

---

## 8. 手工实现

```python
"""
简化PPO实现
"""

import torch
import torch.nn.functional as F

class PPOTrainer:
    def __init__(self, model, lr=1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def compute_loss(self, prompts, responses, rewards, old_log_probs):
        # 生成新策略的概率
        outputs = self.model(prompts, responses)
        new_log_probs = outputs.logits
        
        # 策略比值
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO裁剪
        clipped_ratio = ratio.clamp(0.8, 1.2)
        
        # 优势估计
        advantage = rewards - rewards.mean()
        
        # 损失
        loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        
        return loss
    
    def step(self, batch):
        prompts, responses, rewards = batch
        
        loss = self.compute_loss(prompts, responses, rewards, None)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## 9. 可视化

```python
import matplotlib.pyplot as plt

def visualize_training():
    # RLHF训练曲线
    steps = list(range(100))
    rewards = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards, 'b-o', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Reward Score')
    plt.title('RLHF Training Progress')
    plt.grid(True)
    plt.savefig('rlhf_training.png')
    plt.show()
```

---

## 10. 评估

### 10.1 指标

| 指标 | 含义 |
|------|------|
| Win Rate | 人类选择哪个回答更好 |
| 安全性评分 | 有害内容比例 |
| 帮助性评分 | 任务完成度 |

### 10.2 代码

```python
def evaluate_rlhf(model, test_prompts, human_preferences):
    wins = 0
    for prompt, prefs in zip(test_prompts, human_preferences):
        response = model.generate(prompt)
        if response == prefs:
            wins += 1
    
    return wins / len(test_prompts)
```

---

## 11. 常见问题

### 11.1 问题

**奖励模型过拟合**：使用更多样本来解决

**PPO训练不稳定**：调整学习率和裁剪参数

### 11.2 解决

```python
# 减小学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

# 调整裁剪范围
clip_epsilon = 0.1  # 从0.2减小
```

---

## 12. 学习总结

### 12.1 核心

✓ 人类反馈信号
✓ 奖励模型+PPO
✓ 对齐人类价值观

### 12.2 算法联系

- 前置：GPT、PPO
- 相关：InstructGPT、ChatGPT
- 进阶：Constitutional AI、DPO

---

## 13. 练习题

**问题**：RLHF和SFT的区别？

答案：SFT学习"正确回答"，RLHF学习"人类偏好"。

---

## 14. 学习路径

### 14.1 前置

- [ ] Transformer
- [ ] PPO

### 14.2 进阶

- [ ] ChatGPT原理
- [ ] 对齐研究

### 14.3 资源

1. 论文：InstructGPT (2022)
2. OpenAI Blog

---

## 附录

### A. 代码

见第7节。

### B. 参考文献

1. InstructGPT: Training language models to follow instructions with human feedback, 2022

---

**文档结束**