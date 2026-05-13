# RLHF 偏好调优 学习文档

> 通过人类偏好反馈信号，让LLM生成更符合人类期望的回答。

> 来源线索：本节内容根据原书中关于"preference tuning / RLHF"的相关章节整理、扩展与教学化改写。

## 1. 算法基础认知

### 一句话定义
RLHF (Reinforcement Learning from Human Feedback) 是利用人类对模型输出的偏好排序来优化LLM的方法。

### 直觉类比
老师不只告诉学生"这道题你答对了还是答错了"，而是看了学生几份不同版本的作业后说"这份比那份好"。学生通过老师的比较来学习什么是"好的表达"，而不仅仅是"对的答案"。

### 历史背景
RLHF 的现代形式主要由 OpenAI 在 InstructGPT (2022) 和 ChatGPT (2022) 中发扬光大。其理论基础可追溯到 Christiano 等人2017年的工作。目前RLHF 已成为主流LLM后训练流程的标配：预训练 → SFT → RLHF/DPO。

### 算法定位
- **类型**：强化学习 / 偏好优化 / 对齐 (Alignment)
- **性质**：需要人类标注数据，修改模型权重

### 前置知识
- 了解 SFT（指令微调）的概念
- 基础强化学习概念（奖励、策略、PPO）
- 了解 LLM 的自回归生成原理

## 2. 核心原理

### 核心思想
SFT 能让模型"做对事"，RLHF 能让模型"做对人喜欢的事"。RLHF 的核心是：先训练一个奖励模型 (Reward Model, RM) 来预测人类偏好，再用强化学习（典型是PPO）优化LLM使其最大化奖励模型给出的分数。这样做的好处是：不需要为每次模型更新都收集人类反馈，人类只需标注偏好数据训练RM即可。

### 工作流程
1. **SFT 初始化**：先有一个经过SFT的策略模型（能遵循指令的LLM）
2. **收集偏好数据**：对同一个 prompt，让策略模型生成多个回答，人类标注者排序哪个回答更好
3. **训练奖励模型 (RM)**：用偏好数据训练一个模型，输入 (prompt, response)，输出偏好分数
4. **PPO 微调**：用RM作为奖励信号，使用PPO算法优化LLM，同时加入KL惩罚防止偏离SFT模型太远
5. **迭代**：可选地重复步骤2-4来持续改进

### 关键概念解释
- **奖励模型 (RM)**：一个经过训练来预测人类偏好的模型，输入prompt+response，输出一个标量分数
- **KL惩罚**：在PPO目标中加入KL散度项，限制新策略与SFT模型（参考策略）的偏离程度，防止模型在追求高奖励时失去语言流畅性
- **PPO (Proximal Policy Optimization)**：一种流行的强化学习算法，通过截断目标函数稳定训练
- **DPO (Direct Preference Optimization)**：RLHF的简化替代方案，直接在偏好数据上优化语言模型，不需要显式训练奖励模型和PPO

### 直观解释
```
SFT后的模型                   RLHF后的模型
问题: "解释量子计算"           问题: "解释量子计算"
回答: "量子计算使用量子比特     回答: "量子计算是一种利用量子力学
      实现并行计算。"               原理的新型计算范式。让我用一个
                                   比喻来解释……"
                              ← 更友好、更详细、更有教学感
```

## 3. 数学公式与推导

### 符号约定
| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 要优化的策略（LLM），参数为 $\theta$ |
| $\pi_{\text{ref}}$ | 参考策略（SFT模型），优化时的锚点 |
| $r_\phi(x, y)$ | 奖励模型，给 (prompt, response) 打分 |
| $x$ | 输入 prompt |
| $y$ | 生成的 response |
| $\beta$ | KL 惩罚系数 |

### 奖励模型训练

用 Bradley-Terry 偏好模型：

$$p(\text{response } y_w \succ y_l | x) = \frac{\exp(r_\phi(x, y_w))}{\exp(r_\phi(x, y_w)) + \exp(r_\phi(x, y_l))}$$

其中 $y_w$ 是人类更偏好的回答，$y_l$ 是较差的回答（w=win, l=lose）。

RM的损失函数为负对数似然：

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

这鼓励RM给更好的回答以更高的分数。

### PPO 优化目标

完整的 PPO 优化目标包含三部分：

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}\left[r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))\right]$$

其中 KL 散度衡量新策略与参考策略的差异：

$$\text{KL}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$

### 实际PPO的截断目标

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是策略比率，$\hat{A}_t$ 是优势估计，$\epsilon$ 是截断范围（通常0.2）。

### 总结：RLHF的三步数学

1. SFT得到 $\pi_{\text{ref}}$
2. 训练RM：最小化 $-\log \sigma(r(y_w) - r(y_l))$
3. PPO优化：最大化 $\mathbb{E}[r(y) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})]$

## 4. 训练过程讲解

### 数据预处理
1. **收集偏好数据**：对每个prompt，生成2-4个候选回答(K=2最常见)
2. **人类标注**：标注者排序或二选一（更偏好哪个）
3. **数据量和质量**：通常需要几千到几万对偏好比较数据

### 奖励模型训练
- 从SFT模型初始化RM（用SFT模型的权重，替换最后一层为标量输出）
- 在偏好数据上训练RM，通常1-3个epoch
- 验证RM的准确率（正确预测偏好的比例，好的RM能达到65-75%）

### PPO 训练
1. 从prompt数据集采样batch
2. 用当前策略$\pi_\theta$为每个prompt生成response
3. 用RM给每个(response, prompt)对打分
4. 计算PPO损失（含KL惩罚项），反向传播更新$\pi_\theta$

### 关键超参数
| 参数 | 作用 | 推荐范围 | 默认建议 |
|------|------|----------|----------|
| $\beta$ (KL系数) | 控制偏离SFT的程度 | 0.01 ~ 0.5 | 0.04 |
| $\epsilon$ (PPO clip) | PPO裁剪范围 | 0.1 ~ 0.3 | 0.2 |
| learning_rate (PPO) | PPO阶段学习率 | 1e-7 ~ 1e-5 | 1.41e-6 |
| batch_size (PPO) | PPO每步采样的prompt数 | 32 ~ 512 | 128 |
| RM 训练 epochs | 奖励模型训练轮数 | 1 ~ 3 | 2 |

## 5. 应用场景

### 典型应用
1. **通用AI助手**：ChatGPT、Claude等的对话质量优化。让回答更自然、更有帮助性、更无害。
2. **内容审核与安全**：通过偏好标注"安全的回答 > 不安全的回答"，降低模型生产有害内容的概率。
3. **风格定制**：想让模型回答更幽默/更正式/更简洁，在偏好标注中体现相应偏好即可。
4. **多语言对齐**：收集不同语言的偏好数据，让模型在不同文化背景下都能提供适当的回答。

### 适用数据特征
- 有明确的人类偏好标准（如帮助性、无害性、诚实性）
- SFT模型已具备基本的指令遵循能力
- 有资源收集偏好标注数据

### 不适用场景
- 只有正确/错误标注没有偏好排序的场景
- 基座模型质量太差（RLHF无法修复根本缺陷）
- 推理类任务中的自动化奖励（此时用纯RL即可，不需要HF偏好数据）

## 6. 优缺点分析

### 优点
| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 对齐人类偏好 | 生成的回答更符合人类审美和期望 | 有足够的高质量偏好数据 |
| 不依赖硬规则 | 不需要为"什么是好回答"定义具体规则，通过数据隐式学习 | 偏好标注一致 |
| 保护语言质量 | KL惩罚确保模型不会因追求高奖励而退化 | $\beta$参数设置合理 |
| 灵活可扩展 | 偏好标准可以通过调整标注指南随时更改 | 有新偏好的标注数据 |

### 缺点
| 缺点 | 说明 | 缓解思路 |
|------|------|----------|
| 标注成本高 | 需要大量人工比较判断 | 用AI辅助标注或使用DPO |
| 标注者偏差 | 标注者的偏好不一定代表所有用户 | 多元化标注者池、Red Teaming |
| RM被"游戏化"可能 | PPO可能找出RM的漏洞来获得高分而非真正变好 | 持续更新RM、RM与PPO迭代训练 |
| 训练不稳定 | PPO在大模型上训练极易崩溃 | 使用DPO替代、仔细监控KL值 |
| 难以调试 | 不清楚哪个环节导致了问题（数据/RM/PPO） | 分阶段评估、使用验证集 |

### 与同类方法对比
| 方法 | 需要RM | 训练复杂度 | 数据需求 | 稳定性 |
|------|--------|------------|----------|--------|
| RLHF (PPO) | 是 | 高 | 偏好对 | 中-低 |
| DPO | 否 | 中 | 偏好对 | 高 |
| KTO | 否 | 中 | 单样本偏好 | 中 |
| SFT only | 否 | 低 | 指令回答对 | 高 |

## 7. 调库实现

```python
"""
RLHF 偏好调优的调库实现
使用 TRL 库: SFT → 奖励模型训练 → PPO 优化
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    SFTTrainer, RewardTrainer, PPOTrainer,
    PPOConfig, RewardConfig,
)
from datasets import Dataset


# ===== 第一步: SFT 配置 =====
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# ===== 第二步: 训练奖励模型 =====
def train_reward_model(preference_data: Dataset):
    """
    在偏好比较数据上训练奖励模型

    数据格式: {"prompt": "...", "chosen": "...", "rejected": "..."}
    chosen = 人类更偏好的回答
    rejected = 较差的回答
    """
    # 从SFT模型初始化奖励模型
    rm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )

    # TRL的RewardTrainer自动处理
    # 格式化和损失计算（Bradley-Terry模型）
    reward_config = RewardConfig(
        per_device_train_batch_size=4,
        num_train_epochs=2,
        learning_rate=1.41e-5,
        bf16=True,
        output_dir="./reward_model",
    )

    # 为奖励模型准备数据格式
    def preprocess_reward(example):
        """将偏好数据格式化为奖励模型需要的格式"""
        chosen_text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['chosen']}<|im_end|>"
        rejected_text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['rejected']}<|im_end|>"
        return {"chosen": chosen_text, "rejected": rejected_text}

    preference_data = preference_data.map(preprocess_reward)

    trainer = RewardTrainer(
        model=rm_model,
        args=reward_config,
        tokenizer=tokenizer,
        train_dataset=preference_data,
    )
    trainer.train()
    trainer.save_model("./reward_model/final")

    return rm_model


# ===== 第三步: PPO 优化 =====
def train_ppo(
    sft_model: AutoModelForCausalLM,
    reward_model: AutoModelForCausalLM,
    prompts_dataset: Dataset,
):
    """
    使用PPO + KL惩罚微调SFT模型

    核心公式: max E[r(y) - β * KL(π_θ || π_ref)]
    """
    # PPO配置
    ppo_config = PPOConfig(
        batch_size=128,              # 每次采样的prompt数量
        forward_batch_size=16,        # 每次前向传播的大小
        ppo_epochs=4,                 # PPO更新的epoch数
        learning_rate=1.41e-6,        # PPO学习率，注意比SFT小很多
        kl_penalty="kl",              # KL惩罚类型
        init_kl_coef=0.04,            # β参数初值
        target_kl=6.0,                # 目标KL，超过会自适应调整β
        cliprange=0.2,                # PPO裁剪范围ε
        cliprange_value=0.2,          # 价值函数裁剪范围
    )

    # 创建PPO trainer
    # ref_model = sft_model的深拷贝，作为参考策略π_ref
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=sft_model,                      # π_θ (会被优化)
        ref_model=AutoModelForCausalLM.from_pretrained(
            "./sft_model"
        ),                                      # π_ref (冻结)
        tokenizer=tokenizer,
        dataset=prompts_dataset,
    )

    # PPO训练循环
    # 由于训练循环涉及模型生成+RM打分，这里展示核心步骤
    print("PPO Trainer 已配置。实际训练需要在有GPU的环境中运行。")
    print(f"  KL 惩罚系数 β = {ppo_config.init_kl_coef}")
    print(f"  PPO 裁剪范围 ε = {ppo_config.cliprange}")

    return ppo_trainer


# ===== 模拟测试: 偏好数据示例 =====
sample_preference_data = [
    {
        "prompt": "解释什么是机器学习",
        "chosen": "机器学习是一种让计算机从数据中学习规律的技术。想象教小孩认猫——你不需要写程序告诉它猫有尖耳朵、胡子、四条腿，你只需要给它看够多猫的图片，它自己就能总结出猫的特征。",
        "rejected": "机器学习是人工智能的一个子领域，涉及算法和统计模型的使用。",
    },
    {
        "prompt": "推荐一本好书",
        "chosen": "我推荐《人类简史》。这本书讲述了从认知革命到科学革命的人类发展历程，文笔生动，即使不是历史爱好者也会被吸引。",
        "rejected": "《深度学习》by Ian Goodfellow是一本好书。",
    },
]
print("偏好数据示例已创建。在训练中:
  chosen = 更详细、更有帮助、更友好的回答
  rejected = 简短、冷淡或无帮助的回答")
```

## 8. 手工代码实现

```python
"""
RLHF 核心机制的手工实现
展示 Bradley-Terry 偏好模型 + PPO + KL 惩罚的核心逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RewardModelTrainer:
    """
    手工实现奖励模型训练
    基于 Bradley-Terry 偏好模型
    """

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def compute_reward_loss(self, chosen_scores, rejected_scores):
        """
        计算 Bradley-Terry 损失

        L = -log σ(r(x, y_chosen) - r(x, y_rejected))

        核心直觉：
        - 如果 chosen 的得分远高于 rejected → 误差接近0（已经很好地区分了）
        - 如果两者的差别很小 → 误差较大，需要继续训练
        - 如果 rejected 的得分反而更高 → 误差很大，需要大幅调整
        """
        diff = chosen_scores - rejected_scores
        # σ是sigmoid函数：σ(diff) = 1/(1+exp(-diff))
        loss = -torch.mean(F.logsigmoid(diff))
        return loss

    def train_step(self, chosen_texts, rejected_texts, optimizer):
        """
        单步训练奖励模型
        """
        self.model.train()

        # 计算 chosen（好回答）的得分
        chosen_scores = self.model(chosen_texts)

        # 计算 rejected（差回答）的得分
        rejected_scores = self.model(rejected_texts)

        loss = self.compute_reward_loss(chosen_scores, rejected_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


class PPOStepManual:
    """
    手工实现 PPO + KL 惩罚的核心步骤

    展示策略梯度 + PPO clip + KL散度惩罚的计算逻辑
    """

    @staticmethod
    def compute_ppo_loss_with_kl(
        log_probs: torch.Tensor,            # 当前策略 π_θ 的 log 概率
        old_log_probs: torch.Tensor,        # 旧策略 π_old 的 log 概率
        ref_log_probs: torch.Tensor,        # 参考策略 π_ref 的 log 概率
        advantages: torch.Tensor,           # 优势 A_t
        kl_coef: float = 0.04,              # β 参数
        clip_epsilon: float = 0.2,          # ε 参数
    ) -> torch.Tensor:
        """
        PPO损失 + KL惩罚 = -L_PPO + β * KL

        详细步骤：
        1. 计算策略比率 r_t = exp(log_π_θ - log_π_old)
        2. 计算截断目标 min(r*A, clip(r)*A)
        3. 计算KL散度 λ = log_π_θ - log_π_ref
        4. 总损失 = -(PPO目标 - β * KL)
        """
        # 第一步：计算策略比率
        # r_t = π_θ(a|s) / π_old(a|s) = exp(log π_θ - log π_old)
        ratio = torch.exp(log_probs - old_log_probs)

        # 第二步：PPO 截断目标
        # 无截断的目标
        surr1 = ratio * advantages
        # 有截断的目标：ratio被裁剪到[1-ε, 1+ε]
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        # 取两者中较差的（更保守的估计）——这是PPO保守更新的核心
        ppo_objective = torch.min(surr1, surr2)

        # 第三步：KL 散度
        # KL(π_θ || π_ref) = E[log π_θ - log π_ref]
        # 这里使用一个简单估计：log概率差
        kl_divergence = log_probs - ref_log_probs

        # 第四步：总损失 = -(PPO目标 - β * KL)
        # 加上负号是因为我们要最小化损失，但PPO目标是最大化
        total_loss = -(ppo_objective - kl_coef * kl_divergence).mean()

        return total_loss, {
            "ppo_objective": ppo_objective.mean().item(),
            "kl_divergence": kl_divergence.mean().item(),
            "ratio_mean": ratio.mean().item(),
        }

    @staticmethod
    def compute_advantages(
        rewards: torch.Tensor,
        gamma: float = 1.0,
        lam: float = 0.95,
    ) -> torch.Tensor:
        """
        GAE (Generalized Advantage Estimation) 的实现

        这展示了PPO所需的"优势函数 A_t"如何计算的简化版。
        在一个token的自回归生成中，
        每个生成token可以视为一个"动作"，
        最终RM给出的分数是"回报"。
        """
        # 简化版：优势 = 奖励（每个token获得相同回报的简化处理）
        # 实际中需要完整的GAE计算
        advantages = rewards - rewards.mean()
        advantages = advantages / (advantages.std() + 1e-8)  # 标准化
        return advantages


# ===== 演示: PPO+KL的数值示例 =====
def demo_ppo_kl():
    """演示PPO+KL在不同情况下的表现"""

    ppo_manual = PPOStepManual()

    # 场景1：策略变化不大
    print("场景1: 策略变化不大（ratio接近1）")
    loss1, info1 = ppo_manual.compute_ppo_loss_with_kl(
        log_probs=torch.tensor([-0.5, -0.6, -0.55]),
        old_log_probs=torch.tensor([-0.48, -0.58, -0.53]),
        ref_log_probs=torch.tensor([-0.5, -0.6, -0.55]),
        advantages=torch.tensor([1.0, 0.5, -0.3]),
        kl_coef=0.04,
        clip_epsilon=0.2,
    )
    print(f"  损失: {loss1:.4f}")
    print(f"  PPO目标: {info1['ppo_objective']:.4f}")
    print(f"  KL散度: {info1['kl_divergence']:.4f}")
    print(f"  策略比率: {info1['ratio_mean']:.4f} (接近1=没有大变)")

    # 场景2：策略变化很大（KL惩罚起作用）
    print("\n场景2: 策略变化很大（ratio远离1，KL惩罚阻止过大更新）")
    loss2, info2 = ppo_manual.compute_ppo_loss_with_kl(
        log_probs=torch.tensor([-0.1, -0.2, -0.15]),
        old_log_probs=torch.tensor([-1.0, -1.0, -1.0]),
        ref_log_probs=torch.tensor([-0.5, -0.5, -0.5]),
        advantages=torch.tensor([1.0, 0.5, -0.3]),
        kl_coef=0.04,
        clip_epsilon=0.2,
    )
    print(f"  损失: {loss2:.4f}")
    print(f"  PPO目标: {info2['ppo_objective']:.4f}")
    print(f"  KL散度: {info2['kl_divergence']:.4f} (KL增大=惩罚生效)")
    print(f"  策略比率: {info2['ratio_mean']:.4f} (远大于1=PPO clip起作用)")

    print("""
    结论：
    - KL惩罚的存在防止了模型跑偏（在场景2中loss增加）
    - PPO clip 防止单步更新过大导致训练不稳定
    - 这种"限制偏离"的设计是RLHF成功的关键技术原因
    """)


if __name__ == "__main__":
    demo_ppo_kl()
```

## 9. 可视化与结果理解

```python
"""
RLHF 各阶段的训练过程可视化
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ---- 图1: 奖励模型准确率 ----
ax1 = axes[0]
epochs = np.arange(1, 6)
rm_accuracy = [0.62, 0.68, 0.71, 0.72, 0.71]
ax1.plot(epochs, rm_accuracy, 'b-o', linewidth=2, markersize=8)
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("准确率", fontsize=11)
ax1.set_title("奖励模型训练\n(Bradley-Terry偏好预测准确率)", fontsize=11, fontweight="bold")
ax1.set_ylim(0.5, 0.8)
ax1.grid(alpha=0.3)
ax1.annotate("RM准确率\n不再增长",
             xy=(4, 0.71), fontsize=9, color="red")

# ---- 图2: PPO训练中各项指标 ----
ax2 = axes[1]
steps = np.arange(0, 100)
# 模拟PPO训练中的指标
reward_mean = 1.5 + 2.0 * (1 - np.exp(-steps/20))
kl_div = 8 * np.exp(-steps/15)
kl_target = 6.0

ax2.plot(steps, reward_mean, 'g-', linewidth=2, label="平均奖励 ↑")
ax2.plot(steps, kl_div, 'orange', linewidth=2, label="KL散度 ↓")
ax2.axhline(y=kl_target, color='red', linestyle='--', alpha=0.7, label="目标KL")
ax2.set_xlabel("PPO Step", fontsize=11)
ax2.set_ylabel("值", fontsize=11)
ax2.set_title("PPO训练过程中的指标变化", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ---- 图3: KL系数 β 的影响 ----
ax3 = axes[2]
beta_values = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
reward = [4.5, 4.2, 3.8, 3.2, 2.5, 1.8]
kl_from_ref = [12.0, 8.0, 5.5, 3.0, 1.5, 0.8]
quality_score = [3.0, 3.5, 4.0, 3.8, 3.3, 2.8]

ax3.plot(beta_values, reward, 'g-o', linewidth=2, label="Reward")
ax3.plot(beta_values, kl_from_ref, 'orange-s', linewidth=2, label="KL from ref")
ax3.plot(beta_values, quality_score, 'b-D', linewidth=2, markersize=8, label="人类评测质量")
ax3.axvline(x=0.04, color='red', linestyle='--', alpha=0.7)
ax3.set_xlabel("KL系数 β", fontsize=11)
ax3.set_ylabel("指标值", fontsize=11)
ax3.set_title("KL系数 β 的影响\nβ=0.04 达到最佳平衡", fontsize=11, fontweight="bold")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("""
图表解读：
左图：奖励模型在偏好数据上训练，用作PPO的"裁判"
中图：PPO训练中奖励逐渐上升 + KL逐渐下降收敛到目标范围内
右图：β太小→reward高但KL大（过拟合于RM偏好）
      β太大→KL小但reward低（没有有效学习）
      β=0.04是常见的最佳平衡点
""")
```

## 10. 模型评估

```python
"""
RLHF 模型的评估策略
"""

import numpy as np
from collections import defaultdict


class RLHFEvaluator:
    """RLHF模型的专用评估器"""

    @staticmethod
    def evaluate_helpfulness_harmlessness(predictions, references):
        """评估帮助性与无害性（需要人评或LLM-as-judge）"""
        # 这是RLHF评估中最重要的两个维度
        return {
            "note": "帮助性与无害性通常需要人工评估或LLM-as-judge自动评估"
        }

    @staticmethod
    def win_rate_against_baseline(
        model_responses: list,
        baseline_responses: list,
        prompts: list,
        judge_fn=None,
    ) -> dict:
        """
        计算 vs baseline 的胜率

        这是RLHF实际使用中最重要的评估指标
        """
        wins = 0
        ties = 0
        losses = 0
        results = []

        for i, (model_r, baseline_r, prompt) in enumerate(
            zip(model_responses, baseline_responses, prompts)
        ):
            if judge_fn:
                # 使用judge函数判断胜负
                result = judge_fn(prompt, model_r, baseline_r)
            else:
                # 对前2个示例默认model回答更好
                result = "model_win" if i < 2 else "baseline_win"

            if result == "model_win":
                wins += 1
            elif result == "tie":
                ties += 1
            else:
                losses += 1
            results.append(result)

        total = len(model_responses)
        return {
            "win_rate": wins / total,
            "tie_rate": ties / total,
            "loss_rate": losses / total,
            "total": total,
        }

    @staticmethod
    def response_diversity_score(responses: list) -> dict:
        """
        评估回答的多样性
        RLHF容易导致模型"模式坍缩"——生成大量相似回答
        """
        from collections import Counter

        # 简化：用回答长度分布的熵衡量多样性
        lengths = [len(r) for r in responses]
        len_mean = np.mean(lengths)
        len_std = np.std(lengths)

        # 统计起始词的多样性
        first_words = []
        for r in responses:
            words = r.strip().split()
            if words:
                first_words.append(words[0].lower())

        unique_start_ratio = len(set(first_words)) / max(1, len(first_words))

        return {
            "avg_length": len_mean,
            "length_std": len_std,
            "unique_start_ratio": unique_start_ratio,
            "note": "过于低的多样性可能是RLHF模式坍缩的信号",
        }


# ===== 使用示例 =====
model_responses = [
    "机器学习很有趣的！让我来详细解释……它是一种让计算机从数据中学习的技术。想象你教小孩……",
    "这是一个很好的问题。机器学习本质上是……",
    "简单来说，机器学习就是让计算机学会从数据中找规律。",
]
baseline_sft_responses = [
    "机器学习是AI的子领域。",
    "机器学习使用算法从数据中学习。",
    "机器学习涉及统计学和计算机科学。",
]

evaluator = RLHFEvaluator()

# 胜率评估
win_result = evaluator.win_rate_against_baseline(
    model_responses, baseline_sft_responses,
    prompts=["解释ML", "再解释ML", "ML简述"],
)
print(f"RLHF模型 vs SFT模型 胜率: {win_result['win_rate']:.0%}")

# 多样性评估
diversity = evaluator.response_diversity_score(model_responses)
print(f"回答多样性: 起始词独立率 {diversity['unique_start_ratio']:.0%}")
```

## 11. 常见问题与易错点

### 数据层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 偏好标注不一致 | RM训练准确率偏低(<65%) | 不同标注者偏好不同 | 优化标注指南、多标注者+共识机制 |
| 偏好数据太单调 | 优化后模型回答千篇一律 | 标注数据中"好回答"模式太少 | 增加训练数据的风格多样性 |

### 模型层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| RM被破解(Reward Hacking) | PPO后RM分数极高但实际回答质量下降 | 模型找到了RM的"漏洞"来获得高分 | 持续更新RM、降低KL系数、monitor |
| KL值过高 | 模型基本没变化 | β太大或PPO步数太少 | 降低β到0.02-0.04 |
| 语言质量退化 | PPO后模型生成的文本不通顺 | KL惩罚不够或RM偏好评了不自然的回答 | 提高β、检查RM训练数据 |

### 调参层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| PPO崩溃 | 损失突然爆炸成NaN | 学习率过大或梯度裁剪不够 | 降低lr到1e-7、增大梯度裁剪、检查数据 |

## 12. 学习总结

### 核心思想回顾
RLHF 让人类偏好变成可优化的信号：先训练RM学会预测偏好，再用PPO（带KL约束）优化LLM。三步走——SFT打底、RM当裁判、PPO当教练。

### 关键公式
1. RM损失: $\mathcal{L}=-\log\sigma(r(y_w)-r(y_l))$
2. PPO+KL目标: $\max \mathbb{E}[r(y)-\beta \cdot \text{KL}(\pi_\theta\|\pi_{\text{ref}})]$

### 与前序/相关算法的联系
- 依赖SFT提供初始策略（$\pi_{\text{ref}}$）
- DPO进一步简化了RLHF（不需要RM+PPO，直接在偏好数据上优化）
- 推理RL使用类似机制但用自动化奖励替代人类偏好

### 后续学习方向
- DPO / KTO等替代RLHF的新方法
- LLM-as-Judge自动评估提升RLHF效率
- 推理模型的偏好优化 v.s. 通用对话模型的偏好优化

## 13. 练习题与思考题

### 基础题

**题1**：解释为什么RLHF需要KL惩罚项？如果没有KL惩罚会怎样？

**参考答案**：
KL惩罚限制优化后的策略不能偏离参考策略（SFT模型）太远。原因是：(1) RM只在部分偏好数据上训练，覆盖不全面，模型可能通过"耍小聪明"获取高reward但生成无意义文本；(2) 没有KL约束的话PPO可能过度优化以至于语言流利度和基本能力退化。实际上没有KL的PPO ≈ 让一个只会看单选题的"裁判"来评判所有回答——很容易被欺骗。

**题2**：DPO和RLHF的根本区别是什么？DPO能否完全替代RLHF？

**参考答案**：
根本区别：DPO直接在偏好数据上优化LM，跳过训练RM和PPO的阶段。数学上DPO将RLHF的优化问题重新表述为可在偏好数据上直接优化的形式。DPO在大多数场景可替代RLHF，优点是更简单稳定；缺点是需要高质量的偏好数据，且不能像RLHF那样在线的迭代改进（online exploration）。

### 进阶题

**题3**：一个RM在训练集上准确率75%，但在实际PPO优化时表现不佳（reward上升但人类评测反而下降）。分析可能的原因。

**参考答案**：
这被称为"Reward Overoptimization"现象。可能原因：
1. **分布偏移**：RM训练数据是SFT模型生成的，但PPO过程中模型不断变化，生成内容超出RM训练分布
2. **RM过拟合于表面特征**：RM学会了"回答长=好"等浅层特征而非真正的质量判断
3. **偏好数据覆盖不全**：RM在某些领域缺乏判断力，但模型恰好学会了在这些领域"钻空子"

解决方案：阶段性重新标注PPO模型生成的样本训练新RM、使用集成RM。

### 开放思考题

**题4**：RLHF是对齐AI的安全技术。如果某些人类偏好本身是有害的（如故意给错误建议的偏好），RLHF如何避免放大这些偏好吗？

**参考答案**：
RLHF本身是中性工具，其对齐方向完全取决于偏好数据的设计。避免放大有害偏好需要：
1. 在偏好标注指南中明确规定"有帮助性+无害性+诚实性"三个维度的权衡
2. 构建宪法AI（Constitutional AI）：用书面原则指导偏好判断
3. Red Teaming测试暴露出有害偏好后被进一步修正
4. 使用多群体标注者的共识而非单一决策者
本质上，这是一个数据治理和价值观选择问题，而非纯技术问题。

## 14. 学习路径建议

### 前置算法
- SFT（指令微调）——RLHF的起点
- PPO（强化学习基础算法）
- 交叉熵与KL散度的数学基础

### 平行算法
- DPO (Direct Preference Optimization)
- Constitutional AI
- Red Teaming 方法

### 进阶算法
- Online DPO / Iterative DPO
- Group Relative Policy Optimization (GRPO) ——用于推理模型
- Multi-objective RLHF（多目标优化）

### 推荐资源
1. **论文**：Ouyang et al., "Training language models to follow instructions with human feedback" (2022) — InstructGPT/ChatGPT的RLHF实现
2. **论文**：Rafailov et al., "Direct Preference Optimization" (NeurIPS 2023) — DPO论文，RLHF的简化替代
3. **教程**：HuggingFace TRL 文档 — RLHF的工程实践
