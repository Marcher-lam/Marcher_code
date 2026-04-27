# ChatGPT 与 RLHF 学习文档

> 通过人类反馈强化学习，将大语言模型对齐到人类偏好。

> 来源线索：本节内容根据原书中关于"ChatGPT 与 RLHF"的相关章节（第10章10.1-10.2节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** RLHF（Reinforcement Learning from Human Feedback）通过训练奖励模型学习人类偏好，再用 PPO 优化语言模型，使其输出更符合人类期望。

**直觉类比：** 想象一个实习生（语言模型）写报告。传统方法是给他大量优秀报告学习（预训练）。RLHF 则多了一步：请一位资深编辑（奖励模型）给实习生的报告打分，然后根据分数指导实习生改进写作（PPO 优化）。通过多轮"写稿→打分→改进"的循环，实习生的报告越来越符合编辑的期望。

**历史背景：** RLHF 的概念可追溯到 Christiano 等人 2017 年的 "Deep RL from Human Preferences"。OpenAI 在 2022 年将其应用于 InstructGPT，随后推出 ChatGPT，引爆了大语言模型热潮。

**算法定位：** 大语言模型对齐方法、基于人类反馈的强化学习。

**前置知识：** GPT、PPO、奖励模型、KL 散度、PyTorch。

---

## 2. 核心原理

### RLHF 三步流程

**第一步：监督微调（SFT）**
- 用人类编写的高质量对话数据微调预训练语言模型
- 目标：让模型学会以对话形式回答问题

**第二步：训练奖励模型（RM）**
- 让 SFT 模型对同一问题生成多个回答
- 人类标注员对这些回答排序
- 训练一个奖励模型学习人类偏好排序

**第三步：PPO 强化学习优化**
- 用奖励模型给生成回答打分作为奖励信号
- 用 PPO 算法优化语言模型
- 加入 KL 散度惩罚防止偏离 SFT 模型太远

### 关键概念

- **奖励模型（Reward Model）**：学习人类偏好的打分函数 $r_\phi(x, y)$
- **Bradley-Terry 模型**：将偏好排序转化为概率的数学框架
- **KL 惩罚**：$\beta \cdot KL(\pi_\theta \| \pi_{ref})$，防止模型"作弊"（生成奖励高但无意义的文本）
- **PPO**：限制策略更新幅度的强化学习算法

---

## 3. 数学公式与推导

### 奖励模型训练

给定人类偏好排序 $y_w \succ y_l$（$y_w$ 优于 $y_l$），奖励模型用 Bradley-Terry 模型建模：

$$P(y_w \succ y_l | x) = \frac{\exp(r_\phi(x, y_w))}{\exp(r_\phi(x, y_w)) + \exp(r_\phi(x, y_l))}$$

损失函数：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)}[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]$$

### PPO 优化目标

语言模型 $\pi_\theta$ 的优化目标：

$$\max_\theta \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[r_\phi(x, y) - \beta \cdot KL(\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x))\right]$$

- $r_\phi(x, y)$：奖励模型对回答的评分
- $\beta \cdot KL$：KL 散度惩罚，防止偏离参考模型 $\pi_{ref}$（即 SFT 模型）
- 实际使用 PPO 的裁剪目标进行优化

### KL 散度惩罚的作用

没有 KL 惩罚时，模型可能学到"作弊"策略：
- 生成格式完美但内容空洞的文本
- 利用奖励模型的漏洞获取高分
- 退化到只能回答特定类型的问题

---

## 4. 训练过程讲解

### 数据需求
- **SFT 数据**：人类编写的高质量对话（数万条）
- **偏好数据**：模型生成多个回答，人类排序（数十万条比较）
- **PPO 数据**：来自用户提示（无需人类标注，由奖励模型自动评分）

### 超参数表

| 超参数 | 推荐范围 | 默认 |
|--------|----------|------|
| PPO learning rate | 1e-6 ~ 5e-6 | 3e-6 |
| KL 惩罚系数 $\beta$ | 0.01 ~ 0.2 | 0.05 |
| PPO clip $\varepsilon$ | 0.1 ~ 0.2 | 0.2 |
| reward_scale | 0.1 ~ 1.0 | 0.5 |
| batch_size | 64 ~ 512 | 256 |

---

## 5. 应用场景

1. **对话系统**：ChatGPT、Claude 等对话 AI
2. **内容创作**：写作助手、代码生成
3. **知识问答**：精准回答专业问题
4. **指令遵循**：按照复杂指令完成任务

---

## 6. 优缺点分析

### 优点
1. **对齐人类偏好**：模型输出更安全、更有用
2. **可控性强**：通过奖励信号引导模型行为
3. **减少幻觉**：KL 惩罚约束模型不过度发挥

### 缺点
1. **人类标注成本高**：偏好排序需要大量人工
2. **奖励模型不完美**：可能存在"奖励黑客"问题
3. **训练复杂**：需要同时训练多个模型（SFT、RM、PPO）

---

## 7. 调库实现

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """简化的奖励模型"""
    def __init__(self, hidden_size=768):
        super().__init__()
        # 实际中共享语言模型的主体，只训练额外的价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, hidden_states):
        """输入: 语言模型最后一层隐状态, 输出: 标量奖励"""
        # 取最后一个 token 的隐状态
        last_hidden = hidden_states[:, -1, :]
        return self.value_head(last_hidden).squeeze(-1)

def compute_rm_loss(reward_model, chosen_hidden, rejected_hidden):
    """奖励模型的对比损失"""
    r_chosen = reward_model(chosen_hidden)
    r_rejected = reward_model(rejected_hidden)
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
    return loss

# 模拟训练
rm = RewardModel(hidden_size=256)
chosen = torch.randn(4, 20, 256)   # 好回答的隐状态
rejected = torch.randn(4, 20, 256) # 差回答的隐状态
loss = compute_rm_loss(rm, chosen, rejected)
print(f"RM 损失: {loss.item():.4f}")

# RLHF 的 PPO 训练循环（简化）
def rlhf_training_step(policy_model, ref_model, reward_model, tokenizer, prompts):
    """RLHF 的单步 PPO 训练"""
    # 1. 用策略模型生成回答
    responses = policy_model.generate(prompts)
    # 2. 计算奖励
    rewards = reward_model(responses)
    # 3. 计算 KL 惩罚
    kl_penalty = compute_kl(policy_model, ref_model, prompts, responses)
    # 4. PPO 更新
    adjusted_rewards = rewards - 0.05 * kl_penalty
    # (实际使用 PPO 裁剪目标更新)
    return adjusted_rewards.mean()

def compute_kl(policy, ref, prompts, responses):
    """计算策略模型与参考模型之间的 KL 散度"""
    # 简化实现
    log_p = policy.log_prob(responses | prompts)
    log_ref = ref.log_prob(responses | prompts)
    return (log_p - log_ref).mean()
```

---

## 8. 手工代码实现

```python
import numpy as np

class SimpleRLHF:
    """简化版 RLHF 流程（用于理解核心逻辑）"""
    def __init__(self, vocab_size=100, hidden_size=32):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 策略模型（简化为线性模型）
        self.policy_w = np.random.randn(hidden_size, vocab_size) * 0.01
        # 参考模型（SFT 后的模型）
        self.ref_w = self.policy_w.copy()
        # 奖励模型
        self.reward_w = np.random.randn(hidden_size, 1) * 0.01

    def softmax(self, x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def generate(self, state, length=10):
        """生成回答"""
        tokens = []
        for _ in range(length):
            probs = self.softmax(state @ self.policy_w)
            token = np.random.choice(self.vocab_size, p=probs.flatten())
            tokens.append(token)
        return tokens

    def compute_reward(self, state, response):
        """奖励模型评分"""
        return float(state.mean() @ self.reward_w)

    def compute_kl_penalty(self, state):
        """KL 散度惩罚"""
        p_policy = self.softmax(state @ self.policy_w)
        p_ref = self.softmax(state @ self.ref_w)
        kl = np.sum(p_policy * (np.log(p_policy + 1e-8) - np.log(p_ref + 1e-8)))
        return kl

    def rlhf_step(self, state, beta=0.05):
        """单步 RLHF 更新"""
        response = self.generate(state)
        reward = self.compute_reward(state, response)
        kl = self.compute_kl_penalty(state)
        adjusted_reward = reward - beta * kl
        # 简化的策略梯度更新
        grad = state.T @ np.random.randn(1, self.vocab_size) * 0.01 * adjusted_reward
        self.policy_w += np.clip(grad, -0.01, 0.01)  # 裁剪
        return adjusted_reward, kl

# 测试
rlhf = SimpleRLHF()
state = np.random.randn(1, 32)
for step in range(5):
    reward, kl = rlhf.rlhf_step(state)
    print(f"Step {step+1}: 调整后奖励={reward:.4f}, KL散度={kl:.4f}")
```

---

## 9-14. 评估/问题/总结/练习/路径

### 评估指标
- **人类评估**：有用性、安全性、真实性
- **奖励模型分数**：自动化评估生成质量
- **KL 散度**：监控模型偏离程度

### 常见问题
1. **奖励黑客（Reward Hacking）**：模型找到奖励模型的漏洞获得高分 → 增加 KL 惩罚、改进奖励模型
2. **对齐税（Alignment Tax）**：对齐后模型能力下降 → 平衡安全性和有用性
3. **标注者偏见**：不同标注者标准不一致 → 标注指南和多人交叉验证

### 练习题

**题1：** RLHF 中的 KL 散度惩罚为什么重要？

**参考答案：** 没有 KL 惩罚，模型可能学到"作弊"策略——生成奖励模型给高分但对人类无意义的文本。KL 惩罚确保优化后的模型不会偏离原始 SFT 模型太远，保持语言生成能力。

**题2（开放）：** RLHF 有哪些替代方案？各自的优缺点？

**参考答案思路：** DPO（Direct Preference Optimization）直接从偏好数据优化策略，无需训练奖励模型和 PPO 循环，更简单但灵活性较低。RLAIF（RL from AI Feedback）用 AI 替代人类标注偏好，降低成本但可能引入偏见。

### 学习路径
- 前置：GPT、PPO、强化学习基础
- 平行：DPO（RLHF 的替代方案）
- 进阶：Constitutional AI、RLAIF、多轮对话 RLHF
- 推荐：Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
