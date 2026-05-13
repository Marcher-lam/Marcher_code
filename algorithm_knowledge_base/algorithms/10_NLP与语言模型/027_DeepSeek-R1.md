# DeepSeek-R1 学习文档

## 1. 算法基础认知

DeepSeek-R1是中国AI公司DeepSeek开发的尖端大语言模型，于2025年发布，其核心创新在于将强化学习（Reinforcement Learning）深度融入推理能力的训练中，实现了媲美OpenAI o1模型的推理性能。DeepSeek-R1的突破性在于：它不依赖传统的监督微调，而是通过纯粹的强化学习训练，让模型自我涌现出强大的链式思考（Chain-of-Thought, CoT）能力。

### 1.1 为什么需要DeepSeek-R1？

传统大语言模型的训练范式存在根本性局限：
- **监督微调（SFT）依赖**：需要大量人工标注的高质量数据
- **推理能力不足**：缺乏深层思考和自我验证能力
- **奖励塑形困难**：难以精确设计奖励函数

DeepSeek-R1通过强化学习解决了这些问题：
- **自我涌现**：不依赖显式推理数据，学会自主思考
- **可验证目标**：通过可验证的奖励信号引导学习
- **更少依赖**：无需大量人工标注

### 1.2 DeepSeek-R1的核心创新

DeepSeek-R1系列包括多个版本：
- **DeepSeek-R1-Zero**：纯强化学习版本，无SFT
- **DeepSeek-R1**：加入冷启动数据
- **DeepSeek-R1-Distill**：蒸馏小模型

核心突破：
1. **Group Relative Policy Optimization (GRPO)**：新的强化学习算法
2. **CoT涌现**：自我生成思考过程
3. **自我验证**：学会检查和修正错误

### 1.3 与传统方法的对比

| 方法 | 数据需求 | 推理能力 | 训练方式 |
|------|---------|---------|----------|---------|
| SFT | 大量标注 | 一般 | 监督学习 |
| RLHF | 人类反馈 | 较好 | 强化学习 |
| DeepSeek-R1 | 极少 | 极强 | GRPO+RL |

## 2. 核心原理

### 2.1 GRPO算法原理

GRPO（Group Relative Policy Optimization）是DeepSeek-R1的核心训练算法，它是对PPO的改进，消除了对价值函数（Value Function）的依赖。

**传统PPO的问题**：
- 需要单独训练价值函数
- 训练不稳定
- 内存开销大

**GRPO的创新**：
- 在同一个batch内计算相对排名
- 无需显式价值函数
- 更稳定，更高效

### 2.2 GRPO的数学公式

给定一个问题 $q$ 和一组采样的响应 $\{o_1, o_2, ..., o_G\}$，其中 $G$ 是组大小。

**奖励计算**：
$$r_i = \text{Reward}(q, o_i)$$

**优势函数**：
$$A_i = \frac{r_i - \mu}{\sigma}$$

其中 $\mu$ 和 $\sigma$ 是组内奖励的均值和标准差。

**策略梯度**：
$$\nabla_\mathcal{J} = \mathbb{E}[A_i \cdot \nabla_\theta \log \pi_\theta(o_i|q)]$$

**重要性采样加权**：
$$\nabla_\mathcal{J} = \mathbb{E}\left[ \frac{w_i}{\sum w_j} \cdot A_i \cdot \nabla_\theta \log \pi_\theta(o_i|q) \right]$$

其中 $w_i = \pi_\theta(o_i|q) / \pi_{\theta_{old}}(o_i|q)$

### 2.3 链式思考（CoT）的涌现

DeepSeek-R1-Zero展现了令人惊讶的能力：在没有任何显式推理数据的情况下，通过强化学习自动涌现出链式思考能力。

涌现的思考模式：
1. **问题分析**：理解问题的本质
2. **计划制定**：制定解决步骤
3. **中间推理**：逐步推演
4. **自我验证**：检查中间结果
5. **结论总结**：总结最终答案

### 2.4 奖励函数设计

DeepSeek-R1使用三类奖励：

**准确率奖励**：
$$R_{accuracy} = \begin{cases} 1 & \text{答案正确} \\ 0 & \text{答案错误} \end{cases}$$

**格式奖励**：
$$R_{format} = \begin{cases} 1 & \text{格式正确} \\ 0 & \text{格式错误} \end{cases}$$

**思考过程奖励**：
$$R_{thinking} = \begin{cases} 0.5 & \text{包含<think>标签} \\ 0 & \text{无思考过程} \end{cases}$$

## 3. 数学公式与推导

### 3.1 GRPO完整算法

```
Algorithm: GRPO
---------------------------------
Input: Policy πθ, Questions Q, Group size G
Output: Updated policy πθ

For each batch:
    // 1. 采样多个响应
    o_i ~ πθ(q_i) for i in [1, G]
    
    // 2. 计算奖励
    r_i = Reward(q_i, o_i)
    
    // 3. 计算组内相对优势
    mean = (1/G) Σ r_i
    std = sqrt((1/G) Σ (r_i - mean)²)
    A_i = (r_i - mean) / std
    
    // 4. 计算策略梯度
    advantage = A_i - mean  // 使用相对排名
    ratio = πθ(o_i|q_i) / πθ_old(o_i|q_i)
    
    // 5. 策略更新
    loss = -min(ratio, clip(ratio, 1-ε, 1+ε)) * advantage
    
    // 6. 反向传播
    loss.backward()
    optimizer.step()
End For
```

### 3.2 冷启动数据

DeepSeek-R1在纯RL之前加入了少量冷启动数据（Cold Start Data），用于：
- 建立基础格式意识
- 防止训练崩溃
- 引导正确方向

冷启动数据格式：
```
<think>
让我们仔细分析这个问题...
首先，我们需要...
根据以上分析...
</think>
答案是：{最终答案}
```

### 3.3 拒绝采样优化

在RL训练后进行拒绝采样（Rejection Sampling）：

```python
# 拒绝采样伪代码
def rejection_sampling(model, prompts, threshold=0.8):
    accepted_outputs = []
    
    for prompt in prompts:
        outputs = model.generate(prompt, num_samples=10)
        
        for output in outputs:
            if reward(output) > threshold:
                accepted_outputs.append(output)
    
    return accepted_outputs
```

### 3.4 知识蒸馏

DeepSeek-R1-Distill使用知识蒸馏将能力迁移到小模型：

```python
# 蒸馏损失
L_distill = L_ce + λ * L_rms

# L_ce: 交叉熵损失
L_ce = -Σ y_teacher * log(y_student)

# L_rms: RMSE损失
L_rmse = (||y_teacher - y_student||²)
```

## 4. 训练过程讲解

### 4.1 DeepSeek-R1训练流程

```
Stage 1: 基础模型训练
  - 预训练：大规模语料
  - 后训练：少量SFT数据
  → DeepSeek-V2-Base

Stage 2: 强化学习（GRPO）
  - 采样响应
  - 计算奖励
  - 组内排名
  - 策略更新
  → DeepSeek-R1-Zero

Stage 3: 拒绝采样
  - RL生成多样本
  - 选择高质量样本
  - 精调模型
  → DeepSeek-R1

Stage 4: 蒸馏（可选）
  - 大模型指导小模型
  → DeepSeek-R1-Distill
```

### 4.2 训练配置

```python
# GRPO训练配置
grpo_config = {
    'group_size': 16,           # 每组采样数
    'learning_rate': 1e-6,       # 学习率
    'clip_epsilon': 0.2,          # PPO裁剪率
    'discount_factor': 1.0,        # 折扣因子
    'entropy_coef': 0.01,         # 熵系数
    'max_token_length': 8192,        # 最大生成长度
}

# 奖励函数配置
reward_config = {
    'accuracy_weight': 1.0,
    'format_weight': 0.1,
    'thinking_weight': 0.1,
}
```

### 4.3 训练监控

```python
# 训练监控指标
def compute_metrics(responses, rewards):
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'pass_rate': np.mean(np.array(rewards) > 0),
        'avg_length': np.mean([len(r) for r in responses]),
    }

# 训练循环
for step in range(num_steps):
    # 采样
    responses = sample_responses(policy, prompts, group_size=16)
    
    # 奖励
    rewards = compute_rewards(prompts, responses)
    
    # 计算指标
    metrics = compute_metrics(responses, rewards)
    log_metrics(metrics)
    
    # GRPO更新
    policy = grpo_update(policy, prompts, responses, rewards)
```

### 4.4 推理模式

```python
# 推理时的链式思考
def generate_with_reasoning(model, prompt, max_tokens=4096):
    """生成带推理的回复"""
    
    # 设置思考模式
    model.set_generation_mode('reasoning')
    
    # 生成
    response = model.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    
    return response

# 完整输出示例
"""
<think>
让我仔细分析这个数学问题。

已知条件：
- 函数 f(x) = x² + 2x + 1
- 需要找到导数

求导过程：
f'(x) = lim(h→0) [f(x+h) - f(x)]/h
       = lim(h→0) [(x+h)² + 2(x+h) + 1 - (x²+2x+1)]/h
       = lim(h→0) [x²+2xh+h² + 2x+2h + 1 - x² - 2x - 1]/h
       = lim(h→0) [2xh + h² + 2h]/h
       = lim(h→0) 2x + h + 2
       = 2x + 2

因此，f'(x) = 2x + 2
</think>
答案是：2x + 2
"""
```

## 5. 应用场景

### 5.1 数学推理

```python
# 数学问题示例
math_problem = """
问题：求函数 f(x) = x³ + 3x² + 2x + 1 的导数

请逐步推理并给出最终答案。
"""

# LLM生成带推理的答案
answer = generate_with_reasoning(model, math_problem)
print(answer)
```

### 5.2 代码生成

```python
# 代码生成任务
code_task = """
问题：编写一个Python函数，计算两个日期之间的天数差。

请逐步思考并给出代码。
"""

answer = generate_with_reasoning(model, code_task)
```

### 5.3 复杂问题分析

```python
# 逻辑推理任务
reasoning_task = """
问题：如果所有的鸟都会飞，企鹅是鸟，但企鹅不会飞。以下哪个结论是正确的？
A. 所有的鸟都会飞
B. 有些鸟不会飞
C. 企鹅不是鸟

请分析推理过程。
"""

answer = generate_with_reasoning(model, reasoning_task)
```

### 5.4 部署实现

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DeepSeekR1:
    """DeepSeek-R1模型封装"""
    
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def generate(self, prompt, max_tokens=4096, temperature=0.7):
        """带推理的生成"""
        
        # 构建输入
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 添加推理提示
        prompt_with_hint = f"{prompt}\n\n请在<think>标签中详细推理，然后给出答案。"
        
        inputs = self.tokenizer(prompt_with_hint, return_tensors='pt').to(self.model.device)
        
        # 生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
        )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        return response
    
    def extract_answer(self, response):
        """提取答案"""
        
        if '<think>' in response:
            # 提取思考过程
            think_start = response.find('<think>') + len('<think>')
            think_end = response.find('</think>')
            thinking = response[think_start:think_end]
            
            # 提取答案
            answer_start = response.find('答案是：') + len('答案是：')
            answer = response[answer_start:].strip()
            
            return {
                'thinking': thinking,
                'answer': answer
            }
        
        return {
            'thinking': None,
            'answer': response
        }

# 使用
model = DeepSeekR1('deepseek-ai/DeepSeek-R1')
response = model.generate("如何求函数 y = x² ���导���？")
result = model.extract_answer(response)
print(f"思考过程：{result['thinking'][:200]}...")
print(f"答案：{result['answer']}")
```

## 6. 优缺点分析

### 6.1 DeepSeek-R1的优点

1. **纯RL训练**：无需大量标注数据
2. **推理涌现**：自动学会链式思考
3. **自我验证**：能够检查和修正错误
4. **开源**：模型和代码公开可用
5. **高性能**：推理能力接近o1
6. **蒸馏能力**：可蒸馏到小模型

### 6.2 DeepSeek-R1的缺点

1. **输出冗长**：思考过程可能过长
2. **语言混淆**：中英文混杂
3. **幻觉问题**：仍存在事实性错误
4. **训练不稳定**：RL训练可能崩溃
5. **硬件要求**：需要大量GPU

### 6.3 与o1的对比

| 维度 | DeepSeek-R1 | OpenAI o1 |
|------|-------------|-----------|
| 训练方式 | GRPO+RL | PPO+RL |
| 思考过程 | 显式输出 | 隐式内部 |
| 模型规模 | 671B | 未公开 |
| 开源性 | 开源 | 闭源 |
| API费用 | 极低 | 高 |

### 6.4 使用场景

**推荐使用**：
- 数学推理任务
- 代码生成
- 复杂逻辑分析
- 研究辅助

**谨慎使用**：
- 需要简洁回答
- 事实性要求高
- 实时信息查询

## 7. 调库实现（Python + Transformers）

### 7.1 模型加载与使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载DeepSeek-R1
model_name = "deepseek-ai/DeepSeek-R1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 生成配置
def generate(
    prompt,
    max_tokens=4096,
    temperature=0.7,
    top_p=0.95,
):
    """生成带推理的回复"""
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    
    # 解码
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=False
    )
    
    return response

# 使用示例
response = generate("请计算函数 f(x)=x^3+2x^2+x+1 的导数")
print(response)
```

### 7.2 提取思考过程

```python
def parse_thinking_response(response):
    """解析思考过程和答案"""
    
    result = {
        'thinking': '',
        'answer': ''
    }
    
    # 提取思考过程
    if '<think>' in response:
        think_start = response.find('<think>') + len('<think>')
        think_end = response.find('</think>')
        
        if think_end > think_start:
            result['thinking'] = response[think_start:think_end].strip()
    
    # 提取答案（多种格式）
    for marker in ['答案是：', '答案：', 'answer:', '最终答案：']:
        if marker in response:
            answer_start = response.find(marker) + len(marker)
            result['answer'] = response[answer_start:].strip()
            break
    
    # 如果没有找到标记，取最后部分
    if not result['answer']:
        result['answer'] = response.strip()
    
    return result

# 使用
response = generate("如何求 1+2+3+...+100 的和？")
parsed = parse_thinking_response(response)
print("思考过程：")
print(parsed['thinking'][:300])
print("\n答案：")
print(parsed['answer'])
```

### 7.3 Gradio界面

```python
import gradio as gr

def chat(prompt, history=[]):
    """对话界面"""
    
    response = generate(prompt)
    parsed = parse_thinking_response(response)
    
    return parsed['answer'], parsed['thinking']

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# DeepSeek-R1 对话系统")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True
            )
        with gr.Column(scale=1):
            thinking_display = gr.Textbox(
                label="思考过程",
                height=500,
                interactive=False
            )
    
    with gr.Row():
        msg = gr.Textbox(
            label="输入问题",
            placeholder="请输入您的问题...",
            scale=3
        )
        submit_btn = gr.Button("提交", scale=1)
    
    def respond(prompt, history, chat_history):
        response = generate(prompt)
        parsed = parse_thinking_response(response)
        
        history.append((prompt, parsed['answer']))
        return "", history, parsed['thinking']
    
    submit_btn.click(respond, [msg, chatbot, thinking_display], [msg, chatbot, thinking_display])

demo.launch()
```

### 7.4 API服务

```python
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    prompt: str
    max_tokens: int = 4096
    temperature: float = 0.7

@app.post("/generate")
def generate_api(query: Query):
    """生成API"""
    response = generate(
        query.prompt,
        max_tokens=query.max_tokens,
        temperature=query.temperature
    )
    
    parsed = parse_thinking_response(response)
    
    return {
        "thinking": parsed['thinking'],
        "answer": parsed['answer'],
        "raw": response
    }

# 启动
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 8. 手工代码实现

### 8.1 GRPO实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class GRPO:
    """Group Relative Policy Optimization 实现"""
    
    def __init__(
        self,
        policy_model,
        learning_rate=1e-6,
        clip_epsilon=0.2,
        group_size=16,
        entropy_coef=0.01,
    ):
        self.policy_model = policy_model
        self.clip_epsilon = clip_epsilon
        self.group_size = group_size
        self.entropy_coef = entropy_coef
        
        self.optimizer = torch.optim.Adam(
            policy_model.parameters(),
            lr=learning_rate
        )
    
    def compute_log_prob(self, model, input_ids, attention_mask):
        """计算log probability"""
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 计算每个token的log prob
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取目标token的log prob
        token_log_probs = log_probs[:, :-1, :].gather(
            2,
            input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs
    
    def update(self, prompts, responses, rewards):
        """策略更新"""
        
        # 1. 计算组内相对优势
        mean_reward = torch.mean(torch.tensor(rewards))
        std_reward = torch.std(torch.tensor(rewards))
        
        advantages = []
        for r in rewards:
            adv = (r - mean_reward) / (std_reward + 1e-8)
            advantages.append(adv)
        
        # 2. 计算策略梯度损失
        total_loss = 0
        
        for i, (prompt, response, adv) in enumerate(zip(prompts, responses, advantages)):
            # Tokenize
            inputs = tokenizer(response, return_tensors='pt')
            
            # 计算log probability
            log_prob = self.compute_log_prob(
                self.policy_model,
                inputs.input_ids,
                inputs.attention_mask
            )
            
            # 简化：使用最后一个token的log prob
            last_log_prob = log_prob[:, -1]
            
            # GRPO损失
            loss = -adv * last_log_prob.mean()
            
            # 熵正则化
            if self.entropy_coef > 0:
                dist = Categorical(logits=log_probs)
                entropy = dist.entropy().mean()
                loss -= self.entropy_coef * entropy
            
            total_loss += loss
        
        # 3. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()


def train_grpo(model, prompts, reward_fn, num_steps):
    """GRPO训练循环"""
    
    grpo = GRPO(model)
    
    for step in range(num_steps):
        # 1. 对每个prompt采样多个响应
        all_responses = []
        all_rewards = []
        
        for prompt in prompts:
            responses = model.generate(
                prompt,
                num_return_sequences=grpo.group_size
            )
            all_responses.extend(responses)
            
            # 计算每个响应的奖励
            for response in responses:
                reward = reward_fn(prompt, response)
                all_rewards.append(reward)
        
        # 2. GRPO更新
        loss = grpo.update(prompts, all_responses, all_rewards)
        
        # 3. 日志
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.4f}, "
                  f"Mean Reward: {sum(all_rewards)/len(all_rewards):.4f}")
```

### 8.2 奖励函数实现

```python
import re

def compute_reward(prompt, response):
    """计算奖励"""
    
    total_reward = 0.0
    
    # 1. 格式奖励
    has_think_tag = '<think>' in response and '</think>' in response
    has_answer = any(marker in response for marker in ['答案是', '答案：', 'answer:'])
    
    if has_think_tag:
        total_reward += 0.1
    if has_answer:
        total_reward += 0.1
    
    # 2. 内容奖励（简化版）
    # 实际应用中需要根据具体任务设计
    is_coherent = len(response) > 50  # 响应长度合理
    has_reasoning = is_coherent and has_think_tag
    
    if has_reasoning:
        total_reward += 0.3
    
    # 3. 准确性奖励（需要外部验证器）
    # 这里使用简单的关键词匹配作为示例
    if "正确" in response or "对" in response or "yes" in response.lower():
        total_reward += 0.1
    
    return total_reward


def mathematical_reward(prompt, response):
    """数学问题奖励"""
    
    reward = 0.0
    
    # 格式检查
    if '<think>' in response and '</think>' in response:
        reward += 0.1
    if '答案是' in response or '答案：' in response:
        reward += 0.1
    
    # 根据正确答案检查（需要预先知道答案）
    # 这里简化处理
    if '=' in response and any(c.isdigit() for c in response):
        reward += 0.3
    
    return reward


def code_reward(prompt, response):
    """代码生成奖励"""
    
    reward = 0.0
    
    # 格式检查
    has_think = '<think>' in response
    has_code = '```' in response
    
    if has_think:
        reward += 0.1
    if has_code and 'def ' in response:
        reward += 0.5
    
    # 语法检查（简化）
    if 'import ' in response and ':' in response:
        reward += 0.2
    
    return reward
```

### 8.3 推理生成实现

```python
def generate_with_thinking(
    model,
    tokenizer,
    prompt,
    max_tokens=4096,
    temperature=0.7,
):
    """带思考的生成"""
    
    # 构建输入
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    prompt_with_hint = f"""{prompt}

请详细思考后在<think>标签中给出推理过程，然后给出最终答案。"""
    
    inputs = tokenizer(prompt_with_hint, return_tensors="pt").to(model.device)
    
    # 生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 解码
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[0]:],
        skip_special_tokens=False
    )
    
    return response


def batch_generate(model, tokenizer, prompts):
    """批量生成"""
    
    results = []
    
    for prompt in prompts:
        response = generate_with_thinking(model, tokenizer, prompt)
        results.append(response)
    
    return results
```

## 9. 可视化与结果理解

### 9.1 训练曲线可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curve():
    """可视化训练过程"""
    
    # 模拟训练数据
    steps = np.arange(0, 1000, 10)
    rewards = np.cumsum(np.random.randn(100)) / 10 + np.linspace(0, 1, 100)
    losses = -np.cumsum(np.random.randn(100)) / 5 + np.linspace(-2, 0, 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 奖励曲线
    axes[0].plot(steps[:len(rewards)], rewards, 'b-')
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Reward during Training')
    axes[0].grid(True, alpha=0.3)
    
    # 损失曲线
    axes[1].plot(steps[:len(losses)], losses, 'r-')
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss during Training')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('r1_training.png', dpi=150)
    plt.close()

plot_training_curve()
```

### 9.2 Token分布可视化

```python
def plot_token_distribution():
    """可视化生成token的长度分布"""
    
    # 模拟数据
    think_lengths = np.random.normal(500, 200, 100)
    answer_lengths = np.random.normal(100, 50, 100)
    
    plt.figure(figsize=(10, 6))
    plt.hist(think_lengths, bins=30, alpha=0.6, label='思考过程长度')
    plt.hist(answer_lengths, bins=30, alpha=0.6, label='答案长度')
    plt.xlabel('Token数量')
    plt.ylabel('频数')
    plt.title('生成内容长度分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('token_distribution.png', dpi=150)
    plt.close()

plot_token_distribution()
```

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 测试集 |
|------|------|-------|
|准确率 | 任务正确率 | MATH, HumanEval |
|推理长度 | 思考过程长度 | 采样生成 |
|格式正确率 | 输出格式正确 | 自定义 |
|语言一致性 | 中英文占比 | 自定义 |

### 10.2 基准测试

```python
def evaluate_benchmark(model, tokenizer, benchmarks):
    """基准测试"""
    
    results = {}
    
    for benchmark_name, dataset in benchmarks.items():
        correct = 0
        total = 0
        
        for prompt, answer in dataset:
            # 生成响应
            response = generate(prompt)
            
            # 检查答案
            if answer in response:
                correct += 1
            total += 1
        
        results[benchmark_name] = correct / total
    
    return results

# 使用
benchmarks = {
    'math': [("1+1=", "2"), ("2+2=", "4")],
    'code': [("def add(a,b):", "a+b")],
}
results = evaluate_benchmark(model, tokenizer, benchmarks)
print(results)
```

## 11. 常见问题与易错点

### 11.1 输出格式不规范

**问题**：模型输出格式不统一
**解决**：使用提示词引导格式，或后处理提取答案

### 11.2 思考过程过长

**问题**：思考过程占用太多token
**解决**：限制思考长度或使用更短的思考提示

### 11.3 幻觉问题

**问题**：生成错误的事实
**解决**：结合RAG或外部验证

### 11.4 语言混杂

**问题**：中文问题用英文回答
**解决**：在提示中指定语言

## 12. 学习总结

### 核心要点

1. **GRPO算法**：组内相对排名计算优势，无价值函数
2. **RL训练**：纯强化学习，涌现推理能力
3. **思考涌现**：自动学会链式思考
4. **两阶段**：先RL后蒸馏

### 关键创新

- **无SFT**：DeepSeek-R1-Zero纯RL训练
- **格式奖励**：引导正确输出格式
- **可验证**：基于可验证的奖励

### 应用建议

- 数学推理：首选
- 代码生成：推荐
- 复杂问题分析：推荐

## 13. 练习题与思考题

### 练习题

**Q1**: GRPO和PPO的主要区别是什么？

**答案**：GRPO使用组内相对排名计算优势，不需要单独训练价值函数，比PPO更简单稳定。

**Q2**: DeepSeek-R1是如何涌现出推理能力的？

**答案**：通过强化学习训练，模型在追求更高奖励的过程中自动学会链式思考，不需要显式的推理数据。

**Q3**: 为什么要使用冷启动数据？

**答案**：冷启动数据帮助模型建立基本的格式意识，防止RL训练早期崩溃。

### 思考题

**Q1**: RL训练的挑战和如何解决？

**答案**：训练不稳定、奖励函数设计困难。可以通过适当的奖励塑形和多阶段训练解决。

**Q2**: 如何评估推理能力？

**答案**：使用数学、逻辑等可验证的基准测试，或者设计专门的推理任务测试集。

## 14. 学习路径建议

### 基础阶段
1. 大语言模型基础
2. 强化学习基础（PPO）
3. GRPO原理理解

### 进阶阶段
1. 奖励函数设计
2. 训练稳定性分析
3. 模型蒸馏

### 实践阶段
1. 部署DeepSeek-R1
2. 自定义任务微调
3. 构建对话系统

### 参考资源
- DeepSeek-R1论文
- HuggingFace模型
- 开源代码仓库