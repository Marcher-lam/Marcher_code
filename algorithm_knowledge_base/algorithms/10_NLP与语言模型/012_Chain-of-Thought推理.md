# Chain-of-Thought (CoT) 推理 学习文档

> 通过生成中间推理步骤，使LLM能够逐步解决复杂问题。

> 来源线索：本节内容根据原书中关于"reasoning / chain-of-thought"的相关章节整理、扩展与教学化改写。

## 1. 算法基础认知

### 一句话定义
Chain-of-Thought (CoT) 是一种让大语言模型在给出最终答案前，先生成一系列中间推理步骤的方法。

### 直觉类比
想象你在解一道复杂的数学题。你不会直接写出答案，而是会在草稿纸上写下"首先……然后……因此……"。CoT 就是让 LLM 也这样做——把"思考过程"写出来，从而更可能得到正确答案。

### 历史背景
CoT 的概念在 2022 年由 Google 研究团队在论文《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》中系统提出。但"模型生成中间步骤"的思想在此之前就已存在。2024年9月，OpenAI 发布 o1 模型，将 CoT 推理推向公众视野；2025年1月，DeepSeek-R1 开源了完整的推理训练方案。

### 算法定位
- **类型**：推理时技术 / 提示工程方法
- **性质**：不修改模型权重，仅改变推理时的解码策略或提示方式

### 前置知识
- 了解 LLM 的基本文本生成原理（自回归生成）
- 了解 token 和 tokenizer 的基本概念
- 基本的概率论知识（理解概率分布）

## 2. 核心原理

### 核心思想
传统 LLM 在回答问题时，直接从输入映射到输出。对于简单的事实性问题（如"法国的首都是什么？"），这足够了。但对于需要多步推理的复杂问题（如数学证明、逻辑推理），模型需要将问题分解为子问题，逐步推进。

CoT 的核心思想是：**让模型显式地生成推理的中间步骤**。这些中间步骤形成一个"思维链"，引导模型从问题逐步走向答案。

### 工作流程
1. 用户给出一个需要推理的问题
2. LLM 不直接输出答案，而是先生成推理步骤（如"首先观察到……""因此可以推出……"）
3. LLM 基于推理步骤得出最终结论
4. 输出可以包含推理过程（可见 CoT）或隐藏推理过程（隐藏 CoT）

### 关键概念解释
- **可见 CoT**：用户可以看到模型的中间推理步骤，增强了可解释性
- **隐藏 CoT**：模型内部进行推理但不展示给用户（如 OpenAI o1 系列），保护商业机密并防止蒸馏
- **零样本 CoT**：仅通过添加"Let's think step by step"等提示词触发推理
- **少样本 CoT**：提供几个包含推理步骤的示例，模型模仿推理模式

### 直观解释
```
没有 CoT 的模型：
问题: "小明有5个苹果，给了小红2个，又买了3个，现在有几个？"
输出: "6个"  ← 可能正确，也可能错误，没有过程可查

使用 CoT 的模型：
问题: "小明有5个苹果，给了小红2个，又买了3个，现在有几个？"
输出: "小明初始有5个苹果。给出2个后剩余 5-2=3 个。
       又买了3个后变为 3+3=6 个。所以答案是6个。"
      ← 有完整的推理链，可验证每一步
```

## 3. 数学公式与推导

### 符号约定
| 符号 | 含义 |
|------|------|
| $x$ | 输入问题（token 序列） |
| $y$ | 最终答案 |
| $z = (z_1, z_2, ..., z_k)$ | 中间推理步骤 |
| $p(\cdot)$ | 模型给出的概率分布 |
| $\theta$ | 模型参数 |

### 问题形式化

标准直接回答可表示为：
$$p(y | x; \theta)$$

而 CoT 推理可表示为：
$$p(y | x; \theta) = \sum_{z} p(y | x, z; \theta) \cdot p(z | x; \theta)$$

这意味着模型先生成推理链 $z$，再基于 $x$ 和 $z$ 生成答案 $y$。

### 自回归展开

推理链的生成是自回归的：
$$p(z | x) = \prod_{t=1}^{k} p(z_t | x, z_{<t})$$

同样，答案的生成也是自回归的：
$$p(y | x, z) = \prod_{t=1}^{m} p(y_t | x, z, y_{<t})$$

### 最终公式

整个 CoT 生成过程可统一写为：
$$p(y, z | x) = \underbrace{\prod_{t=1}^{k} p(z_t | x, z_{<t})}_{\text{推理链生成}} \cdot \underbrace{\prod_{t=1}^{m} p(y_t | x, z, y_{<t})}_{\text{答案生成}}$$

其中 $z_{<t}$ 表示前 $t-1$ 个推理步骤 token，$y_{<t}$ 表示前 $t-1$ 个答案 token。

## 4. 训练过程讲解

CoT 本身不涉及训练（这是推理时技术），但训练推理模型时需要考虑以下方面。

### 数据预处理
- 收集或生成包含详细推理步骤的问答对
- 推理步骤需包含清晰的中间计算或逻辑推导
- 需要统一推理格式（如用特定标签包裹推理部分）

### 训练时的推理格式
对于通过 SFT 训练的推理模型，训练数据通常采用以下格式：
```
<|im_start|>user
问题内容
<|im_end|>
<|im_start|>assistant
thinking
详细的推理步骤...

response
最终答案
<|im_end|>
```

### 推理时的超参数
| 参数 | 作用 | 推荐范围 | 默认建议 |
|------|------|----------|----------|
| max_new_tokens | 推理 + 回答的最大总token数 | 512-4096 | 2048 |
| temperature | 控制生成的随机性 | 0.0-1.0 | 0.6-0.7 |
| top_p | 核采样概率阈值 | 0.9-1.0 | 0.95 |

## 5. 应用场景

### 典型应用
1. **数学问题求解**：多步算术、代数、几何证明。CoT 将计算步骤显式化，大幅降低计算错误率。
2. **代码生成与调试**：先分析需求，再逐步构建代码逻辑。适合复杂算法实现。
3. **逻辑推理**：演绎推理、三段论、逻辑谜题。CoT 让模型的推理链可追溯。
4. **多文档问答**：需要综合多个信息源得出结论。CoT 先提取再整合。
5. **科学推理**：物理、化学等需要从定律推导结论的领域。

### 适用数据特征
- 问题需要多步推理
- 单步答案容易出错
- 推理链在训练数据中有迹可循

### 不适用场景
- 简单事实性问答（如"What is the capital of France?"）
- 纯创意写作
- 翻译任务
- 需要极快响应的场景（CoT 因生成长文本更慢）

## 6. 优缺点分析

### 优点
| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 提高复杂任务准确率 | 分解问题减少跳跃推理错误 | 问题确实需要多步推理 |
| 增强可解释性 | 用户可以看到推理过程 | 使用可见 CoT 模式 |
| 零样本泛化 | 仅需"Let's think step by step"即可触发 | 模型已有一定推理基础能力 |
| 无需训练 | 可直接应用于任何预训练LLM | 模型质量足够好 |
| 错误可定位 | 可以检查哪个推理步骤出错 | 推理步骤被输出 |

### 缺点
| 缺点 | 说明 | 缓解思路 |
|------|------|----------|
| 推理成本高 | 输出更长，token 消耗翻倍 | 对简单问题不使用 CoT |
| 可能产生幻觉推理 | 推理过程逻辑正确但基于虚构事实 | 结合检索增强生成(RAG) |
| 推理不一定可靠 | 看似合理的推理链可能导向错误结论 | 多次采样 + 多数投票 |
| 效率低 | 生成速度更慢 | 使用 KV cache 等加速技术 |
| 过度推理 | 简单问题也被过度分析 | 实现问题难度判断的路由机制 |

### 与同类方法对比
| 方法 | 核心思路 | 是否训练 | 成本 | 效果 |
|------|----------|----------|------|------|
| CoT 提示 | 提示模型生成推理链 | 否 | 中 | 显著提升推理 |
| 直接回答 | LLM直接输出答案 | 否 | 低 | 复杂问题较差 |
| SFT 推理训练 | 用推理数据微调模型 | 是 | 高(训练时) | 更稳定 |
| 推理时扩展(MCTS等) | 搜索多条推理路径 | 否 | 很高 | 最佳(成本最高) |

## 7. 调库实现

```python
"""
Chain-of-Thought 推理的调库实现
使用 HuggingFace Transformers + 本地模型 或 API
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- 1. 加载模型（这里以 Qwen2.5 为例） ----
# 实际使用时替换为你的模型路径
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 节省显存
    device_map="auto"            # 自动选择设备
)
model.eval()

# ---- 2. 定义 CoT 提示模板 ----
COT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "When solving problems, think step by step and show your reasoning clearly. "
    "Start with 'Let me think step by step:'"
)

def build_cot_prompt(question: str, use_cot: bool = True) -> str:
    """构建带或不带 CoT 的提示"""
    if use_cot:
        return (
            f"{COT_SYSTEM_PROMPT}\n\n"
            f"Question: {question}\n"
            f"Let me think step by step:"
        )
    else:
        return f"Question: {question}\nAnswer:"


# ---- 3. 文本生成函数 ----
@torch.inference_mode()
def generate_with_cot(
    model,
    tokenizer,
    question: str,
    use_cot: bool = True,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
) -> str:
    """使用或不使用 CoT 生成回答"""
    prompt = build_cot_prompt(question, use_cot)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 只取新生成的部分
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


# ---- 4. 对比测试 CoT vs 无 CoT ----
questions = [
    "小明有15个苹果，他用2/5换了橙子，又将剩下的1/3给了朋友，最后买了8个苹果。现在有多少苹果？",
    "If a train travels at 60 mph for 2.5 hours, then at 45 mph for 1.5 hours, what is the total distance?",
]

for q in questions:
    print(f"\n问题: {q}")
    print("=" * 60)

    print("\n--- 使用 CoT ---")
    answer_cot = generate_with_cot(model, tokenizer, q, use_cot=True)
    print(answer_cot)

    print("\n--- 不使用 CoT ---")
    answer_no_cot = generate_with_cot(model, tokenizer, q, use_cot=False)
    print(answer_no_cot)
```

运行示例输出：
```
问题: 小明有15个苹果，他用2/5换了橙子，又将剩下的1/3给了朋友，最后买了8个苹果。现在有多少苹果？
============================================================

--- 使用 CoT ---
Let me think step by step:
1. 小明初始有15个苹果
2. 用2/5换橙子，即15 × 2/5 = 6个苹果被换走，剩余15 - 6 = 9个苹果
3. 将剩下的1/3给了朋友，即9 × 1/3 = 3个苹果给出，剩余9 - 3 = 6个苹果
4. 买了8个苹果后，6 + 8 = 14个苹果
所以小明现在有 **14个苹果**。

--- 不使用 CoT ---
小明现在有14个苹果。
```

## 8. 手工代码实现

```python
"""
Chain-of-Thought 推理的手工实现
展示 CoT 背后的核心解码逻辑（贪婪解码，可替换为其他解码策略）
"""

import numpy as np


class ChainOfThoughtGenerator:
    """
    CoT 生成器：模拟 LLM 在推理模式下逐 token 生成的过程

    注意：这是一个教学实现，实际 LLM 推理需要完整的 Transformer 架构。
    本实现展示 CoT 的解码逻辑框架。
    """

    def __init__(self, model, tokenizer, cot_trigger: str = "Let me think step by step"):
        """
        参数:
            model: 一个 callable，接受 token_ids 返回 logits
            tokenizer: 提供 encode/decode 功能
            cot_trigger: 触发 CoT 的文本
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cot_trigger = cot_trigger
        self.cot_trigger_ids = tokenizer.encode(cot_trigger)

    def generate_raw(self, token_ids, max_new_tokens, eos_token_id=None):
        """
        基础的逐 token 贪婪生成（不使用 KV cache 的简化版）

        这反映了 CoT 的底层机制：每个 token 都是基于前序内容预测的。
        """
        input_length = len(token_ids)
        generated = []

        for _ in range(max_new_tokens):
            # 将当前序列输入模型获得 logits
            logits = self.model(token_ids)

            # 取最后一个位置的 logits，选 argmax（贪婪解码）
            last_logits = logits[-1]
            next_token_id = int(np.argmax(last_logits))
            next_token_id = int(last_logits.argmax())

            # 检查是否结束
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            generated.append(next_token_id)
            token_ids = np.concatenate([token_ids, [next_token_id]])

        return generated

    def generate_cot(
        self,
        question: str,
        max_reasoning_tokens: int = 256,
        max_answer_tokens: int = 128,
        show_reasoning: bool = True,
    ):
        """完整的 CoT 生成：提示构建 -> 推理链生成 -> 答案生成"""

        # 第一步：构建 CoT 提示
        prompt = f"Question: {question}\n{self.cot_trigger}"
        prompt_ids = self.tokenizer.encode(prompt)

        # 第二步：生成推理链 + 答案
        # 在实际LLM中，推理链和答案是一起自回归生成的，没有明确分界。
        # 这里为了教学分开展示。
        all_generated = self.generate_raw(
            np.array(prompt_ids),
            max_new_tokens=max_reasoning_tokens + max_answer_tokens,
        )

        full_text = self.tokenizer.decode(all_generated)

        result = {
            "question": question,
            "full_response": full_text,
            "reasoning_visible": show_reasoning,
        }

        return result


# ===== 模拟测试 =====
class MockTokenizer:
    """模拟 tokenizer 用于演示 CoT 逻辑"""
    def __init__(self):
        self.vocab = {
            "Question": 1, ":": 2, " ": 3,
            "Let": 4, "me": 5, "think": 6, "step": 7, "by": 8,
            "Let me think step by step": [4, 3, 5, 3, 6, 3, 7, 3, 8, 3, 7],
            "the": 9, "answer": 10, "is": 11, "42": 12, ".": 13,
            "What": 14, "meaning": 15, "of": 16, "life": 17, "?": 18,
            "<|endoftext|>": 0,
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        if text in self.vocab:
            result = self.vocab[text]
            return result if isinstance(result, list) else [result]
        return [self.vocab.get(w, 0) for w in text.split()]

    def decode(self, ids):
        return " ".join(self.reverse_vocab.get(int(i), "?") for i in ids)


def test_cot_generator():
    """测试 CoT 生成器"""
    tokenizer = MockTokenizer()

    # 模拟模型：总是预测 token 9 ("the")
    def mock_model(token_ids):
        vocab_size = 19
        logits = np.zeros((len(token_ids), vocab_size))
        logits[:, 9] = 1.0  # 总是倾向于预测 token 9
        return logits

    generator = ChainOfThoughtGenerator(mock_model, tokenizer)

    result = generator.generate_cot("What is the meaning of life?")
    print("问题:", result["question"])
    print("回答:", result["full_response"])
    print("推理可见:", result["reasoning_visible"])


if __name__ == "__main__":
    test_cot_generator()
```

## 9. 可视化与结果理解

```python
"""
CoT 推理的可视化：对比有无 CoT 的效果差异
"""

import matplotlib.pyplot as plt
import numpy as np

# ---- 模拟数据：是否使用 CoT 在各任务上的准确率对比 ----
tasks = [
    "基础算术", "多步算术", "代数", "逻辑推理",
    "代码生成", "常识问答", "事实问答"
]

# 典型研究中 CoT 与无 CoT 的效果对比（示意数据）
no_cot_accuracy = [0.85, 0.45, 0.35, 0.50, 0.40, 0.75, 0.88]
cot_accuracy =    [0.87, 0.72, 0.58, 0.68, 0.62, 0.77, 0.88]

x = np.arange(len(tasks))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, no_cot_accuracy, width, label="不使用 CoT", color="#E74C3C", alpha=0.8)
bars2 = ax.bar(x + width/2, cot_accuracy, width, label="使用 CoT", color="#2ECC71", alpha=0.8)

# 标注提升幅度
for i, (nc, cc) in enumerate(zip(no_cot_accuracy, cot_accuracy)):
    improvement = (cc - nc) * 100
    if improvement > 2:
        ax.annotate(
            f"+{improvement:.0f}%",
            xy=(i + width/2, cc),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#27AE60",
        )

ax.set_xlabel("任务类型", fontsize=12, fontweight="bold")
ax.set_ylabel("准确率", fontsize=12, fontweight="bold")
ax.set_title("Chain-of-Thought (CoT) 推理效果对比\nCoT 在多步推理任务上提升显著", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)

# 高亮差距大的区域
for i in range(len(tasks)):
    if cot_accuracy[i] - no_cot_accuracy[i] > 0.15:
        ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color="yellow")

plt.tight_layout()
plt.show()

# ---- 解读 ----
print("""
图表解读：
1. 在"多步算术""代数""逻辑推理""代码生成"等需要推理链的任务上，
   CoT 带来了 15-27% 的准确率提升。
2. 在"事实问答""常识问答"等模式匹配任务上，CoT 提升微乎其微。
3. 这说明 CoT 的价值在于分解复杂推理，而非改进简单的记忆召回。

关键结论：CoT 不是万能的——它最适合需要中间推理步骤的任务。
""")
```

## 10. 模型评估

CoT 推理的评估需要关注两方面：推理过程的正确性和最终答案的准确性。

```python
"""
CoT 推理评估指标实现
"""

from typing import List, Dict
import re


class CoTEvaluator:
    """CoT 推理的评估器"""

    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> bool:
        """精确匹配：最严格的评估"""
        return prediction.strip().lower() == ground_truth.strip().lower()

    @staticmethod
    def answer_extraction(text: str) -> str:
        """从 CoT 输出中提取最终答案"""
        # 尝试匹配常见答案模式
        patterns = [
            r"(?:因此|所以|答案[是为]|answer is|the answer is|final answer:?)\s*(.+?)(?:\.|$)",
            r"(?:答案是?|结果为?)\s*(.+?)(?:\.|$)",
            r"\*\*(.+?)\*\*",  # **答案内容**
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        # 退回最后一行
        return text.strip().split("\n")[-1].strip()

    @staticmethod
    def reasoning_step_count(text: str) -> int:
        """估算推理链中的步骤数"""
        step_markers = [
            r"\d+\.",
            r"(?:首先|第一步|然后|接下来|其次|接着|最后|最终)",
            r"(?:First|Step \d|Then|Next|Finally)",
            r"\n\s*-",
        ]
        count = 0
        for marker in step_markers:
            count += len(re.findall(marker, text, re.IGNORECASE))
        return max(count, 1)

    def evaluate(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """批量评估 CoT 推理结果"""
        results = {
            "total": len(predictions),
            "exact_match": 0,
            "extracted_correct": 0,
            "avg_reasoning_steps": 0,
            "per_sample": [],
        }

        for pred, gt in zip(predictions, ground_truths):
            # 精确匹配
            em = self.exact_match(pred, gt)

            # 提取答案后匹配
            extracted = self.answer_extraction(pred)
            ext_correct = self.exact_match(extracted, gt)

            # 推理步骤数
            steps = self.reasoning_step_count(pred)

            results["exact_match"] += int(em)
            results["extracted_correct"] += int(ext_correct)
            results["avg_reasoning_steps"] += steps
            results["per_sample"].append({
                "prediction": pred[:100],
                "extracted_answer": extracted,
                "ground_truth": gt,
                "exact_match": em,
                "extracted_match": ext_correct,
                "reasoning_steps": steps,
            })

        results["exact_match_rate"] = results["exact_match"] / results["total"]
        results["extracted_accuracy"] = results["extracted_correct"] / results["total"]
        results["avg_reasoning_steps"] /= results["total"]

        return results


# ===== 使用示例 =====
evaluator = CoTEvaluator()

predictions = [
    "首先计算 15×2/5=6。余下 15-6=9。再减去 9×1/3=3。9-3=6。加上8得14。所以答案是 14。",
    "Let me think: 60×2.5=150, and 45×1.5=67.5. So total is 217.5.",
]
ground_truths = ["14", "217.5"]

results = evaluator.evaluate(predictions, ground_truths)

print(f"总样本: {results['total']}")
print(f"精确匹配率: {results['exact_match_rate']:.1%}")
print(f"提取答案准确率: {results['extracted_accuracy']:.1%}")
print(f"平均推理步骤: {results['avg_reasoning_steps']:.1f}")

for i, sample in enumerate(results["per_sample"]):
    print(f"\n--- 样本 {i+1} ---")
    print(f"  提取答案: '{sample['extracted_answer']}'")
    print(f"  真实答案: '{sample['ground_truth']}'")
    print(f"  推理步骤: {sample['reasoning_steps']}")

# ---- 结果解读 ----
print("""
评估指标选择理由：
- 精确匹配：适合数学/代码类有唯一答案的任务
- 提取+匹配：CoT 输出通常包含大量文本，需要先提取答案再比较
- 推理步骤数：用于分析推理链是否完整（太少可能跳步，太多可能冗余）

注意事项：
- 对于开放式任务，需使用 LLM-as-judge 或人工评估
- 推理链的正确性评估是最困难的——链可能逻辑正确但结论错误
""")
```

## 11. 常见问题与易错点

### 数据层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 提示过短 | CoT 输出就像直接回答，没有推理链 | 没有明确要求模型生成推理 | 使用明确的触发词"请逐步思考""Let's think step by step" |
| 提示过长 | 推理链被示例格式"锁死"，对新问题不适应 | 少样本示例限制了模型灵活性 | 少样本不超过3个示例，且覆盖不同推理模式 |
| 未指定输出格式 | 不确定性高，答案在不同位置 | 模型输出格式不稳定 | 规定输出格式，如"最终答案：XXX" |

### 模型层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 推理幻觉 | 中间步骤引用不存在的事实 | 模型编造看似合理的推理 | 搭配 RAG 提供事实基础 |
| 推理链错误但答案正确 | 表面正确但推理过程有缺陷 | 巧合或记忆而非推理 | 同时评估推理链和最终答案 |
| 预测下一个 token 的后门效应 | CoT 只是美化了的模式匹配 | LLM 本质仍是统计模型 | 理解 CoT 的局限性，不要过高期待 |

### 调参层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| temperature 过低 | 推理链千篇一律，缺乏灵活性 | 贪婪解码总是选择概率最高的路径 | 设置 temperature=0.6-0.7 增加多样性 |
| temperature 过高 | 推理链混乱，逻辑不通 | 随机性太高导致离题 | 降低到 0.3-0.5 |

## 12. 学习总结

### 核心思想回顾
Chain-of-Thought 是一种推理时方法，通过让 LLM 在给出答案前生成中间推理步骤来提升复杂任务的准确率。它利用了 LLM 的自回归生成特性，将"思考过程"显式化为 token 序列。

### 关键公式
1. CoT 的联合概率：$p(y, z | x) = p(z | x) \cdot p(y | x, z)$
2. 自回归推理链：$p(z | x) = \prod_{t=1}^{k} p(z_t | x, z_{<t})$

### 与前序/相关算法的联系
- CoT 是"推理时计算扩展"的基础技术
- 与自回归文本生成共享底层解码机制
- 强化学习推理训练是对 CoT 能力的系统化强化

### 后续学习方向
- 推理时扩展的进阶技术（MCTS、Best-of-N、multi-agent 推理）
- 强化学习训练推理模型（GRPO、PPO）
- 推理模型的评估方法与基准

## 13. 练习题与思考题

### 基础题

**题1**：请解释为什么 CoT 在"法国首都是什么？"这类问题上不会带来提升。

**参考答案**：
这类问题是简单的事实回忆任务。LLM 已经学会了"法国 → 巴黎"这个强关联，无需推理步骤。加入 CoT 只会生成多余的 token（如"让我想想...法国是欧洲国家，首都是巴黎"），增加了计算成本但不会改变答案。CoT 适合的是需要组合多条信息或进行多步推导的问题。

**题2**：修改本教程第7节中的代码，添加一个函数使用少样本 CoT（提供3个含推理步骤的示例）。

**参考答案**：
```python
FEW_SHOT_EXAMPLES = """
Question: 小明有10个苹果，吃了3个，又买了5个，一共有几个？
Let me think step by step:
初始: 10个苹果
吃了3个: 10 - 3 = 7个
买了5个: 7 + 5 = 12个
答案: 12个

Question: 一个长方形长8cm宽5cm，周长是多少？
Let me think step by step:
周长公式: 2 × (长 + 宽)
计算: 2 × (8 + 5) = 2 × 13 = 26
答案: 26cm

"""

def build_few_shot_cot_prompt(question: str) -> str:
    return f"{FEW_SHOT_EXAMPLES}Question: {question}\nLet me think step by step:"
```

### 进阶题

**题3**：一个模型在 CoT 方式下回答数学题时，推理链完全正确但最终给出了错误答案。分析可能的原因并提出至少两种解决方案。

**参考答案**：
可能原因：
1. 推理链完全正确但最后一步计算出了错——这通常是自回归生成中"长距离依赖遗忘"的表现。模型在生成推理链的长度中"忘记"了前面的计算结果。
2. 答案提取逻辑有缺陷——推理链和答案的 token 边界不清晰。

解决方案：
- **方案A**：在推理链末尾明确重复关键计算结果，如"回顾：剩余6个苹果，买8个，6+8=14"
- **方案B**：使用两阶段生成——先生成推理链，再基于原问题 + 推理链重新生成答案（但需要两次前向传播）
- **方案C**：使用输出格式约束，要求最终答案用特定标签包裹如`<answer>14</answer>`，便于解析。

### 开放思考题

**题4**：当前推理模型（如 OpenAI o1、DeepSeek-R1）普遍使用"隐藏 CoT"（用户看不到推理过程）。请分析这种设计的利弊。

**参考答案**：
**利**：
- 防止竞争对手通过 API 输出蒸馏推理能力
- 避免推理内容可能包含的不安全内容被用户看到
- 允许模型使用更自由的内部推理格式，不必考虑可读性
- 保护商业机密和训练数据隐私

**弊**：
- 降低了可解释性和可信度（"黑箱"推理）
- 用户无法验证推理是否正确
- 不利于安全审计和偏差检测
- 推理过程出现错误时无从调试

这是一个产品策略权衡：安全/商业保护 vs 透明度/可信度。

## 14. 学习路径建议

### 前置算法
- 了解 LLM 的 Token 化原理（BPE）
- 掌握自回归文本生成机制

### 平行算法
- 自洽性(Self-Consistency)：多次采样 CoT 后多数投票
- Tree-of-Thought：分支探索多条推理路径
- 推理时计算扩展：MCTS、Best-of-N 等更复杂的推理时方法

### 进阶算法
- 强化学习推理训练（GRPO / PPO 训练推理模型）
- 知识蒸馏：将大模型的推理能力迁移到小模型
- Multi-Agent 推理：多个模型协同推理

### 推荐资源
1. **论文**：Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (NeurIPS 2022) — CoT 的奠基性论文
2. **文章**：Sebastian Raschka, "Understanding Reasoning LLMs" — 深入浅出的推理模型综述
3. **论文**：DeepSeek-R1 Technical Report (arXiv:2501.12948) — 开源推理模型训练的完整方案
