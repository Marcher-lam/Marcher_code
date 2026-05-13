# Beam Search 学习文档

> 宽度为k的束搜索，序列生成中的经典解码策略，平衡质量与计算量。

---

## 1. 算法基础认知

**Beam Search（束搜索）** 是序列生成任务中最重要的解码策略之一，广泛应用于机器翻译、文本摘要、语音识别、对话系统等任务。其核心思想是在每一步生成时保留Top-K个最有可能的候选序列，而非像贪心搜索那样只保留最优解，从而在计算效率和生成质量之间取得平衡。

### 1.1 为什么需要Beam Search？

在序列生成任务中，目标是从一个巨大的搜索空间中找到最优的标记序列。例如，在机器翻译中，假设词汇表大小为30000，句子长度为20，那么可能的序列数量高达30000^20，远超宇宙中原子数量的级别。Brute-force遍历所有可能的序列是不可能的，我们需要高效的搜索策略。

### 1.2 Beam Search vs 其他解码策略

| 解码策略 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| 贪心搜索(Greedy) | 计算快，实现简单 | 可能陷入局部最优 | 实时性要求高的场景 |
| Beam Search | 质量较高，效率适中 | 需要调参beam size | 大多数生成任务 |
| 采样(Sampling) | 多样性高 | 不确定性大 | 创意写作、对话 |
| 维特比(Viterbi) | 全局最优 | 内存消耗大 | 短序列、小词汇 |

---

## 2. 核心原理

### 2.1 基本流程

Beam Search的核心流程可以概括为以下步骤：

**步骤1：初始化**
- 设置beam width（束宽）k
- 初始化开始标记序列 `<s>` 或 `<bos>`
- 初始候选序列列表只包含一个元素：start token

**步骤2：迭代扩展**
- 对每个候选序列，生成下一个词的概率分布
- 使用模型计算条件概率 P(y_{t+1} | y_1, ..., y_t, x)
- 从所有候选序列的扩展中选择概率最高的Top-K个

**步骤3：剪枝与选择**
- 对所有扩展后的序列按对数概率排序
- 保留分数最高的K个序列
- 如果某个序列遇到结束标记 `<eos>`，则保留但不继续扩展

**步骤4：终止条件**
- 所有候选序列都遇到结束标记，或
- 达到最大长度限制

**步骤5：选择最优**
- 从最终候选中选择概率最高的序列作为输出

### 2.2 概率计算

在模型解码时，给定输入序列x和已生成序列y_{1:t}，模型预测下一个词y_{t+1}的条件概率为：

$$P(y_{t+1} | y_{1:t}, x) = \text{Softmax}(W \cdot h_t)$$

其中h_t是解码器在位置t的隐藏状态。

整个序列的分数为各条件概率的对数和（加法而非乘法，避免数值下溢）：

$$\text{Score}(y) = \sum_{t=1}^{T} \log P(y_t | y_{<t}, x)$$

---

## 3. 数学公式与推导

### 3.1 标准Beam Search

设beam width为k，词汇表大小为|V|，最大长度为T。

**每步扩展**：
对于每个当前候选序列s_i，生成所有可能的下一个词：

$$\text{candidates} = \{ (s_i + w, \text{score}(s_i) + \log P(w|s_i)) \mid w \in V \}$$

**Top-K选择**：
$$\text{beam} = \text{TopK}_{i=1}^{k} (\text{candidates})$$

### 3.2 Length Normalization

长序列的累积对数和通常较低，导致Beam Search偏向于短序列。使用长度归一化可以缓解这个问题：

$$\text{normalized\_score}(y) = \frac{1}{T^{\alpha}} \sum_{t=1}^{T} \log P(y_t | y_{<t}, x)$$

其中α是归一化因子，通常取0.6-0.7。

### 3.3 Coverage Penalty

有时序列中某些词可能被"忽略"，coverage penalty鼓励生成更覆盖的输出：

$$\text{coverage} = \sum_{w \in V} \log(\min(1, \sum_{t} \mathbb{1}_{y_t = w}))$$

$$\text{final\_score} = \text{normalized\_score} + \lambda \cdot \text{coverage}$$

### 3.4 复杂度分析

- **时间复杂度**：O(k × |V| × T × batch_size)
- **空间复杂度**：O(k × T)

其中k为beam width，|V|为词汇表大小，T为序列长度。

---

## 4. 训练过程讲解

### 4.1 在模型推理中使用Beam Search

Beam Search不是训练过程的一部分，而是用于**推理（Inference）阶段**的解码策略。训练时使用教师强制（Teacher Forcing）或负采样。

### 4.2 训练与推理的差异

| 阶段 | 解码策略 | 目的 |
|------|---------|------|
| 训练 | Teacher Forcing | 快速收敛，学习条件概率 |
| 推理 | Beam Search | 生成高质量序列 |

### 4.3 Beam Size选择

- **k=1**：等价于贪心搜索，质量最低但最快
- **k=3-5**：常见选择，平衡质量和速度
- **k=10+**：质量更高但计算成本增加

实际应用中，不同任务的最优k值不同：
- 机器翻译：k=4-5
- 文本摘要：k=3-4
- 对话生成：k=10+（需要多样性）

---

## 5. 应用场景

### 5.1 机器翻译

Beam Search是神经机器翻译（NMT）的标准解码策略。例如Google Translate、DeepL等商用系统都使用Beam Search。

```python
# 示例：翻译 "Hello world" -> "你好世界"
# Beam Search 保留多个候选，选择最优
```

### 5.2 文本摘要

生成式摘要模型使用Beam Search平衡摘要长度和质量。

### 5.3 语音识别

CTC解码或Attention-based ASR模型使用Beam Search生成文本。

### 5.4 对话生成

对话系统中，Beam Search生成回复，可结合diversity penalty增加多样性。

### 5.5 代码生成

Codex、CodeLLama等代码生成模型使用Beam Search解码。

---

## 6. 优缺点分析

### 6.1 优点

1. **质量较高**：相比贪心搜索，Beam Search能找到更好的序列
2. **效率适中**：只保留k个候选，控制搜索空间
3. **通用性强**：适用于各种序列生成任务
4. **实现简单**：相比A*等启发式搜索更容易实现

### 6.2 缺点

1. **仍是局部最优**：不能保证全局最优解
2. **需要调参**：beam size k需要手动调整
3. **偏向短序列**：未归一化时倾向于短输出
4. **重复生成**：可能生成重复的n-gram

### 6.3 改进方向

1. **Length Normalization**：缓解短序列偏好
2. **Coverage Penalty**：鼓励覆盖输入
3. **Diversity Penalty**：增加生成多样性
4. **Prefix-Aware Normalization**：针对重复问题

---

## 7. 调库实现（PyTorch完整代码）

### 7.1 使用Hugging Face Transformers

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class BeamSearchDecoder:
    """使用Hugging Face Transformers实现Beam Search解码"""
    
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        """
        初始化模型和分词器
        
        Args:
            model_name: Hugging Face模型名称
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
    
    def translate(self, source_text, beam_size=4, max_length=128, 
                  length_penalty=0.6, num_return_sequences=1):
        """
        使用Beam Search进行翻译
        
        Args:
            source_text: 源语言文本
            beam_size: Beam宽度
            max_length: 最大生成长度
            length_penalty: 长度惩罚因子
            num_return_sequences: 返回的候选序列数量
            
        Returns:
            翻译后的文本列表
        """
        # 编码输入
        inputs = self.tokenizer(
            source_text, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 生成参数
        generation_config = {
            "max_length": max_length,
            "num_beams": beam_size,  # Beam Search
            "length_penalty": length_penalty,
            "num_return_sequences": num_return_sequences,
            "early_stopping": True,
        }
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # 解码
        results = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        return results


# 示例使用
if __name__ == "__main__":
    # 注意：首次运行会下载模型，请确保网络畅通
    # decoder = BeamSearchDecoder(" Helsinki-NLP/opus-mt-zh-en")
    # result = decoder.translate("你好世界")
    # print(result)
    print("Beam Search decoder initialized")
    print("Usage:")
    print("  decoder = BeamSearchDecoder('model_name')")
    print("  results = decoder.translate('Hello world', beam_size=4)")
```

### 7.2 自定义Beam Search with PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralBeamSearchDecoder(nn.Module):
    """带神经网络模型的Beam Search解码器"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.start_token = 0  # <s>
        self.end_token = 1    # </s>
    
    def forward(self, x):
        """前向传播计算logits"""
        embed = self.embed(x)
        output, _ = self.lstm(embed)
        logits = self.output(output[:, -1, :])  # 最后一个位置的输出
        return logits
    
    def beam_search(self, encoder_hidden, encoder_outputs, 
                  beam_size=3, max_len=50, length_penalty=0.6):
        """
        Beam Search 解码算法
        
        Args:
            encoder_hidden: 编码器隐藏状态
            encoder_outputs: 编码器输出
            beam_size: Beam宽度
            max_len: 最大长度
            length_penalty: 长度惩罚因子
            
        Returns:
            best_sequences: 最优序列列表
            best_scores: 对应分数
        """
        batch_size = encoder_outputs.shape[0]
        
        # 初始化：每个样本一个beam
        # 当前序列（包含start token）
        current_sequences = torch.full(
            (batch_size, 1), 
            self.start_token, 
            dtype=torch.long,
            device=encoder_outputs.device
        )
        
        # 当前分数（初始为0）
        current_scores = torch.zeros(
            batch_size, 
            device=encoder_outputs.device
        )
        
        # 存储完成的序列
        done_sequences = [[] for _ in range(batch_size)]
        done_scores = [[] for _ in range(batch_size)]
        
        # 迭代生成
        for step in range(max_len):
            # 获取当前隐藏状态
            logits = self.forward(current_sequences)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 计算beam_size个候选
            log_probs_np = log_probs.cpu().numpy()
            
            # 存储此步的top-k候选
            all_candidates = []
            
            for batch_idx in range(batch_size):
                if len(done_sequences[batch_idx]) >= beam_size:
                    # 已经完成足够多的序列
                    continue
                
                # 为每个当前的beam扩展
                candidate_list = []
                
                for beam_idx in range(len(current_sequences[batch_idx])):
                    log_p = log_probs_np[batch_idx]
                    score = current_scores[batch_idx].item()
                    
                    if step > 0:
                        # 扩展得分，加上当前词的对数概率
                        current_score = score + log_p
                    else:
                        current_score = log_p
                    
                    # 排序获取top-k
                    topk_indices = np.argsort(current_score)[-beam_size:]
                    for idx in topk_indices:
                        new_seq = torch.cat([
                            current_sequences[batch_idx:batch_idx+1],
                            torch.tensor([[idx]], device=encoder_outputs.device)
                        ], dim=1)
                        new_score = current_score[idx]
                        candidate_list.append((new_seq, new_score))
                
                # 从所有候选中选择top-k
                if candidate_list:
                    sorted_candidates = sorted(
                        candidate_list, 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:beam_size]
                    all_candidates.append(sorted_candidates)
                else:
                    all_candidates.append([])
            
            # 更新当前序列
            new_sequences = []
            new_scores = []
            
            for batch_idx in range(batch_size):
                if not all_candidates[batch_idx]:
                    done_sequences[batch_idx].append(
                        current_sequences[batch_idx]
                    )
                    done_scores[batch_idx].append(
                        current_scores[batch_idx]
                    )
                    continue
                
                batch_candidates = all_candidates[batch_idx]
                selected = batch_candidates[:beam_size]
                new_sequences.extend([s[0] for s in selected])
                new_scores.extend([s[1] for s in selected])
            
            if new_sequences:
                current_sequences = torch.cat(new_sequences, dim=0)
                current_scores = torch.stack(new_scores)
            
            # 检查是否全部完成
            all_done = all(
                len(done_sequences[i]) >= beam_size 
                for i in range(batch_size)
            )
            if all_done:
                break
        
        # 返回最优序列
        return done_sequences, done_scores


# 示例使用
def demo_beam_search():
    """演示Beam Search的基本流程"""
    print("=== Beam Search 演示 ===\n")
    
    # 假设词汇表大小为10000，beam_size=3
    vocab_size = 10000
    beam_size = 3
    
    # 模拟logits（真实场景中来自模型）
    np.random.seed(42)
    
    # 模拟第一步的输出分布
    logits = np.random.randn(vocab_size)
    log_probs = np.log_softmax(logits, axis=0)
    
    # 选择top-k
    top_k_indices = np.argsort(log_probs)[-beam_size:]
    
    print(f"词汇表大小: {vocab_size}")
    print(f"Beam Size: {beam_size}")
    print(f"\nTop-{beam_size}候选词及其对数概率:")
    
    for idx in reversed(top_k_indices):
        print(f"  词ID {idx:5d}: log_prob = {log_probs[idx]:.4f}")
    
    # 长度归一化示例
    print(f"\n=== 长度归一化 ===")
    print(f"原始分数: {log_probs[top_k_indices[-1]]:.4f}")
    print(f"归一化分数(α=0.6, T=10): {log_probs[top_k_indices[-1]]/(10**0.6):.4f}")
    
    return


if __name__ == "__main__":
    demo_beam_search()
```

---

## 8. 手工代码实现

```python
import numpy as np
from typing import List, Tuple, Optional

class SimpleBeamSearch:
    """纯Python实现的Beam Search解码器"""
    
    def __init__(self, vocab_size: int, beam_size: int = 3, 
                 max_length: int = 50, length_penalty: float = 0.6):
        """
        初始化Beam Search解码器
        
        Args:
            vocab_size: 词汇表大小
            beam_size: Beam宽度
            max_length: 最大生成长度
            length_penalty: 长度惩罚因子
        """
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
    
    def _get_next_token_probs(self, current_seq: List[int]) -> np.ndarray:
        """
        模拟获取下一个token的概率分布
        真实场景中这里会调用神经网络的forward
        
        Args:
            current_seq: 当前序列
            
        Returns:
            下一个token的概率分布
        """
        # 模拟：基于序列内容计算不同的分布
        np.random.seed(sum(current_seq) % 1000 + 1)
        probs = np.random.randn(self.vocab_size)
        probs = np.exp(probs - np.max(probs))  # 数值稳定的softmax
        probs = probs / np.sum(probs)
        return probs
    
    def _normalize_score(self, score: float, length: int) -> float:
        """
        长度归一化分数
        
        Args:
            score: 原始分数
            length: 序列长度
            
        Returns:
            归一化分数
        """
        return score / (length ** self.length_penalty)
    
    def search(self) -> Tuple[List[int], float]:
        """
        执行Beam Search
        
        Returns:
            最优序列和分数
        """
        # 初始化：只有一个开始序列
        # 每个元素是 (序列, 累计分数)
        beams = [([0], 0.0)]  # 0是start token
        
        completed = []
        
        for step in range(self.max_length):
            # 存储此步的所有候选
            candidates = []
            
            for seq, score in beams:
                # 如果遇到结束标记，停止扩展
                if seq[-1] == 1:  # 1是end token
                    # 归一化并保存
                    norm_score = self._normalize_score(score, len(seq))
                    completed.append((seq, norm_score))
                    continue
                
                # 获取下一个token的概率
                probs = self._get_next_token_probs(seq)
                
                # 取top-k
                top_k_indices = np.argsort(probs)[-self.beam_size:]
                
                for idx in top_k_indices:
                    # 累加对数概率（使用log避免下溢）
                    new_score = score + np.log(probs[idx] + 1e-10)
                    new_seq = seq + [idx]
                    candidates.append((new_seq, new_score))
            
            if not candidates:
                break
            
            # 从所有候选中选择top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:self.beam_size]
            
            # 检查是否所有beam都完成了
            done_count = sum(1 for seq, _ in beams if seq[-1] == 1)
            if done_count == self.beam_size:
                break
        
        # 收集所有完成的序列
        for seq, score in beams:
            if seq[-1] == 1:
                norm_score = self._normalize_score(score, len(seq))
                completed.append((seq, norm_score))
        
        if not completed:
            # 如果没有完成的，使用未完成的最好的
            best = max(beams, key=lambda x: self._normalize_score(x[1], len(x[0])))
            return best[0], self._normalize_score(best[1], len(best[0]))
        
        # 返回最优
        best = max(completed, key=lambda x: x[1])
        return best[0], best[1]


def demo():
    """演示Beam Search"""
    print("=== 简单Beam Search演示 ===\n")
    
    # 创建解码器
    beam_search = SimpleBeamSearch(
        vocab_size=1000,
        beam_size=3,
        max_length=10,
        length_penalty=0.6
    )
    
    # 运行搜索
    # result, score = beam_search.search()
    # print(f"生成序列: {result}")
    # print(f"分数: {score:.4f}")
    
    # 展示Beam Search的思考过程
    print("Beam Search流程示例：")
    print("-" * 40)
    print("Step 0: 初始序列 [START]")
    print("         -> 扩展到top-3候选")
    print("Step 1: 保留3个序列")
    print("         -> 继续扩展...")
    print("Step t: 遇到 END token，停止扩展")
    print("       -> 选择归一化分数最高的序列")
    print("-" * 40)
    print("\n注意：实际使用时需要调用真实模型获取概率")


if __name__ == "__main__":
    demo()
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_beam_search():
    """可视化Beam Search过程"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Beam Search 分析', fontsize=14, fontweight='bold')
    
    # 1. Beam Size vs 质量
    ax1 = axes[0, 0]
    beam_sizes = [1, 2, 3, 4, 5, 10]
    # 模拟BLEU分数
    bleu_scores = [0.25, 0.32, 0.38, 0.41, 0.43, 0.45]
    ax1.plot(beam_sizes, bleu_scores, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Beam Size', fontsize=10)
    ax1.set_ylabel('BLEU Score', fontsize=10)
    ax1.set_title('Beam Size vs 生成质量', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(beam_sizes)
    
    # 2. 序列分数分布
    ax2 = axes[0, 1]
    scores = np.random.exponential(scale=0.5, size=1000)
    ax2.hist(scores, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=np.mean(scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(scores):.2f}')
    ax2.set_xlabel('Log Probability', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('序列分数分布', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Length Normalization效果
    ax3 = axes[1, 0]
    lengths = np.arange(5, 50, 5)
    raw_scores = -lengths * 0.5  # 原始分数（负对数）
    norm_scores_short = -lengths * 0.5 / (lengths ** 0.6)  # 归一化（α=0.6）
    norm_scores_medium = -lengths * 0.5 / (lengths ** 0.7)
    norm_scores_long = -lengths * 0.5 / (lengths ** 0.8)
    
    ax3.plot(lengths, raw_scores, 'b-o', label='Raw', linewidth=2)
    ax3.plot(lengths, norm_scores_short, 'g-s', label='α=0.6', linewidth=2)
    ax3.plot(lengths, norm_scores_medium, 'r-^', label='α=0.7', linewidth=2)
    ax3.plot(lengths, norm_scores_long, 'm-d', label='α=0.8', linewidth=2)
    ax3.set_xlabel('Sequence Length', fontsize=10)
    ax3.set_ylabel('Normalized Score', fontsize=10)
    ax3.set_title('长度归一化效果', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Beam Search vs 其他方法
    ax4 = axes[1, 1]
    methods = ['Greedy', 'Beam-3', 'Beam-5', 'Beam-10', 'Sampling']
    quality = [0.65, 0.78, 0.82, 0.86, 0.72]
    speed = [1.0, 0.8, 0.6, 0.4, 0.9]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, quality, width, label='Quality', color='blue', alpha=0.7)
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, speed, width, label='Speed', color='orange', alpha=0.7)
    
    ax4.set_xlabel('Decoding Method', fontsize=10)
    ax4.set_ylabel('Quality Score', fontsize=10)
    ax4_twin.set_ylabel('Relative Speed', fontsize=10)
    ax4.set_title('解码方法对比', fontsize=11)
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, rotation=15)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('beam_search_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n图表已保存到 beam_search_analysis.png")


if __name__ == "__main__":
    visualize_beam_search()
```

**输出图表说明**：

1. **Beam Size vs 质量**：展示beam size增大时质量提升的趋势（边际递减）
2. **序列分数分布**：展示Beam Search候选序列的分数分布
3. **长度归一化效果**：展示不同α值对长序列的影响
4. **解码方法对比**：对比各种解码方法的质量和速度

---

## 10. 模型评估

```python
import numpy as np
from typing import List, Dict
from collections import Counter

def evaluate_beam_search(generated_sequences: List[List[int]], 
                       reference: List[int],
                       vocab_size: int = 10000) -> Dict[str, float]:
    """
    评估Beam Search生成质量
    
    Args:
        generated_sequences: 生成的序列列表
        reference: 参考序列
        vocab_size: 词汇表大小
        
    Returns:
        评估指标字典
    """
    results = {}
    
    # 1. BLEU分数（简化版）
    def simple_bleu(candidate, reference, n=4):
        """简化的BLEU计算"""
        scores = []
        for i in range(1, min(n, min(len(candidate), len(reference)) + 1):
            ref_ngrams = Counter([tuple(reference[j:j+i]) for j in range(len(reference)-i+1)])
            cand_ngrams = Counter([tuple(candidate[j:j+i]) for j in range(len(candidate)-i)])
            
            matches = sum((ref_ngrams & cand_ngrams).values())
            total = sum(cand_ngrams.values())
            
            if total > 0:
                precision = matches / total
                score = precision * (0.5 ** (1/i))
                scores.append(score)
        
        return np.exp(np.mean(scores)) if scores else 0
    
    # 2. 重复率
    def repetition_rate(sequence):
        """计算n-gram重复率"""
        ngrams = [tuple(sequence[i:i+3]) for i in range(len(sequence)-3)]
        if not ngrams:
            return 0
        return 1 - len(set(ngrams)) / len(ngrams)
    
    # 3. 长度惩罚
    avg_length = np.mean([len(seq) for seq in generated_sequences])
    length_penalty = np.exp(min(0, (avg_length - len(reference)) / len(reference)))
    
    results['avg_length'] = avg_length
    results['length_penalty'] = length_penalty
    results['repetition_rate'] = repetition_rate(generated_sequences[0])
    
    # 计算多样性
    unique_sequences = len(set(tuple(seq) for seq in generated_sequences))
    results['diversity'] = unique_sequences / len(generated_sequences)
    
    return results


def compare_beam_sizes():
    """对比不同Beam Size的效果"""
    print("=== Beam Size对比实验 ===\n")
    
    beamconfigs = [
        {'beam_size': 1, 'name': 'Greedy'},
        {'beam_size': 3, 'name': 'Beam-3'},
        {'beam_size': 5, 'name': 'Beam-5'},
        {'beam_size': 10, 'name': 'Beam-10'},
    ]
    
    print(f"{'方法':<12} {'BLEU':<10} {'长度':<10} {'重复率':<10} {'多样性':<10}")
    print("-" * 55)
    
    for config in beamconfigs:
        # 模拟结果
        bleu = 0.3 + 0.05 * config['beam_size']
        length = 15 + config['beam_size']
        rep_rate = max(0, 0.3 - 0.02 * config['beam_size'])
        div = min(1, 0.5 + 0.05 * config['beam_size'])
        
        print(f"{config['name']:<12} {bleu:<10.3f} {length:<10.1f} {rep_rate:<10.3f} {div:<10.3f}")
    
    print("\n结论：")
    print("  - Beam Size增大 → BLEU分数提高")
    print("  - Beam Size增大 → 重复率降低")
    print("  - Beam Size增大 → 多样性增加")
    print("  - Beam Size增大 → 计算成本增加")


if __name__ == "__main__":
    compare_beam_sizes()
```

---

## 11. 常见问题与易错点

### 11.1 数值不稳定

**问题**：累乘多个小数导致下溢
**解决**：使用对数概率，加法代替乘法

```python
# 错误：累乘
score = 1.0
for p in probs:
    score *= p  # 会下溢

# 正确：使用对数
score = 0.0
for p in probs:
    score += np.log(p + 1e-10)  # 加法代替乘法
```

### 11.2 偏向短序列

**问题**：未归一化时倾向于生成短序列
**解决**：使用长度归一化

```python
# 长度归一化
normalized_score = raw_score / (length ** alpha)  # alpha通常取0.6-0.7
```

### 11.3 重复生成

**问题**：生成重复的n-gram
**解决**：添加coverage penalty或blocking

```python
def block_repeated_ngrams(sequence, n=3, blocklist=None):
    """阻止重复的n-gram"""
    ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n)]
    for ng in ngrams:
        if ng in blocklist:
            return True
    return False
```

### 11.4 Beam Size选择

**问题**：不知道选择多大的beam size
**建议**：
- 机器翻译：k=4-5
- 文本摘要：k=3-4
- 对话生成：k=10+
- 实时系统：k=1（贪心）

### 11.5 结束标记处理

**问题**：遇到EOS后不停止扩展
**解决**：标记已完成，不继续扩展

```python
# 正确的处理方式
if seq[-1] == EOS_TOKEN:
    completed.append(seq)
    continue  # 不再扩展
```

---

## 12. 学习总结

**Beam Search核心要点**：

1. **平衡搜索空间**：每步保留Top-K个候选，在质量和效率间平衡
2. **对数概率**：使用对数避免数值下溢
3. **长度归一化**：防止偏向短序列
4. **Beam Size调参**：不同任务最优值不同，需实验确定
5. **应用广泛**：机器翻译、文本生成、语音识别等任务的核心组件

**为什么Beam Search有效**：
- 相比贪心搜索，考虑更多候选，避免局部最优
- 相比穷举搜索，只保留有限候选，控制计算量
- 简单有效，是序列生成任务的主流解码策略

---

## 13. 练习题与思考题与思考题

### 13.1 选择题

1. Beam Search中，当beam_size=1时，等价于哪种方法？
   - A) 贪心搜索
   - B) A*搜索
   - C) 维特比算法
   - D) 蒙特卡洛搜索
   
   **答案：A**（beam_size=1即贪心搜索，每步只保留最优解）

2. 长度归一化的主要目的是？
   - A) 加速计算
   - B) 防止数值下溢
   - C) 防止偏向短序列
   - D) 增加多样性
   
   **答案：C**

3. 使用对数概率的好处是？
   - A) 提高精度
   - B) 避免数值下溢
   - C) 简化计算
   - D) 便于并行
   
   **答案：B**

### 13.2 简答题

1. **问题**：简述Beam Search的基本流程。
   
   **答案**：
   - 初始化：将start token作为唯一的beam
   - 迭代：对每个beam扩展所有可能的下一个词
   - 剪枝：选择累计分数最高的K个beam
   - 终止：所有beam都结束或达到最大长度
   - 输出：选择分数最高的完整序列

2. **问题**：Beam Search和贪心搜索的本质区别是什么？
   
   **答案**：贪心搜索只保留每一步的最优解，而Beam Search保留Top-K个最优解。贪心搜索是局部最优，Beam Search在每步考虑更多选择，更可能找到全局较优的解。

3. **问题**：为什么Beam Search需要长度归一化？
   
   **答案**：因为分数是累加的对数概率，序列越长，累积分数越低。未归一化时，会偏向于生成短序列。长度归一化通过除以长度幂来抵消这种偏见。

### 13.3 编程题

1. **题目**：实现一个简单的Beam Search解码器，处理一个模拟的词汇表，并输出top-k候选。
   
   ```python
   import numpy as np
   
   def beam_search_step(logits, current_scores, beam_size):
       """执行一步Beam Search扩展"""
       # 计算log_prob
       log_probs = np.log_softmax(logits, axis=-1)
       
       # 扩展所有候选
       new_scores = current_scores[:, None] + log_probs
       
       # 展平并排序
       new_scores_flat = new_scores.reshape(-1)
       topk_indices = np.argsort(new_scores_flat)[-beam_size:]
       
       return topk_indices, new_scores_flat[topk_indices]
   
   # 测试
   logits = np.random.randn(5, 10000)  # batch=5, vocab=10000
   current_scores = np.zeros(5)
   topk_indices, topk_scores = beam_search_step(logits, current_scores, beam_size=3)
   print(f"Top-3 indices: {topk_indices}")
   print(f"Top-3 scores: {topk_scores}")
   ```

### 13.4 思考题

1. **问题**：Beam Search能否保证找到最优解？如果不能，有什么方法可以找到最优解？
   
   **思考**：Beam Search只保留Top-K个候选，不能保证全局最优。维特比算法（Viterbi）可以找到全局最优，但需要枚举所有路径，适用于短序列和小词汇表。A*搜索结合启发式可以更高效地找到最优解。

2. **问题**：在实际应用中，如何决定beam_size的大小？
   
   **思考**：需要权衡质量和速度。可以通过实验，在验证集上比较不同beam_size的BLEU分数，找到质量-速度的平衡点。也要考虑实际应用场景：实时系统可能需要小的beam_size，离线系统可以用更大的beam_size。

3. **问题**：Beam Search有哪些改进方向？
   
   **思考**：
   - 长度归一化（Length Normalization）
   - 覆盖惩罚（Coverage Penalty）
   - 多样性惩罚（Diversity Penalty）
   - 前缀感知归一化（Prefix-aware Normalization）
   - 噪声链蒙特卡洛（Noisy Channel Model）

---

## 14. 学习路径建议建议

### 14.1 入门路径

1. **理解序列生成问题** → 了解Seq2Seq模型
2. **贪心搜索** → 理解最简单的解码策略
3. **Beam Search** → 掌握平衡质量和效率的方法
4. **变体** → 学习长度归一化等改进

### 14.2 进阶路径

1. **高级解码策略** → 了解采样、多样性解码
2. **神经机器翻译** → 学习Transformer、NMT系统
3. **实际应用** → 使用Hugging Face实现生产级系统
4. **优化技术** → 模型压缩、推理加速

### 14.3 推荐学习资源

**论文**：
- "Neural Machine Translation and Sequence to Sequence Learning" (2017)
- "Achieving Human Parity on Machine Translation" (2016)

**书籍**：
- 《深度学习：核心技术与案例分析》
- 《Speech and Language Processing》

**实践框架**：
- Hugging Face Transformers
- fairseq
- OpenNMT

### 14.4 相关算法链接

**前置算法**：
- 动态规划
- 贪心搜索
- Softmax函数

**后续扩展**：
- Length Normalization
- Coverage Penalty
- Diverse Beam Search
- Approximate Search

---

*学习Beam Search是掌握序列生成任务的关键第一步，它不仅是一种解码策略，更是理解自然语言生成的基础。*

## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述Beam_Search的核心思想及适用场景。
<details><summary>参考答案</summary>
Beam_Search通过数据驱动学习输入到输出的映射，适用于强化学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出Beam_Search的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现Beam_Search核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. Beam_Search在什么情况下会失效？
2. 训练数据很少时，Beam_Search还能有效工作吗？
3. 如何将Beam_Search与其他方法结合？

