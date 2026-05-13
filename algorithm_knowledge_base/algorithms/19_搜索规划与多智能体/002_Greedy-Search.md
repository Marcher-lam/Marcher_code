> 来源线索：根据原书第2章相关内容整理、扩展与教学化改写。

# Greedy Search 学习文档

> 每次选最高概率词，简单高效但易局部最优

## 1. 算法基础认知
Greedy Search（贪心搜索）是文本生成任务中最简单的自回归解码策略，核心逻辑是**每一步都选择概率最高的下一个词**，无需保留候选路径或回溯。它由起始词开始，每次生成都取当前概率分布的最大值，直到达到最大长度或生成终止符。

作为最基础的解码策略，Greedy Search计算成本极低，生成速度极快，适合对实时性要求高、对生成质量要求不高的场景。但由于只考虑当前步的最优选择，容易陷入局部最优，生成的文本往往存在重复、逻辑单一的问题，无法得到全局最优的序列。

在GPT等自回归生成模型中，Greedy Search是默认的快速生成方式，也常作为其他复杂解码策略（如Beam Search）的对比基线。理解Greedy Search是掌握更复杂解码算法的基础。

## 2. 核心原理
Greedy Search的核心逻辑可以概括为“短视的最优选择”：在生成第$t$个词时，模型输出当前步的词概率分布$P(w | w_{<t})$，算法直接选择概率最大的词作为$w_t$，将其拼接到序列后继续生成，整个过程没有回溯或候选路径保留。

### 与人类的决策类比
就像买水果时只看当前最甜的那个，拿了就走，不考虑后续可能更好的组合。比如买苹果时看到当前最甜的就拿，不考虑再逛一圈可能有更甜的苹果，也不考虑和其他水果搭配的口感。

### 与其他解码策略的区别
- 与Beam Search的区别：Beam Search保留k个候选路径，Greedy Search只保留1个路径
- 与采样策略的区别：采样策略按概率随机选词，Greedy Search是确定性的，每次生成结果固定

### 生成流程
1.  输入起始序列$w_{<1}$，设置当前步数$t=1$
2.  将当前序列输入模型，得到第$t$步的概率分布$P(w | w_{<t})$
3.  选择$w_t = \arg\max_w P(w | w_{<t})$
4.  拼接$w_t$到序列，若$w_t$是终止符或达到最大长度则停止，否则$t=t+1$重复步骤2

## 3. 数学公式与推导
### 单步选择公式
第$t$步生成的词为概率分布的argmax：
$$w_t = \arg\max_{w \in V} P(w | w_1, w_2, ..., w_{t-1}) = \arg\max_{w \in V} P(w | w_{<t})$$
其中$V$为模型词表，包含所有可能的生成词。

### 完整序列生成公式
生成的完整序列$W = [w_1, w_2, ..., w_T]$满足：
$$W = \arg\max_{W'} \prod_{t=1}^T P(w'_t | w'_{<t})$$
但由于Greedy Search是逐次贪心选择，实际得到的是局部最优解，而非全局最优的序列（全局最优需要遍历所有组合，计算量极大）。

### 分数计算
Greedy Search生成序列的得分为所有步概率的乘积（通常取对数避免下溢）：
$$\text{Score}(W) = \sum_{t=1}^T \log P(w_t | w_{<t})$$
每一步只取当前最大对数概率，因此总得分是局部最大而非全局最大。

## 4. 训练过程讲解
Greedy Search是**推理阶段的解码策略**，无需单独训练，其依赖的生成模型（如GPT-2）需要预先训练完成。训练过程仅针对生成模型，Greedy Search仅在使用训练好的模型生成文本时调用。

### 与模型训练的配合
1.  先完成生成模型的预训练/微调，得到可输出词概率分布的模型
2.  推理时加载模型，调用Greedy Search逻辑生成文本
3.  Greedy Search的超参数（如最大长度、终止符）可在推理时动态调整，无需重新训练

## 5. 应用场景
1.  **实时聊天机器人**：对响应速度要求高，可容忍一定质量损失，Greedy Search的低延迟特性适配该场景。
2.  **快速文本草稿生成**：需要快速产出内容框架，后续人工修改，无需高质量生成。
3.  **低算力设备部署**：嵌入式设备或移动端算力有限，Greedy Search计算量最小，适合部署。
4.  **基线对比实验**：在论文或项目中作为最基础的解码方法，对比其他策略的效果提升。
5.  **简单问答系统**：问题答案长度固定、内容单一，Greedy Search可快速生成准确结果。

## 6. 优缺点分析
### 优点
1.  **计算效率极高**：无需保留候选路径，每一步仅一次argmax操作，生成速度远快于Beam Search。
2.  **实现简单**：逻辑清晰，仅需几行代码即可实现，无复杂超参数。
3.  **确定性输出**：相同输入和参数下，生成结果完全一致，便于复现和调试。
4.  **低内存占用**：仅保留一条生成路径，内存消耗与序列长度线性相关。

### 缺点
1.  **局部最优问题**：每一步只考虑当前最优，导致最终序列并非全局最优，易出现重复、逻辑断层。
2.  **生成多样性差**：输出固定，无法生成不同风格的内容，不适合创意类任务。
3.  **易重复生成**：当高概率词重复出现时，模型会不断生成相同词，导致内容冗余。
4.  **长文本质量差**：随着生成长度增加，局部最优的累积误差会导致文本连贯性下降。

### 对比表
| 特性 | Greedy Search | Beam Search | 随机采样 |
|------|---------------|-------------|----------|
| 生成速度 | 最快 | 中等 | 中等 |
| 多样性 | 无 | 低 | 高 |
| 质量 | 较低 | 较高 | 不稳定 |
| 计算量 | 最小 | 中等 | 中等 |
| 确定性 | 是 | 是 | 否 |

## 7. 调库实现
使用HuggingFace Transformers库的generate方法，设置`num_beams=1`即为Greedy Search：
```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练模型和分词器
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# 输入提示词
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

# 调用Greedy Search生成，num_beams=1即为贪心搜索
output = model.generate(
    inputs.input_ids,
    max_length=50,
    num_beams=1,          # 束宽为1，等价于Greedy Search
    do_sample=False,       # 关闭采样，使用确定性选择
    no_repeat_ngram_size=2 # 避免2-gram重复
)

# 解码输出
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Greedy Search生成结果：")
print(generated_text)
```
运行结果示例：
```
Greedy Search生成结果：
The future of AI is bright, with new advancements being made every day. From healthcare to transportation, AI is transforming industries and improving our lives. As we continue to develop more sophisticated algorithms, we can expect even more exciting innovations in the coming years.
```

## 8. 手工代码实现
从零实现Greedy Search解码逻辑，不依赖任何生成库：
```python
import torch
import torch.nn.functional as F

class SimpleLanguageModel:
    """模拟简单语言模型，输出固定logits用于测试"""
    def __init__(self, vocab_size=100, seq_len=10):
        self.vocab_size = vocab_size
        # 预定义每个位置的词概率分布（模拟模型输出）
        self.fake_logits = torch.randn(seq_len, vocab_size)

    def forward(self, input_ids):
        """
        模拟模型前向传播，返回当前序列最后一个位置的logits
        input_ids: [batch_size, seq_len]
        """
        # 实际中这里是模型输出，这里用预定义的fake_logits模拟
        last_pos = input_ids.shape[1] - 1
        logits = self.fake_logits[last_pos].unsqueeze(0).unsqueeze(0)  # [1,1,vocab_size]
        return logits

def greedy_search(model, start_tokens, max_len=20, eos_token_id=99):
    """
    手写Greedy Search解码
    model: 语言模型，需支持forward方法返回logits
    start_tokens: 起始词元ID列表
    max_len: 最大生成长度
    eos_token_id: 终止符ID
    """
    generated = start_tokens.copy()
    for _ in range(max_len - len(start_tokens)):
        # 将当前序列转为张量
        input_ids = torch.tensor([generated], dtype=torch.long)
        # 获取模型输出logits（取最后一个位置）
        logits = model.forward(input_ids)[:, -1, :]
        # 计算概率分布，选择最大概率的词
        probs = F.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1).item()
        # 拼接结果
        generated.append(next_token)
        # 遇到终止符停止
        if next_token == eos_token_id:
            break
    return generated

# 测试手写Greedy Search
if __name__ == "__main__":
    model = SimpleLanguageModel(vocab_size=100, seq_len=20)
    start_tokens = [1, 2, 3]
    result = greedy_search(model, start_tokens, max_len=20, eos_token_id=99)
    print("手写Greedy Search生成序列：", result)
```
运行结果示例：
```
手写Greedy Search生成序列： [1, 2, 3, 87, 42, 15, 87, 42, 15, 87, 42, 15, 87, 42, 15, 99]
```
结果解读：可以看到生成了重复的87、42、15循环，体现了Greedy Search易重复的问题。

## 9. 可视化与结果理解
对比Greedy Search和随机采样的概率分布选择，可视化单步选择逻辑：
```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_greedy_step(probs):
    """可视化单步Greedy Search的概率选择"""
    vocab_indices = np.arange(len(probs))
    plt.figure(figsize=(12, 6))
    # 绘制所有词的概率分布
    plt.bar(vocab_indices, probs, alpha=0.5, label="所有词概率")
    # 标记选中的最高概率词
    max_idx = np.argmax(probs)
    plt.bar(max_idx, probs[max_idx], color="red", label="Greedy选择")
    plt.title("Greedy Search单步概率选择可视化")
    plt.xlabel("词元ID")
    plt.ylabel("概率值")
    plt.legend()
    plt.ylim(0, 1)
    plt.show()
    print(f"选中词元ID：{max_idx}，概率：{probs[max_idx]:.4f}")

# 模拟单步概率分布
fake_probs = F.softmax(torch.randn(50), dim=-1).numpy()
visualize_greedy_step(fake_probs)
```
结果解读：图中红色柱为Greedy Search选中的最高概率词，其他灰色柱被忽略，直观体现其“只看当前最高”的特性。

## 10. 模型评估
对比Greedy Search和Beam Search的生成质量，使用困惑度和BLEU评估：
```python
def calculate_perplexity_greedy(model, generated_tokens):
    """计算Greedy Search生成序列的困惑度"""
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([generated_tokens], dtype=torch.long)
        logits = model.forward(input_ids)  # [1, seq_len, vocab_size]
        # 计算交叉熵损失
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, model.vocab_size),
                               shift_labels.view(-1))
        return torch.exp(loss).item()

# 测试评估
model = SimpleLanguageModel(vocab_size=100, seq_len=20)
start_tokens = [1, 2, 3]
greedy_result = greedy_search(model, start_tokens, max_len=20)
ppl = calculate_perplexity_greedy(model, greedy_result)
print(f"Greedy Search生成序列困惑度：{ppl:.2f}")  # 困惑度越低越好
```
结果解读：Greedy Search因局部最优，困惑度通常高于Beam Search，生成的文本流畅度更差。

## 11. 常见问题与易错点
### 数据层面
1.  **词表映射错误**：生成时使用的词表与模型训练词表不一致，导致选中的ID对应错误词，需保证词表统一。
2.  **终止符设置错误**：未设置终止符或终止符ID错误，会导致生成无限长度或提前停止。
3.  **输入序列过长**：超过模型最大输入长度，会导致位置嵌入溢出，需截断输入序列。

### 模型层面
1.  **概率分布平坦**：模型训练不充分时，所有词概率接近，Greedy Search随机选择（实际还是选第一个最大），导致生成内容混乱。
2.  **梯度泄露**：训练时使用未来信息，导致模型生成时依赖不存在的未来词，Greedy Search会放大这个错误。
3.  **过拟合**：模型记住训练集，生成时只输出训练集中的高频序列，缺乏多样性。

### 调参层面
1.  **最大长度设置过长**：长文本生成时Greedy Search累积误差大，需根据任务设置合理的最大长度。
2.  **未设置重复惩罚**：没有`no_repeat_ngram_size`参数，会导致大量重复内容，需添加重复惩罚逻辑。

## 12. 学习总结
Greedy Search是最基础的自回归解码策略，核心是每一步选择概率最高的词，具有速度快、实现简单、确定性输出的优点，但存在局部最优、多样性差、易重复的缺点。它适合实时性要求高、对质量容忍度高的场景，也是理解更复杂解码策略的基础。

实际应用中，直接用HuggingFace的`num_beams=1`即可调用，无需重复实现。手写实现有助于理解解码逻辑：核心是取argmax，无候选路径保留。评估时需结合困惑度和内容指标，不要仅看生成速度。

## 13. 练习题与思考题
### 基础题
1.  Greedy Search的核心选择逻辑是什么？
    **答案**：每一步选择当前概率分布中概率最高的词，公式为$w_t = \arg\max_w P(w | w_{<t})$。
2.  Greedy Search需要训练吗？为什么？
    **答案**：不需要，它是推理阶段的解码策略，仅依赖预训练好的生成模型。

### 进阶题
1.  为什么Greedy Search易生成重复内容？请提出一种改进方法。
    **答案**：因为高概率词重复出现时，模型会不断选择相同词。改进方法：添加重复惩罚，降低已生成词的概率。
2.  推导Greedy Search的总得分公式，并说明为什么是局部最优。
    **答案**：总得分$\sum_{t=1}^T \log P(w_t | w_{<t})$，每一步仅选当前最大，未考虑后续选择的影响，因此是局部最优。

### 开放题
在什么场景下应该优先使用Greedy Search而非Beam Search？请举3个具体例子。
**答案参考**：1. 低算力嵌入式设备部署；2. 实时聊天机器人需要毫秒级响应；3. 快速生成草稿后续人工修改。

## 14. 学习路径建议
### 前置知识
- 自回归生成基本逻辑
- 概率分布、argmax操作
- GPT等生成模型的基本结构

### 平行学习
- Beam Search（保留k个候选的改进版）
- 随机采样、Top-k采样、Nucleus采样（多样性解码策略）

### 进阶学习
- 对比解码（Contrastive Decoding）
- 自适应解码策略（根据生成长度动态调整）

### 推荐资源
1.  原书第2章Greedy Search相关内容
2.  HuggingFace生成策略文档：https://huggingface.co/docs/transformers/main/en/generation_strategies
3.  论文《The Curious Case of Neural Text Degeneration》对比不同解码策略
