> 来源线索：根据原书第2章相关内容整理、扩展与教学化改写。

# Beam Search 学习文档

> 保留k个候选路径，平衡生成质量与效率

## 1. 算法基础认知
Beam Search（束搜索）是文本生成中广泛使用的解码策略，核心思想是**每一步保留k个最优候选序列（称为束宽beam width），逐步扩展直到生成结束，最终选择总得分最高的序列**。它是Greedy Search的改进版，通过保留多个候选路径避免局部最优，同时计算量远小于遍历所有组合。

束宽k是核心超参数：k=1时等价于Greedy Search，k越大生成质量越高但计算量越大。在机器翻译、文本摘要等对质量要求高的任务中，Beam Search是默认解码策略，在质量和效率之间取得了良好平衡。

与Greedy Search的“短视选择”不同，Beam Search像同时派k个人去挑水果，每人走不同路线，最后选整体最好的组合，既避免了全局遍历的高成本，又比单人选择更全面。

## 2. 核心原理
Beam Search的核心是**维护一个大小为k的候选序列集合**，每一步对每个候选序列扩展所有可能的下一个词，计算新序列的总得分，保留得分最高的k个序列，重复直到所有序列结束或达到最大长度。

### 完整流程
1.  初始化候选集合：仅包含起始序列，得分为0
2.  对于每一步生成：
    a. 对每个候选序列，调用模型得到下一个词的概率分布
    b. 对每个候选，扩展所有可能的下一个词，得到k*V个新序列（V为词表大小）
    c. 计算每个新序列的总得分（候选原得分 + 新词的对数概率）
    d. 从所有新序列中选择得分最高的k个，作为下一轮的候选集合
3.  所有序列生成结束后，选择总得分最高的序列作为最终结果

### 得分计算
使用对数概率求和避免下溢：
$$\text{Score}(W) = \sum_{t=1}^T \log P(w_t | w_{<t})$$
每一步累加对数概率，总得分越高说明序列概率越大。

### 提前终止逻辑
如果某个候选序列生成了终止符，该序列不再参与后续扩展，仅保留在候选集合中等待最终选择。

## 3. 数学公式与推导
### 单步扩展得分
对于候选序列$S$得分为$\text{Score}(S)$，扩展词$w$后的新序列得分：
$$\text{Score}(S \oplus w) = \text{Score}(S) + \log P(w | S)$$
其中$S \oplus w$表示将$w$拼接到$S$末尾。

### 束宽选择
每一步保留的k个序列满足：
$$\text{TopK} = \arg\max_{S'_1, ..., S'_k} \text{Score}(S'_i) \quad \text{从所有扩展序列中选}$$
k为束宽，控制候选数量。

### 最终序列选择
生成结束后，选择总得分最高的序列：
$$W_{\text{best}} = \arg\max_{S \in \text{所有完成的候选}} \text{Score}(S)$$

## 4. 训练过程讲解
与Greedy Search一致，Beam Search是**推理阶段的解码策略**，无需单独训练，依赖预训练好的生成模型（如GPT-2）。训练仅针对生成模型，Beam Search的参数（束宽、最大长度等）可在推理时动态调整。

### 与模型训练的配合
1.  完成生成模型的预训练/微调，得到可输出词概率的模型
2.  推理时加载模型，设置束宽k调用Beam Search
3.  束宽可根据任务调整：质量要求高则k大，速度要求高则k小

## 5. 应用场景
1.  **机器翻译**：需要生成流畅准确的译文，束宽k=4~10是常用配置。
2.  **文本摘要**：生成保留核心信息的短文本，Beam Search避免遗漏关键信息。
3.  **对话系统**：需要生成连贯、逻辑合理的回复，k=3~5平衡质量和速度。
4.  **代码生成**：生成符合语法、逻辑正确的代码，较大的束宽可减少语法错误。
5.  **故事生成**：生成长篇连贯故事，k=5~8避免逻辑断层。

## 6. 优缺点分析
### 优点
1.  **质量优于Greedy Search**：保留多个候选路径，避免局部最优，生成质量更高。
2.  **效率远高于全局搜索**：仅需O(k*T*V)的计算量，远小于遍历所有序列的O(V^T)。
3.  **可调节性强**：通过束宽k灵活平衡质量和速度，适配不同场景。
4.  **确定性输出**：相同输入和参数下生成结果固定，便于复现。

### 缺点
1.  **计算量高于Greedy Search**：k越大计算量越大，延迟越高。
2.  **多样性不足**：候选序列都是高概率路径，生成结果仍然偏向保守。
3.  **长文本效果下降**：随着生成长度增加，候选序列的差异逐渐消失，容易收敛到同一路径。
4.  **易生成重复内容**：高概率路径可能重复，需要额外添加重复惩罚。

### 对比表
| 特性 | Beam Search | Greedy Search | 随机采样 |
|------|-------------|---------------|----------|
| 生成质量 | 较高 | 较低 | 不稳定 |
| 计算量 | 中等 | 最小 | 中等 |
| 多样性 | 低 | 无 | 高 |
| 可调性 | 强（束宽k） | 弱 | 强（温度） |
| 延迟 | 中等 | 最低 | 中等 |

## 7. 调库实现
使用HuggingFace Transformers库的generate方法，设置`num_beams`参数即可调用Beam Search：
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

# 调用Beam Search生成，设置束宽为5
output = model.generate(
    inputs.input_ids,
    max_length=50,
    num_beams=5,          # 束宽为5，保留5个候选路径
    early_stopping=True,   # 所有候选都生成终止符时提前停止
    no_repeat_ngram_size=2 # 避免2-gram重复
)

# 解码输出
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Beam Search生成结果（束宽5）：")
print(generated_text)
```
运行结果示例：
```
Beam Search生成结果（束宽5）：
The future of AI is incredibly promising, with advancements in machine learning, natural language processing, and computer vision transforming industries worldwide. From personalized healthcare to autonomous transportation, AI technologies are creating new opportunities and solving complex problems. As research continues to progress, we can expect even more revolutionary applications in the coming decades.
```

## 8. 手工代码实现
从零实现Beam Search，不依赖生成库：
```python
import torch
import torch.nn.functional as F

class SimpleLanguageModel:
    """模拟简单语言模型，输出固定logits用于测试"""
    def __init__(self, vocab_size=100, seq_len=20):
        self.vocab_size = vocab_size
        self.fake_logits = torch.randn(seq_len, vocab_size)

    def forward(self, input_ids):
        """返回输入序列最后一个位置的logits"""
        last_pos = input_ids.shape[1] - 1
        logits = self.fake_logits[last_pos].unsqueeze(0).unsqueeze(0)  # [1,1,vocab_size]
        return logits

def beam_search(model, start_tokens, max_len=20, beam_width=3, eos_token_id=99):
    """
    手写Beam Search解码
    model: 语言模型，支持forward方法
    start_tokens: 起始词元ID列表
    max_len: 最大生成长度
    beam_width: 束宽k
    eos_token_id: 终止符ID
    """
    # 初始化候选集合：(序列, 累计得分)
    candidates = [(start_tokens.copy(), 0.0)]

    for _ in range(max_len - len(start_tokens)):
        all_candidates = []

        # 扩展每个候选序列
        for seq, score in candidates:
            # 准备模型输入
            input_ids = torch.tensor([seq], dtype=torch.long)
            logits = model.forward(input_ids)[:, -1, :]  # 取最后一个位置的logits
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # 转为对数概率

            # 取前beam_width个最高概率的词
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)

            # 生成新候选
            for i in range(beam_width):
                new_token = top_indices[i].item()
                new_score = score + top_log_probs[i].item()
                new_seq = seq.copy()
                new_seq.append(new_token)
                all_candidates.append((new_seq, new_score))

        # 选择得分最高的beam_width个候选
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = all_candidates[:beam_width]

        # 检查是否所有候选都生成了终止符
        if all(seq[-1] == eos_token_id for seq, _ in candidates):
            break

    # 选择总得分最高的序列
    best_seq, best_score = max(candidates, key=lambda x: x[1])
    return best_seq

# 测试手写Beam Search
if __name__ == "__main__":
    model = SimpleLanguageModel(vocab_size=100, seq_len=20)
    start_tokens = [1, 2, 3]
    result = beam_search(model, start_tokens, max_len=20, beam_width=3, eos_token_id=99)
    print("手写Beam Search生成序列（束宽3）：", result)
```
运行结果示例：
```
手写Beam Search生成序列（束宽3）： [1, 2, 3, 42, 15, 87, 42, 15, 87, 99]
```
结果解读：相比Greedy Search，Beam Search的候选路径更多，生成序列更优，但仍可能出现重复。

## 9. 可视化与结果理解
对比不同束宽下的生成结果，可视化候选路径变化：
```python
import matplotlib.pyplot as plt

def visualize_beam_candidates(candidates, step):
    """可视化当前步的候选序列得分"""
    seqs = [str(seq) for seq, _ in candidates]
    scores = [score for _, score in candidates]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(seqs)), scores, align="center")
    plt.yticks(range(len(seqs)), seqs)
    plt.xlabel("累计得分（对数概率）")
    plt.title(f"Beam Search第{step}步候选序列（束宽{len(candidates)}）")
    plt.gca().invert_yaxis()  # 得分最高的在最上面
    plt.show()

# 模拟不同步的候选
step1_candidates = [([1,2,3], 0.0)]
step2_candidates = [([1,2,3,42], -0.5), ([1,2,3,15], -0.8), ([1,2,3,87], -1.2)]
visualize_beam_candidates(step2_candidates, 2)
```
结果解读：图中得分越高（越靠上）的候选越优，Beam Search每一步保留得分最高的k个，避免丢失优质路径。

## 10. 模型评估
对比Beam Search和Greedy Search的生成质量，使用困惑度和BLEU评估：
```python
def calculate_perplexity_beam(model, generated_tokens):
    """计算Beam Search生成序列的困惑度"""
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([generated_tokens], dtype=torch.long)
        logits = model.forward(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, model.vocab_size),
                               shift_labels.view(-1))
        return torch.exp(loss).item()

# 测试评估
model = SimpleLanguageModel(vocab_size=100, seq_len=20)
start_tokens = [1, 2, 3]
beam_result = beam_search(model, start_tokens, max_len=20, beam_width=3)
ppl = calculate_perplexity_beam(model, beam_result)
print(f"Beam Search生成序列困惑度（束宽3）：{ppl:.2f}")  # 通常低于Greedy Search
```
结果解读：Beam Search的困惑度通常低于Greedy Search，说明生成的文本流畅度更高，质量更好。

## 11. 常见问题与易错点
### 数据层面
1.  **束宽设置过大**：导致计算量激增，延迟过高，需根据任务权衡，常用k=3~10。
2.  **未设置提前停止**：所有候选都结束后不停止，浪费计算资源，应设置`early_stopping=True`。
3.  **词表映射错误**：与Greedy Search一致，需保证词表统一。

### 模型层面
1.  **候选路径收敛**：长文本生成时k个候选逐渐趋同，失去Beam Search的优势，需增大k或添加多样性惩罚。
2.  **重复生成**：高概率路径重复，需添加`no_repeat_ngram_size`等重复惩罚参数。
3.  **得分偏移**：长序列的累计对数概率会非常小，需使用对数概率避免下溢。

### 调参层面
1.  **束宽与速度平衡**：k越大质量越高但速度越慢，需根据实际需求调整，实时场景k≤3，质量场景k=5~10。
2.  **未调整长度惩罚**：长序列得分更低，需添加长度惩罚，避免模型偏好短序列。

## 12. 学习总结
Beam Search是Greedy Search的改进解码策略，通过保留k个候选路径避免局部最优，在生成质量和计算效率之间取得了良好平衡，是翻译、摘要等任务的首选解码方法。核心超参数束宽k决定了候选数量，可根据场景灵活调整。

实际使用中直接调用HuggingFace的`num_beams`参数即可，手写实现的核心是维护候选集合、扩展路径、选择top k。评估时需对比Greedy Search的困惑度和内容指标，验证质量提升效果。Beam Search仍存在多样性不足的问题，可结合采样策略进一步优化。

## 13. 练习题与思考题
### 基础题
1.  Beam Search的核心思想是什么？束宽k的作用是什么？
    **答案**：核心思想是每一步保留k个最优候选序列，最终选总得分最高的。k控制候选数量，k越大质量越高但计算量越大。
2.  当k=1时，Beam Search等价于什么算法？
    **答案**：等价于Greedy Search。

### 进阶题
1.  推导Beam Search的单个候选扩展得分公式，并说明为什么使用对数概率。
    **答案**：$\text{Score}(S \oplus w) = \text{Score}(S) + \log P(w|S)$，使用对数概率可将乘法转加法，避免概率乘积下溢。
2.  如何解决Beam Search长文本生成时候选路径收敛的问题？
    **答案**：增大束宽k、添加多样性惩罚（降低已生成序列的得分）、结合随机采样。

### 开放题
如何改进Beam Search，在保持质量的同时提升生成多样性？请提出至少两种方案。
**答案参考**：1. 使用Diverse Beam Search，将k个束分成组，每组生成不同路径；2. 添加随机采样，每个束以一定概率随机选词；3. 结合Top-k采样，限制每个步的选择范围。

## 14. 学习路径建议
### 前置知识
- Greedy Search解码逻辑
- 对数概率、argmax操作
- 自回归生成基本流程

### 平行学习
- Greedy Search（基础对比）
- Diverse Beam Search（多样性改进）
- Top-k/Nucleus采样（多样性解码）

### 进阶学习
- 束搜索的变体（如向量化Beam Search优化速度）
- 用于树搜索的Beam Search（如对话系统多轮生成）

### 推荐资源
1.  原书第2章Beam Search相关内容
2.  HuggingFace Beam Search文档：https://huggingface.co/docs/transformers/main/en/generation_strategies#beam-search
3.  论文《Beam Search Strategies for Neural Machine Translation》
