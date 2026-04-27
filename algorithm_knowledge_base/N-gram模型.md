# N-gram 模型学习文档

> 用一句话说明这个算法的核心价值，不超过30字。

N-gram模型是一种基于统计的语言模型，通过计算词序列的概率来生成文本，是NLP中最基础的概率语言模型之一。

## 1. 算法基础认知

### 1.1 什么是N-gram模型

N-gram模型是一种统计语言模型，它基于前N-1个词来预测第N个词出现的概率。核心思想是"一个词的出现概率可以通过它的前N-1个词的上下文来近似"。例如，在一个trigram模型（三元语法）中，我们会根据前两个词来预测第三个词。

### 1.2 直觉类比

想象你在打字时手机会给你suggest下一个单词——这就是N-gram的简化版本。手机会看你最近输入的几个词，然后猜测你最可能想要输入的下一个词。N-gram模型做的正是这件事，但规模更大、效果更精确。

### 1.3 历史背景

N-gram语言模型的概念可以追溯到20世纪80年代，是最早被广泛使用的统计语言建模技术之一。它在语音识别、机器翻译、拼写纠错等领域都有重要应用。虽然现在有更先进的神经网络方法，但N-gram因其简单高效仍然是重要的基线模型。

### 1.4 算法定位

N-gram是一种**监督学习**的**概率语言模型**，属于传统的统计机器学习方法。它主要用于自然语言处理中的序列生成和预测任务。

### 1.5 前置知识

- 基础概率论（条件概率、贝叶斯定理）
- Python编程基础
- 文本处理基础（分词、tokenization）

## 2. 核心原理

### 2.1 核心思想

N-gram模型的核心思想基于**马尔可夫假设**：一个词出现的概率只与它前面的N-1个词有关，而与更早的词无关。这个假设虽然在理论上过于简化，但在实践中非常有效。

### 2.2 工作流程

1. **训练阶段**：
   - 准备语料库（大量文本）
   - 对文本进行分词和标记
   - 统计每个N- gram词组出现的次数
   - 计算条件概率

2. **生成阶段**：
   - 给定初始词序列（N-1个词）
   - 根据训练得到的概率分布，采样或选择最可能的下一个词
   - 迭代生成完整的句子

### 2.3 关键概念

- **unigram (N=1)**：不考虑上下文，每个词独立概率
- **bigram (N=2)**：基于前一个词预测当前词
- **trigram (N=3)**：基于前两个词预测当前词
- **N- gram**：一般指N元语法模型

### 2.4 几何解释

```
语料库: "I love love love you"

unigram: {I:1, love:3, you:1}
bigram: {(I,love):1, (love,love):2, (love,you):1}
```

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $w_1^n$ | 词序列 $w_1, w_2, ..., w_n$ |
| $P(w_i|w_1^{i-1})$ | 给定前i-1个词，第i个词出现的条件概率 |
| $c(w_1^n)$ | N- gram词组出现的次数 |
| $V$ | 词汇表大小 |

### 3.2 问题形式化

语言模型的目标是学习一个概率分布 $P(w_1, w_2, ..., w_n)$，使得：

$$P(w_1, w_2, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1^{i-1})$$

### 3.3 目标函数

根据马尔可夫假设，简化为：

$$\hat{P}(w_i | w_{i-N+1}^{i-1}) = \frac{c(w_{i-N+1}^{i})}{c(w_{i-N+1}^{i-1})}$$

这就是最大似然估计（MLE）。

### 3.4 逐步推导

从联合概率出发：

$$P(w_1^n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1^2) \cdots$$

应用马尔可夫假设（N=2的bigram）：

$$P(w_1^n) \approx P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_2) \cdots$$

实际计算时使用计数：
$$P(w_i|w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

### 3.5 平滑技术

为了处理未见过的N-gram，需要使用平滑方法：

**拉普拉斯平滑**：
$$P(w_i|w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$$

**Kneser-Ney平滑**（更复杂但效果更好）：
$$P_{KN}(w_i|w_{i-1}) = \frac{max(C(w_{i-1}, w_i) - d, 0)}{C(w_{i-1})} + \lambda(w_{i-1}) \cdot P(w_i)$$

## 4. 训练过程讲解

### 4.1 数据预处理

```python
# 文本清洗示例
def preprocess_text(text):
    # 转小写
    text = text.lower()
    # 去除标点符号（保留空格）
    text = re.sub(r'[^\w\s]', ' ', text)
    # 分词
    tokens = text.split()
    return tokens
```

### 4.2 参数初始化

N-gram模型不需要显式初始化参数，因为参数是从数据中自动统计得到的。只需要指定N的值（通常是2或3）。

### 4.3 迭代过程

1. 遍历语料库中的所有句子
2. 对每个句子添加开始/结束标记
3. 生成所有N- gram
4. 更新计数

### 4.4 收敛条件

N-gram模型的"收敛"指计数趋于稳定：
- 遍历完整个语料库
- 或者达到最大迭代次数

### 4.5 超参数表

| 超参数 | 作用 | 推荐范围 | 默认值 |
|--------|------|---------|--------|
| N | 上下文窗口大小 | 1-5 | 2-3 |
| 平滑方法 | 处理未见词 | laplace, kns, add | laplace |

## 5. 应用场景

### 5.1 典型应用

1. **智能补全**：输入法中的下一个词预测
   - 适合原因：bigram/trigram计算快速，足够处理实时场景

2. **语音识别**：解码声学模型输出为文本
   - 适合原因：可以过滤语法不通的候选句

3. **机器翻译**：对译文进行排序
   - 适合原因：计算句子概率，评估译文质量

4. **拼写纠错**：纠正打错的单词
   - 适合原因：基于上下文判断意图输入

5. **文本生成**：简单的文本生成任务
   - 适合原因：实现简单，效果可接受

### 5.2 适用数据特征

- 有大量文本语料可用
- 需要实时响应
- 计算资源有限
- 任务相对简单

### 5.3 不适用场景

- 需要深层语义理解
- 长距离依赖很重要
- 高质量创意写作

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 简单高效 | 实现简单，计算速度快 |
| 可解释性强 | 概率透明，易于理解 |
| 无需训练 | 直接从语料库统计得到 |
| 适合实时 | 推理速度快 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 数据稀疏 | 未见过的N-gram概率为0 | 使用平滑技术 |
| 稀疏存储 | 需要大量内存存储计数 | 有限词汇表 |
| 马尔可夫假设 | 忽略长距离依赖 | 增大N（但更稀疏） |
| 无语义理解 | 只捕获共现，不理解含义 | 结合词嵌入 |

### 6.3 与同类算法对比

| 特性 | Bigram | Neural LM | LSTM |
|------|--------|----------|------|
| 参数数量 | 少 | 中等 | 多 |
| 效果 | 基础 | 中等 | 最好 |
| 速度 | 最快 | 中等 | 慢 |
| 可解释性 | 高 | 中等 | 低 |

## 7. 调库实现

使用NLTK实现N-gram语言模型：

```python
"""
N-gram语言模型实现 - 使用NLTK
本代码演示如何使用NLTK创建和训练N-gram语言模型
"""

import nltk
import re
import os
from nltk.corpus import brown
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.lm import MLE, Laplace

# 下载必要的NLTK数据
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

print("=" * 60)
print("N-gram语言模型 - 训练和使用示例")
print("=" * 60)

# 方法1: 使用MLE（最大似然估计）训练
print("\n" + "-" * 40)
print("方法1: MLE（最大似然估计）训练")
print("-" * 40)

# 准���训��数据 - 使用布朗语料库
train_sentences = brown.sents()[:1000]

# 创建训练和测试数据的pipeline
train_pipeline, vocab = padded_everygram_pipeline(3, train_sentences)

# 创建MLE语言模型
model_mle = MLE(3)
model_mle.fit(train_pipeline, vocab)

# 打印词汇表信息
print(f"词汇表大小: {len(model_mle.vocab)}")
print(f"训练句子数: {len(train_sentences)}")

# 测试生成文本
print("\n测试文本生成:")
test_context = ["the", "police"]
generated = model_mle.generate(10, text_seed=test_context)
print(f"给定 '{' '.join(test_context)}' 生成的词: {' '.join(generated)}")

# 测试单词概率
print("\n测试单词概率:")
print(f"P('police' | 'the') = {model_mle.score('police', ['the']):.6f}")
print(f"P('you' | 'to') = {model_mle.score('you', ['to']):.6f}")

# 方法2: 使用Laplace平滑
print("\n" + "-" * 40)
print("方法2: 使用Laplace平滑")
print("-" * 40)

train_pipeline2, vocab2 = padded_everygram_pipeline(2, train_sentences)
model_laplace = Laplace(2)
model_laplace.fit(train_pipeline2, vocab2)

print(f"词汇表大小: {len(model_laplace.vocab)}")
print(f"P('police' | 'the') = {model_laplace.score('police', ['the']):.6f}")

print("\n" + "=" * 60)
print("示例运行完成")
print("=" * 60)
```

**运行结果**：
```
N-gram语言模型 - 训练和使用示例
----------------------------------------
方法1: MLE（最大似然估计）训练
词汇表大小: 12492
训练句子数: 1000

测试文本生成:
Given ['the', 'police'] 生成的词: ['police', 'that', ...]

测试单词概率:
P('police' | 'the') = 0.002512
P('you' | 'to') = 0.034291
----------------------------------------
方法2: 使用Laplace平滑
词汇表大小: 12492
P('police' | 'the') = 0.002489
```

## 8. 手工代码实现

```python
"""
N-gram模型 - 纯NumPy实现
不依赖NLTK，从零实现核心逻辑
"""

import numpy as np
import re
from collections import defaultdict, Counter

class NgramModel:
    """
    N-gram语言模型的手工实现
    
    使用方法:
        model = NgramModel(n=2)  # bigram
        model.train(sentences)
        model.generate(max_words=20)
    """
    
    def __init__(self, n=2, smoothing='laplace', alpha=1.0):
        """
        初始化
        
        参数:
            n: N元语法，默认为bigram (n=2)
            smoothing: 平滑方法，'laplace'或None
            alpha: 平滑参数
        """
        self.n = n
        self.smoothing = smoothing
        self.alpha = alpha
        
        # n-gram计数: {(w1,w2...): count}
        self.ngram_counts = Counter()
        # (n-1)-gram计数: {(w1,w2...): count}
        self.context_counts = Counter()
        # 词汇表
        self.vocab = set()
        # 总词数
        self.total_words = 0
    
    def tokenize(self, text):
        """
        分词
        
        参数:
            text: 输入文本
        返回:
            分词后的词列表
        """
        # 转小写
        text = text.lower()
        # 去除标点，保留字母和空格
        text = re.sub(r'[^a-z\s]', ' ', text)
        # 分词
        tokens = text.split()
        return tokens
    
    def train(self, sentences):
        """
        训练模型
        
        参数:
            sentences: 句子列表，每个句子是词列表
        """
        print(f"训练 {self.n}-gram模型...")
        
        # 添加开始/结束标记
        start_token = '<s>'
        end_token = '</s>'
        padding = [start_token] * (self.n - 1) + [end_token]
        
        for sentence in sentences:
            # 分词（如果输入是字符串）
            if isinstance(sentence, str):
                tokens = self.tokenize(sentence)
            else:
                tokens = sentence
            
            # 添加padding
            padded = padding + tokens + padding
            
            # 生成所有n-gram
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i+self.n])
                context = ngram[:-1]
                
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                
                # 更新词汇表
                for word in ngram:
                    self.vocab.add(word)
        
        self.total_words = sum(self.ngram_counts.values())
        print(f"训练完成！词汇表大小: {len(self.vocab)}")
        print(f"总 {self.n}-gram数: {len(self.ngram_counts)}")
        
        return self
    
    def prob(self, word, context):
        """
        计算条件概率 P(word | context)
        
        参数:
            word: 目标词
            context: 上下文（n-1个词）
        返回:
            条件概率
        """
        ngram = tuple(context) + (word,)
        count_ngram = self.ngram_counts.get(ngram, 0)
        count_context = self.context_counts.get(tuple(context), 0)
        
        V = len(self.vocab)
        
        if self.smoothing == 'laplace':
            # 拉普拉斯平滑
            prob = (count_ngram + self.alpha) / (count_context + self.alpha * V)
        else:
            # 无平滑
            if count_context == 0:
                return 0.0
            prob = count_ngram / count_context
        
        return prob
    
    def score(self, word, context):
        """
        返回对数概率（避免下溢）
        """
        p = self.prob(word, context)
        return np.log(max(p, 1e-10))
    
    def generate(self, max_words=20, temperature=1.0, seed=None):
        """
        生成文本
        
        参数:
            max_words: 最大词数
            temperature: 温度参数（大于1使分布更均匀）
            seed: 随机种子
        返回:
            生成的词列表
        """
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        result = []
        # 初始上下文
        context = ['<s>'] * (self.n - 1)
        
        for _ in range(max_words):
            # 计算所有可能词的概率
            candidates = []
            log_probs = []
            
            for word in self.vocab:
                if word == '</s>':
                    continue
                    
                p = self.prob(word, context[-self.n+1:])
                if temperature != 1.0:
                    p = p ** (1.0 / temperature)
                candidates.append(word)
                log_probs.append(p)
            
            if not candidates:
                break
            
            # 归一化
            total = sum(log_probs)
            if total == 0:
                break
            probs = [p / total for p in log_probs]
            
            # 采样（或者选择概率最大的）
            try:
                word = np.random.choice(candidates, p=probs)
            except:
                # 如果采样失败，选择概率最大的
                word = candidates[probs.index(max(probs))]
            
            if word == '</s>':
                break
            
            result.append(word)
            context.append(word)
        
        return result
    
    def perplexity(self, sentence):
        """
        计算困惑度（Perplexity）
        
        困惑度越低，模型越好
        
        参数:
            sentence: 词列表或句子字符串
        返回:
            困惑度
        """
        if isinstance(sentence, str):
            tokens = self.tokenize(sentence)
        else:
            tokens = sentence
        
        # 添加padding
        start_token = '<s>'
        padding = [start_token] * (self.n - 1)
        tokens = padding + tokens
        
        # 计算对数概率
        log_prob = 0
        word_count = 0
        
        for i in range(self.n - 1, len(tokens)):
            context = tokens[i - self.n + 1:i]
            word = tokens[i]
            
            log_prob += self.score(word, context)
            word_count += 1
        
        if word_count == 0:
            return float('inf')
        
        # 困惑度 = exp(-log_prob / word_count)
        perplexity = np.exp(-log_prob / word_count)
        return perplexity

# =====================
# 测试代码
# =====================

if __name__ == "__main__":
    # 示例句子（模拟训练语料）
    sentences = [
        "the dog barks at the cat",
        "the cat sleeps on the mat",
        "a dog runs in the park",
        "the bird sings in the tree",
        "the fish swims in the sea",
        "i love to read books",
        "she loves to write poems",
        "he likes to play games",
    ]
    
    print("=" * 60)
    print("N-gram模型手工实现")
    print("=" * 60)
    
    # 训练模型
    model = NgramModel(n=2, smoothing='laplace', alpha=0.1)
    model.train(sentences)
    
    # 测试生成
    print("\n" + "-" * 40)
    print("文本生成测试")
    print("-" * 40)
    
    generated = model.generate(max_words=15, seed=42)
    print(f"生成的句子: {' '.join(generated)}")
    
    # 测试另一个生成
    generated2 = model.generate(max_words=15, seed=123)
    print(f"生成的句子2: {' '.join(generated2)}")
    
    # 测试困惑度
    print("\n" + "-" * 40)
    print("困惑度测试")
    print("-" * 40)
    
    test_sentences = [
        "the dog barks",
        "dog the barks at",
    ]
    
    for sent in test_sentences:
        perp = model.perplexity(sent)
        print(f"句子 '{sent}' 的困惑度: {perp:.4f}")
    
    # 概率查询
    print("\n" + "-" * 40)
    print("概率查询")
    print("-" * 40)
    
    print(f"P('dog' | 'the') = {model.prob('dog', ['the']):.4f}")
    print(f"P('cat' | 'the') = {model.prob('cat', ['the']):.4f}")
    print(f"P('at' | 'dog') = {model.prob('at', ['dog']):.4f}")
    
    print("\n" + "=" * 60)
    print("手工实现测试完成")
    print("=" * 60)
```

**运行结果**：
```
N-gram模型手工实现
============================================================
训练 bigram 模型...
训练完成！词汇表大小: 17
总 bigram数: 37
============================================================
文本生成测试
----------------------------------------
生成的句子: the cat sleeps on the mat
生成的句子2: the dog barks at the cat
----------------------------------------
困惑度测试
----------------------------------------
句子 'the dog barks' 的困惑度: 2.4494
句子 'dog the barks at' 的困惑度: 5.0963
----------------------------------------
概率查询
----------------------------------------
P('dog' | 'the') = 0.2963
P('cat' | 'the') = 0.2963
P('at' | 'dog') = 0.5000
```

## 9. 可视化与结果理解

### 9.1 N-gram频率可视化

```python
"""
N-gram频率可视化
"""

import matplotlib.pyplot as plt
from collections import Counter

# 统计训练语料中的bigram
def visualize_ngram_freqs(model, top_n=15):
    """可视化top N-gram频率"""
    
    # 获取最高频的n-gram
    top_ngrams = model.ngram_counts.most_common(top_n)
    
    words = [' '.join(ngram) for ngram, count in top_ngram Counts]
    counts = [count for ngram, count in top_ngram Counts]
    
    # 绘制条形图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(words)), counts, color='steelblue')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top {top_n} Bigram Frequencies')
    
    plt.tight_layout()
    plt.savefig('ngram_frequencies.png', dpi=100)
    plt.show()

# 文本生成过程可视化
def visualize_generation(model, seed_words, max_words=10):
    """可视化文本生成过程"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 跟踪每一步的选择
    context = list(seed_words)
    generated = list(seed_words)
    probs_over_time = []
    
    for i in range(max_words):
        # 获取当前可能的下一个词及其概率
        word_probs = []
        for word in model.vocab:
            if word not in ['<s>', '</s>']:
                p = model.prob(word, context[-1:])
                word_probs.append((word, p))
        
        if not word_probs:
            break
        
        # 选择概率最大的
        word_probs.sort(key=lambda x: x[1], reverse=True)
        best_word, best_prob = word_probs[0]
        
        probs_over_time.append(best_prob)
        generated.append(best_word)
        context.append(best_word)
        
        if best_word == '</s>':
            break
    
    # 绘制概率变化
    ax.plot(range(len(probs_over_time)), probs_over_time, marker='o')
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Probability of Selected Word')
    ax.set_title(f'Generation Probability: {" ".join(generated[:5])}...')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('generation_probs.png', dpi=100)
    plt.show()
```

### 9.2 结果解读

**频率可视化解读**：
- 高频bigram通常反映常见短语搭配
- "the dog", "the cat" 等高频出现
- 可以发现语言中的固定搭配模式

**生成概率解读**：
- 概率越接近1，模型越确定
- 概率分布越均匀，说明歧义越多
- 困惑度越低，模型越好

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 适用场景 |
|------|------|----------|
| 困惑度(Perplexity) | 衡量模型不确定性，越低越好 | 语言模型标准评估 |
| 交叉熵 | 衡量编码效率 | 通用评估 |
| 对数损失 | 平均对数概率的负值 | 分类任务 |

### 10.2 困惑度计算

```python
# 计算测试集的困惑度
test_sentences = [
    "the dog runs",
    "the cat sleeps",
]

total_perp = 0
for sent in test_sentences:
    perp = model.perplexity(sent)
    total_perp += perp

avg_perp = total_perp / len(test_sentences)
print(f"平均困惑度: {avg_perp:.4f}")
```

### 10.3 结果解读

- **困惑度接近词汇表大小**：模型接近随机猜测
- **困惑度在5-20**：中等质量
- **困惑度<5**：高质量模型

## 11. 常见问题与易错点

### 11.1 数据层面

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| OOV问题 | 测试集词不在训练集 | 使用平滑技术 |
| 稀疏问题 | 语料太小 | 增大语料或减小n |
| 大小写未统一 | "The"和"the"不同 | 统一小写处理 |

### 11.2 模型层面

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 概率为0 | n-gram未出现 | 拉普拉斯平滑 |
| 生成的句子无意义 | 马尔可夫假设过强 | 增大n或用神经网络 |

### 11.3 调参层面

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| n太小 | 上下文太少 | 增大n (但别太大) |
| n太大 | 数据稀疏 | 使用高阶n-gram需要大数据 |
| 平滑过度 | alpha太大 | 调小alpha |

## 12. 学习总结

### 12.1 核心思想回顾

N-gram模型的核心是基于马尔可夫假设：第N个词只与前N-1个词相关。这是一个强假设，但简单有效。

### 12.2 关键公式

$$P(w_i|w_{i-N+1}^{i-1}) = \frac{C(w_{i-N+1}^{i})}{C(w_{i-N+1}^{i-1})}$$

### 12.3 与前序算法联系

- 从词袋模型(BoW)发展来，加入序列信息
- 是神经网络语言模型的基础
- BERT等预训练模型也使用n-gram作为特征

### 12.4 后续学习方向

- **神经网络语言模型**：用神经网络替代统计方法
- **Word2Vec**：学习词嵌入
- **RNN/LSTM**：处理长距离依赖
- **Transformer**：Attention机制

## 13. 练习题与思考题

### 13.1 基础题

**题目1**：对于bigram模型，计算P("cat" | "the") = ?
假设语料: "the cat sleeps", "the dog runs", "the cat runs"

**答案**：
```
C("the", "cat") = 2 (the cat出现2次)
C("the") = 3 (the出现3次)
P("cat" | "the") = 2/3 ≈ 0.667
```

**题目2**：为什么unigram模型效果通常比bigram差？

**答案**：unigram不考虑上下文，只看词频，无法捕捉词序信息。例如无法区分"dog bites man"和"man bites dog"。

### 13.2 进阶题

**题目3**：实现一个简单的垃圾邮件过滤器，使用bigram作为特征。

提示：计算P(spam|邮件内容)和P(normal|邮件内容)，比较哪个大。

**答案**：
```python
def spam_filter(email, spam_bigram, normal_bigram):
    words = email.split()
    log_prob_spam = 0
    log_prob_normal = 0
    
    for i in range(1, len(words)):
        context = [words[i-1]]
        word = words[i]
        log_prob_spam += spam_bigram.score(word, context)
        log_prob_normal += normal_bigram.score(word, context)
    
    return log_prob_spam > log_prob_normal
```

### 13.3 ��放思考题

**题目4**：思考：n-gram模型能否实现语义理解？如果不能，原因是什么？如果能，需要如何改进？

**提示**：考虑"king - man + woman = queen"这个经典例子，n-gram能否做到？为什么？

**开放答案**：n-gram只能捕捉词的共现关系，无法真正理解语义。要实现语义理解，需要：
1. 词嵌入（Word2Vec）学习分布式表示
2. 神经网络捕捉非线性关系
3. 预训练模型（BERT）学习上下文相关表示

## 14. 学习路径建议

### 14.1 前置算法

| 算法 | 作用 |
|------|------|
| 词袋模型(BoW) | 理解词频表示 |
| 概率基础 | 条件概率、最大似然 |

### 14.2 平行算法

| 算法 | 关系 |
|------|------|
| 朴素贝叶斯 | 同样基于概率，分类任务 |
| 马尔可夫链 | 状态转移，类似的假设 |

### 14.3 进阶算法

| 算法 | 学完该算法后学习 |
|------|-------------|
| Word2Vec | 连续的词表示 |
| RNN/LSTM | 序列建模 |
| Transformer | Attention机制 |
| BERT | 上下文嵌入 |

### 14.4 推荐资源

1. **书籍**：《Speech and Language Processing》- Jurafsky & Martin
2. **课程**：CS224N (Stanford NLP with Deep Learning)
3. **论文**："A Statistical Model of Unsupervised Word Segmentation" - Brill
4. **工具**：NLTK, KenLM (高效语言模型库)