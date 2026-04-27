# N-gram语言模型 学习文档

> N-gram是一种基于词频统计的语言模型，通过计算n个词连续出现的概率来预测下一个词。

## 1. 算法基础认知

### 1.1 什么是N-gram模型

N-gram是一种简单的统计语言模型，它根据前n-1个词来预测第n个词出现的概率。模型核心是计算词序列的联合概率。

### 1.2 直觉类比

类似于玩文字接龙游戏：如果我说"我今天"，根据历史文本，"去"比"挖"更可能出现在后面。N-gram就是通过统计这种连续出现的模式来进行预测。

### 1.3 历史背景

- 1940年代：香农信息论为N-gram奠定基础
- 1970年代：广泛应用于语音识别
- 至今仍是NLP baseline

### 1.4 算法定位

- **任务类型**：语言模型/序列预测
- **所属类别**：统计语言模型
- **前置知识**：概率论基础

## 2. 核心原理

### 2.1 核心思想

通过滑动窗口统计n个词连续出现的频率，计算条件概率。

### 2.2 工作流程

1. 分词：将文本分割成词序列
2. 统计：计算每个n-gram的出现频率
3. 预测：根据前n-1个词预测下一个词
4. 生成：按概率采样生成文本

### 2.3 参数说明

- **N**：n-gram的n值（unigram=1, bigram=2, trigram=3）
- **V**：词汇表大小

## 3. 数学公式与推导

### 3.1 核心公式

给定词序列 $w_1, w_2, ..., w_n$，N-gram模型计算：

$$P(w_n|w_1^{n-1}) = \frac{C(w_1^{n-1}, w_n)}{C(w_1^{n-1})}$$

其中 $C(\cdot)$ 表示计数。

### 3.2 马尔可夫假设

模型假设第n个词只与前n-1个词相关：
$$P(w_n|w_1^{n-1}) \approx P(w_n|w_{n-N+1}^{n-1})$$

## 4. 应用场景

### 4.1 典型应用

1. **文本生成**：简单对话/补全
2. **语音识别**：音素到文本
3. **拼写纠错**：上下文纠错
4. **机器翻译**：短语对齐

### 4.2 适用数据特征

- 大量标注语料
- 简单场景
- 资源受限环境

## 5. 优缺点分析

### 5.1 优点

| 优点 | 说明 |
|------|------|
| 简单高效 | 实现和计算都简单 |
| 可解释 | 统计结果透明 |
| 速度快 | 无需GPU |

### 5.2 缺点

| 缺点 | 说明 |
|------|------|
| 数据稀疏 | 很多n-gram不出现 |
| 语义缺失 | 不理解词义 |
| 长依赖差 | n较小时不考虑远上下文 |

## 6. 调库实现

```python
from nltk import ngrams
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

# 准备文本数据
text = [["hello", "world"], ["how", "are", "you"]]

# 构建N-gram模型
train_data, vocab = padded_everygram_pipeline(2, text)
model = MLE(2)
model.fit(train_data, vocab)

# 生成文本
print(model.generate(5, ["hello"]))
```

## 7. 手工代码实现

```python
import re
from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
    
    def train(self, sentences):
        for sentence in sentences:
            # 添加padding
            padded = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            # 统计ngram
            for ng in ngrams(padded, self.n):
                context = ng[:-1]
                word = ng[-1]
                self.ngram_counts[ng] += 1
                self.context_counts[context] += 1
    
    def probability(self, context, word):
        ng = tuple(context) + (word,)
        count = self.ngram_counts.get(ng, 0)
        total = self.context_counts.get(tuple(context), 0)
        if total == 0:
            return 0
        return count / total
    
    def predict_next(self, context):
        candidates = {}
        for word in self.context_counts:
            if word[:-1] == tuple(context[-self.n+1:]):
                candidates[word[-1]] = self.probability(context, word[-1])
        if not candidates:
            return None
        return max(candidates, key=candidates.get)

# 测试
texts = [["i", "love", "you"], ["i", "love", "machine"]]
model = NGramModel(2)
model.train(texts)
print(model.probability(["i"], "love"))
print(model.predict_next(["i"]))
```

## 8. 学习总结

N-gram是语言建模的经典方法，核心是通过词频统计预测下一个词。虽然简单，但为复杂模型奠定了基础。

## 9. 练习题

**题目**：为什么N-gram模型会有数据稀疏问题？

**答案**：随着n增大，可能的n-gram组合数量指数增长，而实际语料中能覆盖的组合非常有限，导致很多组合的计数为0。

## 10. 学习路径建议

- **前置**：概率论基础
- **进阶**：神经网络语言模型→LSTM→Transformer
- **资源**：Language Modeling综述论文