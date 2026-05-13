# Tokenization 学习文档

## 1. 算法基础认知

Tokenization（分词/词元化）是自然语言处理中的基础步骤，将原始文本转换为模型可以处理的离散符号序列。在深度学习时代，tokenization的质量直接影响模型性能。本文档将深入讲解四种主流的分词方法：基于空格分词、Byte Pair Encoding（BPE）、WordPiece和SentencePiece。

### 1.1 为什么需要Tokenization？

计算机处理的是离散符号，无法直接理解原始文本。分词的目的：
- **降低词汇表大小**：将无限词汇映射到有限ID
- **保留语义信息**：保持基本的语义单元
- **统一输入格式**：为模型提供统一的输入表示
- **处理OOV问题**：通过子词分词处理未登录词

### 1.2 分词方法的发展历程

**第一代：基于空格分词**
- 按空格切分，词汇表巨大
- 无法处理未登录词（OOV）
- 语义粒度粗

**第二代：基于规则的分词**
- 正则表达式匹配
- 字典查表
- 仍无法很好处理OOV

**第三代：子词分词**
- BPE（Byte Pair Encoding）
- WordPiece
- SentencePiece
- 有效减少OOV问题

### 1.3 Tokenization的重要性

好的tokenization应该：
- **低词表大小**：减少模型参数量
- **低比特率**：每个token携带更多信息
- **处理OOV**：通过子词组合表示新词
- **保持语义**：保留基本的语义单元

## 2. 核心原理

### 2.1 基于空格的分词

最简单的方法，直接按空格和标点切分：

```python
def whitespace_tokenize(text):
    """空格分词"""
    return text.split()

# 示例
text = "深度学习是机器学习的一个分支"
tokens = whitespace_tokenize(text)
print(tokens)  # ['深度学习是机器学习的一个分支']
# 全部分到一个token，词汇表：全部中文=1个词
```

### 2.2 Byte Pair Encoding (BPE)

BPE最初是一种数据压缩算法，由Gage于1994年提出。2016年，Google将其应用于神经机器翻译的分词。

**BPE的核心思想**：将高频的相邻字节对合并成新的符号。

```
Algorithm: BPE Training
---------------------------------
Input: text, vocab_size
Output: merges, vocab

Step 1: 初始化字符表
    vocab = set(all characters in text)

Step 2: 统计字符对频率
    while len(vocab) < vocab_size:
        pairs = count_adjacent_pairs(text)
        
Step 3: 选择最高频的字符对
        best_pair = max(pairs, key=pairs.get)

Step 4: 合并字符对
        text = merge(text, best_pair)
        vocab.add(best_pair)
        merges.append(best_pair)
```

### 2.3 WordPiece

WordPiece是Google开发的方法，BERT使用的分词方式。与BPE不同，WordPiece基于语言模型选择合并。

**WordPiece的核心思想**：优先合并使语言模型得分最高的字符对。

```
Algorithm: WordPiece Training
---------------------------------
Input: text, vocab_size
Output: vocab

Step 1: 初始化字符表
    vocab = set(all characters)

Step 2: 构建语言模型
    lm = build_language_model(text)

Step 3: 贪心合并
    while len(vocab) < vocab_size:
        best_pair = None
        best_score = -inf
        
        for pair in all_adjacent_pairs(text):
            score = lm.score(merge(text, pair))
            if score > best_score:
                best_score = score
                best_pair = pair
        
        vocab.add(best_pair)
```

**语言模型得分**：
$$\text{score}(p) = \log P(\text{merged}) - \log P(\text{original})$$

### 2.4 SentencePiece

SentencePiece是Google开发的端到端分词工具，解决了前几种方法的问题。

**SentencePiece的特点**：
- **无空格假设**：不依赖空格分词
- **BPE和Unigram支持**：支持两种分词算法
- **动态词汇表**：训练时确定词汇表大小

## 3. 数学公式与推导

### 3.1 BPE的合并操作

设文本为字符序列 $C = [c_1, c_2, ..., c_n]$，相邻字符对为 $P = (c_i, c_{i+1})$。

**频率统计**：
$$\text{freq}(p) = \sum_{i} \mathbb{1}[c_i = p[0] \land c_{i+1} = p[1]]$$

**合并规则**：选择频率最高的字符对进行合并：
$$p^* = \arg\max_{p \in P} \text{freq}(p)$$

**合并操作**：将相邻的 $p^*$ 替换为单个符号：
$$ \text{merge}(C, p^*) = [x \text{ if } x = p^* \text{ else } x] $$

### 3.2 WordPiece的语言模型得分

WordPiece使用bigram语言模型：
$$P(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})}$$

**合并得分**：
$$\text{score}(p) = \log P(\text{merged}) = \sum_{i} \log P(w_i | w_{i-1})$$

### 3.3 Unigram Language Model

SentencePiece的Unigram模型使用概率方式：

**字符序列的联合概率**：
$$P(\text{sequence}) = \prod_{i} P(x_i) \cdot P(x_{i+1} | x_i)$$

**训练目标**：最大化训练数据的似然：
$$\mathcal{L} = \sum_{w \in \text{train}} \log P(w)$$

## 4. 训练过程讲解

### 4.1 BPE训练流程

```python
def train_bpe(corpus, vocab_size):
    """训练BPE分词器"""
    
    # Step 1: 初始化字符表
    vocab = set()
    for text in corpus:
        for char in text:
            vocab.add(char)
    
    # Step 2: 迭代合并
    merges = []
    for _ in range(vocab_size - len(vocab)):
        # 统计字符对频率
        pairs = {}
        for text in corpus:
            for i in range(len(text) - 1):
                pair = (text[i], text[i+1])
                pairs[pair] = pairs.get(pair, 0) + 1
        
        if not pairs:
            break
        
        # 选择最高频
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        
        # 合并
        for i in range(len(text):
            text = merge(text, best_pair)
        
        vocab.add(best_pair)
        
        if len(vocab) >= vocab_size:
            break
    
    return vocab, merges
```

### 4.2 分词过程

```python
def tokenize(text, vocab, merges):
    """使用BPE分词"""
    tokens = list(text)
    
    for merge in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge:
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    
    return tokens
```

### 4.3 SentencePiece训练

```python
import sentencepiece as spm

# 训练SentencePiece模型
spm.SentencePieceTrainer.train(
    input='text.txt',
    model_prefix='sp_model',
    vocab_size=8000,
    character_coverage=0.9995,
    model_type='unigram',  # BPE或unigram
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='<unk>',
    unk_piece='<unk>',
    bos_piece='<s>',
    eos_piece='</s>'
)

# 使用
sp = spm.SentencePieceProcessor()
sp.load('sp_model.model')

# 分词
pieces = sp.encode('深度学习很有趣', out_type=str)
print(pieces)  # ['▁深度', '学习', '很', '有趣']
```

## 5. 应用场景

### 5.1 GPT/BERT的分词

```python
from transformers import GPT2Tokenizer, BertTokenizer

# GPT2 Tokenizer（BPE）
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = gpt_tokenizer.encode("深度学习")
print(tokens)  # [28276, 3007, 2430, 2455, 2533]

# BERT Tokenizer（WordPiece）
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokens = bert_tokenizer.encode("深度学习")
print(tokens)  # [5001, 4289]
```

### 5.2 分词对比

| 方法 | 示例输出 | 词汇表大小 |
|------|---------|-----------|
| 空格 | ["深度学习"] | ~50k |
| BPE | ["▁深", "度学习"] | ~30k |
| WordPiece | ["深", "度", "学习"] | ~20k |
| SentencePiece | ["▁深", "度", "学", "习"] | ~8k |

### 5.3 Token ID映射

```python
class Tokenizer:
    """简单的Tokenizer实现"""
    
    def __init__(self, vocab):
        # 构建词表
        self.vocab = vocab
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.unk_id = self.word2id.get('<unk>', 0)
    
    def encode(self, text):
        """文本转ID"""
        return [self.word2id.get(t, self.unk_id) for t in text]
    
    def decode(self, ids):
        """ID转文本"""
        return [self.id2word.get(i, '<unk>') for i in ids]
```

## 6. 优缺点分析

### 6.1 各种方法的对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 空格分词 | 简单 | OOV严重 | 英文(空格分隔) |
| BPE | 效果好 | 需要训练 | 通用 |
| WordPiece | 语义好 | 速度慢 | BERT等 |
| SentencePiece | 端到端 | 库依赖 | 生产部署 |

### 6.2 OOV处理

**问题**：未登录词（Out-of-Vocabulary）的处理

**解决方案**：
1. 子词组合：如["深", "度学", "习"]组合成"深学习"
2. 使用[UNK]：标记为未知
3. Byte-level：使用字节表示

### 6.3 词汇表大小选择

- **英文**：30k-50k词汇表
- **中文**：通常需要5000-30000
- **多语言**：更大词汇表

## 7. 调库实现（Python + HuggingFace）

### 7.1 使用HuggingFace Tokenizers

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, Sequence

# BPE训练
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=['<pad>', '<unk>', '<s>', '</s>']
)
tokenizer.train(files=['text.txt'], trainer=trainer)

# 分词
output = tokenizer.encode("深度学习很有趣")
print(output.tokens)
print(output.ids)
```

### 7.2 完整的分词示例

```python
from transformers import AutoTokenizer
import torch

# 加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

# 单句分词
single = tokenizer("深度学习")
print("Single:", single.input_ids)

# 批量分词
batch = tokenizer(["深度学习", "机器学习"])
print("Batch:", batch.input_ids)

# 带padding
padded = tokenizer(
    ["深度学习", "机器学习和"],
    padding=True,
    max_length=10,
    truncation=True,
    return_tensors='pt'
)
print("Padded:", padded)

# 返回token类型
tokens = tokenizer(
    "深度学习是机器学习的一个分支",
    return_token_type_ids=True,
    return_attention_mask=True
)
print("Tokens:", tokens.keys())
```

### 7.3 自定义Vocabulary

```python
# 构建自定义词表
vocab = {
    '<pad>': 0,
    '<unk>': 1,
    '<s>': 2,
    '</s>': 3,
    '深': 4,
    '度': 5,
    '学': 6,
    '习': 7,
    ...

}

# 分词器实现
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_id = vocab['<unk>']
    
    def encode(self, text):
        return [self.vocab.get(c, self.unk_id) for c in text]
    
    def decode(self, ids):
        id2word = {v: k for k, v in self.vocab.items()}
        return ''.join(id2word.get(i, '<unk>') for i in ids)

# 使用
tokenizer = SimpleTokenizer(vocab)
ids = tokenizer.encode("深度学习")
print(ids)
text = tokenizer.decode(ids)
print(text)
```

### 7.4 分词可视化

```python
def visualize_tokenization():
    """可视化不同分词方法"""
    from transformers import (GPT2Tokenizer, 
                               BertTokenizer)
    
    text = "深度学习是人工智能的核心技术"
    
    # GPT2
    gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokens = gpt2.tokenize(text)
    print(f"GPT2: {gpt2_tokens}")
    
    # BERT
    bert = BertTokenizer.from_pretrained('bert-base-chinese')
    bert_tokens = bert.tokenize(text)
    print(f"BERT: {bert_tokens}")

visualize_tokenization()
```

## 8. 手工代码实现

### 8.1 简单BPE实现

```python
from collections import Counter, defaultdict
import re

class SimpleBPE:
    """BPE分词器的简化实现"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
    
    def train(self, corpus):
        """训练BPE
        
        corpus: 列表形式的文本
        """
        # Step 1: 初始化字符表
        self.vocab = set()
        for text in corpus:
            for char in text:
                self.vocab.add(char)
        
        # Step 2: 迭代合并
        text = corpus
        for _ in range(self.vocab_size - len(self.vocab)):
            # 统计字符对频率
            pairs = Counter()
            for t in text:
                for i in range(len(t) - 1):
                    pairs[t[i:i+2]] += 1
            
            if not pairs:
                break
            
            # 最常见的字符对
            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)
            self.vocab.add(best_pair)
            
            # 合并
            new_text = []
            for t in text:
                new_text.append(t.replace(best_pair, ''))
            text = new_text
        
        # 构建词表
        self.vocab = list(self.vocab)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
    
    def encode(self, text):
        """分词"""
        tokens = list(text)
        for merge in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] + tokens[i+1] == merge:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return [self.word2idx.get(t, 0) for t in tokens]
    
    def decode(self, ids):
        """解码"""
        idx2word = {i: w for w, i in self.word2idx.items()}
        return ''.join(idx2word.get(i, '') for i in ids)

# 测试
corpus = ["深度学习", "机器学习", "人工智能", "深度机器学习"]
bpe = SimpleBPE(vocab_size=100)
bpe.train(corpus)
print("Vocab:", len(bpe.vocab))
print("Encode:", bpe.encode("深度学习"))
print("Decode:", bpe.decode(bpe.encode("深度学习")))
```

### 8.2 WordPiece简化实现

```python
class SimpleWordPiece:
    """WordPiece简化实现"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.word2idx = {}
    
    def train(self, corpus):
        """训练WordPiece
        
        基于频率的贪心算法
        """
        # 初始化字符表
        for text in corpus:
            for char in text:
                self.vocab.add(char)
        
        # 迭代增加词汇
        for _ in range(self.vocab_size - len(self.vocab)):
            # 统计所有可能的bigram
            bigrams = Counter()
            for text in corpus:
                for i in range(len(text) - 1):
                    bigrams[text[i:i+2]] += 1
            
            if not bigrams:
                break
            
            # 选择频率最高的
            new_word = bigrams.most_common(1)[0][0]
            if new_word in self.vocab:
                break
            
            self.vocab.add(new_word)
        
        # 构建词表
        self.vocab = list(self.vocab)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
    
    def tokenize(self, text):
        """分词（最大匹配）"""
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for j in range(len(text) - i, 0, -1):
                if text[i:i+j] in self.vocab:
                    tokens.append(text[i:i+j])
                    i += j
                    matched = True
                    break
            
            if not matched:
                tokens.append(text[i])
                i += 1
        
        return tokens

# 测试
corpus = ["深度学习", "机器学习", "人工智能", "深度机器学习"]
wp = SimpleWordPiece(vocab_size=50)
wp.train(corpus)
print("Vocab:", len(wp.vocab))
print("Tokenize:", wp.tokenize("深度学习"))
```

## 9. 可视化与结果理解

### 9.1 分词结果对比

```python
def compare_tokenizers():
    """对比不同分词器的结果"""
    import matplotlib.pyplot as plt
    
    text = "深度学习是人工智能的核心技术"
    
    # 模拟不同分词器的输出
    tokenizers = {
        'Whitespace': ['深度学习是人工智能的核心技术'],
        'BPE': ['▁深度', '学习', '是', '人工', '智能', '的', '核��', '���术'],
        'WordPiece': ['深', '度', '学习', '是', '人工', '智能', '的', '核心', '技术'],
        'SentencePiece': ['▁深度', '学习', '是', '人工', '智能', '的', '核心', '技术']
    }
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = range(len(tokenizers))
    for i, (name, tokens) in enumerate(tokenizers.items()):
        ax.barh(i, len(tokens), color=f'C{i}', alpha=0.7)
        for j, token in enumerate(tokens[:10]):
            ax.text(j, i, token, fontsize=8, va='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokenizers.keys())
    ax.set_xlabel('Token Count')
    ax.set_title('Tokenization Comparison')
    
    plt.tight_layout()
    plt.savefig('tokenizer_comparison.png', dpi=150)

compare_tokenizers()
```

### 9.2 词汇表大小与OOV关系

```python
def plot_vocab_oov():
    """可视化词汇表与OOV的关系"""
    import numpy as np
    
    vocab_sizes = [1000, 2000, 5000, 10000, 20000, 50000]
    oov_rates = [0.5, 0.3, 0.15, 0.08, 0.04, 0.02]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(vocab_sizes, oov_rates, 'b-o')
    plt.xlabel('Vocabulary Size')
    plt.ylabel('OOV Rate')
    plt.title('Vocabulary Size vs OOV Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig('vocab_oov.png', dpi=150)

plot_vocab_oov()
```

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| 词汇表大小 | 实际词表大小 |
| 平均token数 | 每个句子平均token数 |
| OOV率 | 未登录词比例 |
| 压缩率 | 原始字符/Token数 |

### 10.2 评估代码

```python
def evaluate_tokenizer(tokenizer, test_data):
    """评估分词器"""
    total_tokens = 0
    total_chars = 0
    oov_count = 0
    
    for text in test_data:
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        total_chars += len(text)
        
        for t in tokens:
            if t not in tokenizer.vocab:
                oov_count += 1
    
    return {
        'avg_tokens': total_tokens / len(test_data),
        'compression_rate': total_chars / total_tokens,
        'oov_rate': oov_count / total_tokens
    }
```

## 11. 常见问题与易错点

### 11.1 中文分词问题

**问题**：中文按字分还是按词分？
**建议**：
- 通用任务：使用子词分词
- 专业领域：可结合专业词典

### 11.2 标点处理

**问题**：标点如何处理？
**建议**：保留标点作为独立token

### 11.3 特殊字符

**问题**：如何处理URL、邮箱等？
**建议**：
- 按字符级拆分
- 或保留为整体

### 11.4 多语言混合

**问题**：中英文混合文本
**建议**：使用多语言分词器

## 12. 学习总结

### 核心要点

1. **分词的目标**：将文本映射为有限ID
2. **BPE**：基于频率的合并算法
3. **WordPiece**：基于语言模型得分
4. **SentencePiece**：端到端方案

### 方法选择

- **通用**：SentencePiece
- **BERT**：WordPiece
- **GPT**：BPE

## 13. 练习题与思考题

### 练习题

**Q1**: BPE和WordPiece的主要区别是什么？

**答案**：BPE选择最高频的字符对合并，WordPiece选择使语言模型得分最高的字符对合并。

**Q2**: 为什么中文需要的词汇表更大？

**答案**：中文字符组合更多，需要更大词汇表覆盖，常用汉字约3500个，而词组组合更多。

**Q3**: 什么是OOV问题？如何解决？

**答案**：OOV是未登录词问题。子词分词可以将OOV拆分为已知子词，保证模型可以处理。

### 思考题

**Q1**: 如何选择词汇表大小？

**答案**：根据任务复杂度、数据量、计算资源综合考虑。 一般30k-50k适合大多数场景。

**Q2**: 分词对模型性能的影响��

**答案**：好的分词能降低OOV率、提高效率、保留语义。不合适的分词会导致信息损失。

## 14. 学习路径建议

### 基础阶段
1. 理解基于空格的分词
2. 学习BPE原理
3. 实现简单BPE

### 进阶阶段
1. 学习WordPiece原理
2. 对比不同分词器
3. SentencePiece使用

### 实践阶段
1. 在项目中使用分词器
2. 调优词汇表大小
3. 处理特殊字符

### 参考资源
- HuggingFace tokenizers库
- SentencePiece GitHub
- BERT论文