# 词分词（Tokenization）学习文档

> 用一句话说明这个算法的核心价值，不超过30字。

词分词是NLP的基础步骤，将文本切分成模型可处理的最小单位（token），直接影响模型对语言的理解能力，是所有语言模型的前置步骤。

---

## 1. 算法基础认知

### 1.1 什么是词分词

词分词（Tokenization）是将文本字符串分解为更小的、可管理的单位（称为token）的过程。这些token可以是：
- **词（Word）**：英文中的单词，中文中的词语
- **子词（Subword）**：更小的语义单元，如"unhappiness"可以分成"un"+"happ"+"i"+"ness"
- **字符（Character）**：单个字符
- **字节（Byte）**：字节单位

词分词是NLP管道的第一部，所有的文本处理、特征提取都依赖于分词的质量。一个好的分词器能够让模型更准确地理解语言结构。

### 1.2 直觉类比

把词分词想象成学习外语时查词典的过程：当你阅读一段英文文章时，你不会把整个句子作为一个整体来理解，而是先把句子分成单词，然后逐个查单词的意思，最后理解整个句子的含义。

对于中文，分词更加重要。例如"我爱你"如果不断开是"我爱你"一个词，但如果正确断开是"我""爱""你"三个词，不同的分词方式会导致完全不同的理解。

### 1.3 历史背景

词分词的发展经历了几个重要阶段：
- **早期（1960-1980）**：基于规则的分词，主要使用词典和有限状态机
- **统计方法（1990-2010）**：基于隐马尔可夫模型（HMM）和条件随机场（CRF）的统计分词
- **子词方法（2015-至今）**：BPE、WordPiece、SentencePiece等子词分词方法成为主流
- **神经分词（2018-至今）**：基于BiLSTM的序列标注分词

2019年后，基于Transformer的预训练模型（如BERT、GPT）统一使用子词分词，成为NLP的标准做法。

### 1.4 算法定位

词分词是**NLP管道**的**基础组件**，是所有文本处理任务的第一步：

- 文本分类 → 分词 → 特征提取 → 分类
- 机器翻译 → 分词 → 编码 → 解码 → 分词
- 对话系统 → 分词 → 编码 → 对话管理 → 解码 → 分词

### 1.5 前置知识

- 正则表达式
- 字符串处理
- 基础数据结构（ Trie树、前缀树）
- 基础概率统计

---

## 2. 核心原理

### 2.1 分词方法分类

```
词分词方法
├── 基于词典的分词
│   ├── 正向最大匹配
│   ├── 逆向最大匹配
│   └── 双向最大匹配
├── 基于统计的分词
│   ├── N-gram语言模型
│   ├── HMM
│   └── CRF
├── 子词分词
│   ├── BPE
│   ├── WordPiece
│   └── SentencePiece
└── 字符级分词
    ├── Char-level
    └── Byte-level
```

### 2.2 核心概念

**词表（Vocabulary）**：分词后所有token的集合。

**未登录词（OOV, Out-of-Vocabulary）**：不在词表中的词。

**OOV问题**：当测试集中出现训练集中没有见过的词时，分词器无法正确处理。

**分词粒度**：可以按词级别、字符级别、子词级别进行分词。

### 2.3 工作流程

1. **读取文本**：输入原始字符串
2. **预处理**：小写化、去除标点、标准化
3. **分词**：按照规则将文本切分为token
4. **编码**：将token转换为ID
5. **后处理**：添加特殊标记（[CLS], [SEP]等）

---

## 3. 数学公式与推导

### 3.1 符号约定

| 符号 | 含义 |
|------|------|
| $T$ | 分词后的token序列 |
| $V$ | 词表 |
| $v$ | 词表大小 |
| $x$ | 输入文本 |
| $w$ | 词/子词 |

### 3.2 最大匹配算法

**前向最大匹配（FMM）**：
```
从左到右扫描
取最长的匹配词
```

**后向最大匹配（BMM）**：
```
从右到左扫描
取最长的匹配词
```

**双向最大匹配**：
```
同时进行FMM和BMM
选择词数较少或OOV较少的版本
```

### 3.3 BPE算法原理

BPE（Byte Pair Encoding）的核心是合并高频的连续字节对：

1. **初始化**：将文本按字符级别分割
2. **统计对**：统计所有相邻token对的出现频率
3. **合并**：将最高频的token对合并为一个新token
4. **重复**：直到达到目标词表大小

$$P = \arg\max_{p \in Pairs} Count(p)$$

### 3.4 WordPiece算法

WordPiece使用最大似然估计来选择合并：

$$Score = \log P(merged) - \log P(part1) - \log P(part2)$$

### 3.5 CRF分词

使用条件随机场学习最佳分词序列：

$$y^* = \arg\max_{y} P(y|x) = \arg\max_{y} \frac{\exp(Score(x,y))}{\sum_{y'}\exp(Score(x,y'))}$$

---

## 4. 训练过程讲解

### 4.1 数据准备

```python
class TokenizationDataset:
    """分词训练数据准备"""
    
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
    
    def prepare_for_training(self):
        """准备训练数据"""
        # 收集所有文本
        all_texts = []
        for text in self.texts:
            # 预处理
            text = text.lower()
            text = self.tokenizer.normalize(text)
            all_texts.append(text)
        
        return all_texts
```

### 4.2 训练分词器

```python
def train_tokenizer(texts, vocab_size=30000):
    """训练BPE分词器"""
    # 1. 字符级分词
    words = []
    for text in texts:
        word = list(text) + ['</w>']
        words.append(word)
    
    # 2. 统计词频
    vocab = set()
    for word in words:
        vocab.update(word)
    
    # 3. BPE合并
    merges = {}
    while len(vocab) < vocab_size:
        pairs = Counter()
        for word in words:
            for i in range(len(word) - 1):
                if word[i] in vocab and word[i+1] in vocab:
                    pairs[(word[i], word[i+1])] += 1
        
        if not pairs:
            break
        
        best = max(pairs, key=pairs.get)
        merges[best] = best[0] + best[1]
        vocab.add(best[0] + best[1])
        
        # 应用合并
        words = apply_merges(words, best)
    
    return vocab, merges
```

### 4.3 分词器配置

| 参数 | 作用 | 说明 |
|------|------|------|
| vocab_size | 词表大小 | 控制词汇量 |
| min_frequency | 最小频率 | 过滤低频词 |
| unk_token | 未登录词 | 未知词标记 |
| pad_token | 填充标记 |  Padding |
| bos_token | 开始标记 | 序列开始 |
| eos_token | 结束标记 | 序列结束 |

---

## 5. 应用场景

### 5.1 英文分词

```python
class EnglishTokenizer:
    """英文分词器"""
    
    def __init__(self):
        import re
        self.pattern = re.compile(r"[\w']+|[.,!?;:]")
    
    def tokenize(self, text):
        return self.pattern.findall(text)

# 测试
tokenizer = EnglishTokenizer()
text = "Hello, world! This is an example."
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# ['Hello', ',', 'world', '!', 'This', 'is', 'an', 'example', '.']
```

### 5.2 中文分词

```python
import jieba

def chinese_tokenization():
    """中��分词示例"""
    text = "我爱自然语言处理"
    
    # 精确模式
    tokens = jieba.lcut(text, cut_all=False)
    print(f"精确模式: {tokens}")
    
    # 全模式
    tokens_all = jieba.lcut(text, cut_all=True)
    print(f"全模式: {tokens_all}")
    
    # 搜索引擎模式
    tokens_search = jieba.lcut_for_search(text)
    print(f"搜索模式: {tokens_search}")

chinese_tokenization()
```

### 5.3 BPE子词分词

```python
from collections import Counter

class BPETokenizer:
    """BPE子词分词器"""
    
    def __init__(self):
        self.vocab = {}
        self.merges = {}
    
    def train(self, texts, vocab_size=1000):
        """训练BPE"""
        # 字符级分词
        words = []
        for text in texts:
            word = list(text.lower()) + ['</w>']
            words.append(word)
        
        # 统计单字符
        vocab = set()
        for word in words:
            vocab.update(word)
        
        # 合并迭代
        while len(vocab) < vocab_size:
            pairs = Counter()
            for word in words:
                for i in range(len(word) - 1):
                    if word[i] in vocab and word[i+1] in vocab:
                        pairs[(word[i], word[i+1])] += 1
            
            if not pairs:
                break
            
            best = max(pairs, key=pairs.get)
            self.merges[best] = best[0] + best[1]
            vocab.add(best[0] + best[1])
            
            # 应用合并
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best:
                        new_word.append(best[0] + best[1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words
        
        self.vocab = vocab
        return self
    
    def encode(self, text):
        """编码"""
        tokens = list(text.lower()) + ['</w>']
        
        for merge, merged in sorted(self.merges.items(), key=lambda x: -len(x[1])):
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return [t for t in tokens if t in self.vocab or t == '</w>']

# 测试
texts = ["hello world", "good morning", "thank you"]
bpe = BPETokenizer()
bpe.train(texts, vocab_size=50)
print(f"BPE Tokens: {bpe.encode('hello')}")
```

### 5.4 WordPiece分词

```python
import tensorflow as tf
from tensorflow_text import WordpieceTokenizer

def wordpiece_tokenization():
    """WordPiece分词"""
    tokenizer = WordpieceTokenizer(
        vocablookup_table=None,
        suffix_indicator='##',
        max_bytes_per_token=100,
        unknown_token='[UNK]',
        split_unknown_tokens=False
    )
    
    # 文本
    text = "unhappiness"
    tokens = tokenizer.tokenize(text).numpy()
    
    print(f"WordPiece tokens: {tokens}")

try:
    wordpiece_tokenization()
except:
    print("使用HuggingFace实现")
```

### 5.5 SentencePiece分词

```python
import sentencepiece as spm

def sentencepiece_demo():
    """SentencePiece分词"""
    # 训练
    model = spm.SentencePieceTrainer.train(
        input='text.txt',
        model_prefix='m',
        vocab_size=1000,
        character_coverage=1.0,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>'
    )
    
    # 加载
    sp = spm.SentencePieceProcessor()
    sp.load('m.model')
    
    # 分词
    text = "我爱自然语言处理"
    tokens = sp.encode(text, out_type=int)
    pieces = sp.encode(text, out_type=str)
    
    print(f"Tokens: {tokens}")
    print(f"Pieces: {pieces}")

sentencepiece_demo()
```

### 5.6 多语言分词

```python
from transformers import BertTokenizer

def multilingual_tokenization():
    """多语言分词"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # 中文
    text_zh = "我爱人工智能"
    tokens_zh = tokenizer.tokenize(text_zh)
    print(f"中文: {tokens_zh}")
    
    # 英文
    text_en = "I love AI"
    tokens_en = tokenizer.tokenize(text_en)
    print(f"英文: {tokens_en}")
    
    # 日文
    text_ja = "私はAIが好きです"
    tokens_ja = tokenizer.tokenize(text_ja)
    print(f"日文: {tokens_ja}")

multilingual_tokenization()
```

---

## 6. 优缺点分析

### 6.1 优点

| 分词方法 | 优点 |
|---------|------|
| 基于词典 | 实现简单、速度快 |
| 统计分词 | 可解决歧义 |
| BPE | 解决OOV问题 |
| WordPiece | 效果好、灵活性好 |

### 6.2 缺点

| 分词方法 | 缺点 |
|---------|------|
| 词典分词 | 无法处理新词 |
| 统计分词 | 需要标注数据 |
| 子词分词 | 词表可能膨胀 |

### 6.3 方法对比

| 方法 | 准确率 | 速度 | OOV处理 | 需要训练数据 |
|------|--------|------|---------|--------------|
| 词典 | 中 | 快 | 差 | 否 |
| CRF | 高 | 中 | 好 | 是 |
| BPE | 高 | 快 | 好 | 是 |
| WordPiece | 高 | 快 | 好 | 是 |

---

## 7. 调库实现

### 7.1 中文jieba分词

```python
import jieba
from collections import Counter

# 设置词典
jieba.load_userdict("dict.txt")

# 分词
text = "自然语言处理是人工智能的重要组成部分"
tokens = jieba.lcut(text)

print(f"分词结果: {tokens}")
# ['自然语言', '处理', '是', '人工智能', '的', '重要', '组成部分']

# 词性标注
for word, pos in jieba.posseg(text):
    print(f"{word}: {pos}")

# 关键词提取
import jieba.analyse
keywords = jieba.analyse.extract_tags(text, topK=5)
print(f"关键词: {keywords}")
```

### 7.2 英文NLTK分词

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# 句子分词
text = "Hello world. This is a test."
sentences = sent_tokenize(text)
print(f"句子: {sentences}")

# 单词分词
tokens = word_tokenize(text)
print(f"单词: {tokens}")

# 词干提取
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed = [stemmer.stem(t) for t in tokens]
print(f"词干: {stemmed}")
```

### 7.3 子词HuggingFace

```python
from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer

# BERT分词器
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "unhappiness"
tokens = bert_tokenizer.tokenize(text)
ids = bert_tokenizer.encode(text)
print(f"BERT: {tokens}, IDs: {ids}")

# GPT2分词器
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = gpt_tokenizer.tokenize(text)
print(f"GPT2: {tokens}")

# T5分词器
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
tokens = t5_tokenizer.tokenize(text)
print(f"T5: {tokens}")
```

### 7.4 完整训练流程

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

def train_bpe_tokenizer():
    """训练BPE分词器"""
    # 1. 创建BPE模型
    tokenizer = Tokenizer(models.BPE())
    
    # 2. 预分词器
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # 3. 训练
    trainer = trainers.BpeTrainer(
        vocab_size=1000,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    files = ["text.txt"]
    tokenizer.train(files, trainer)
    
    # 4. 解码器
    tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
    
    # 5. 保存
    tokenizer.save("tokenizer.json")
    
    # 使用
    encoding = tokenizer.encode("Hello world!")
    print(f"IDs: {encoding.ids}")
    print(f"Tokens: {encoding.tokens}")
    
    return tokenizer

train_bpe_tokenizer()
```

### 7.5 完整的分词示例

```python
class CompleteTokenizer:
    """完整的分词器"""
    
    def __init__(self, vocab_file=None):
        if vocab_file:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_file)
        else:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __call__(self, text, return_tensors='pt'):
        """分词"""
        # 编码
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors=return_tensors
        )
        return encoding
    
    def decode(self, ids):
        """解码"""
        return self.tokenizer.decode(ids)
    
    def batch_decode(self, batch_ids):
        """批量解码"""
        return self.tokenizer.batch_decode(batch_ids)

# 使用
tokenizer = CompleteTokenizer()
encoding = tokenizer("Hello, world!")
print(f"Input IDs: {encoding['input_ids']}")
print(f"Attention: {encoding['attention_mask']}")
```

---

## 8. 手工代码实现

### 8.1 词典分词实现

```python
import re
from collections import Counter

class DictionaryTokenizer:
    """基于词典的分词"""
    
    def __init__(self, dictionary):
        self.dictionary = set(dictionary)
    
    def forward_max_match(self, text):
        """前向最大匹配"""
        result = []
        i = 0
        
        while i < len(text):
            matched = False
            
            for j in range(len(text), i, -1):
                if text[i:j] in self.dictionary:
                    result.append(text[i:j])
                    i = j
                    matched = True
                    break
            
            if not matched:
                result.append(text[i])
                i += 1
        
        return result
    
    def backward_max_match(self, text):
        """后向最大匹配"""
        result = []
        i = len(text)
        
        while i > 0:
            matched = False
            
            for j in range(i, 0, -1):
                if text[j:i] in self.dictionary:
                    result.insert(0, text[j:i])
                    i = j
                    matched = True
                    break
            
            if not matched:
                result.insert(0, text[i-1])
                i -= 1
        
        return result
    
    def bidirectional_match(self, text):
        """双向最大匹配"""
        fmm = self.forward_max_match(text)
        bmm = self.backward_max_match(text)
        
        # 选择词数少的
        if len(fmm) != len(bmm):
            return fmm if len(fmm) < len(bmm) else bmm
        
        # 选择单字数少的
        fmm_single = sum(1 for w in fmm if len(w) == 1)
        bmm_single = sum(1 for w in bmm if len(w) == 1)
        
        return fmm if fmm_single < bmm_single else bmm

# 测试
dictionary = ["自然语言", "处理", "人工", "智能", "重要", "组成", "部分"]
tokenizer = DictionaryTokenizer(dictionary)

text = "自然语言处理是人工智能的重要组成部分"
result = tokenizer.bidirectional_match(text)
print(f"分词结果: {result}")
```

### 8.2 统计分词实现

```python
import numpy as np
from collections import Counter

class NGramTokenizer:
    """基于N-gram的语言模型分词"""
    
    def __init__(self, n=2):
        self.n = n
        self.ngram_counts = {}
        self.word_counts = {}
        self.total = 0
    
    def train(self, texts, words):
        """训练"""
        # 词频统计
        self.word_counts = Counter(words)
        self.total = sum(self.word_counts.values())
        
        # N-gram统计
        for text in texts:
            padded = ['<S>'] * (self.n-1) + text + ['</S>']
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i+self.n])
                self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1
    
    def probability(self, word, prev_words):
        """计算条件概率"""
        context = tuple(prev_words[-(self.n-1):])
        ngram = context + (word,)
        
        count = self.ngram_counts.get(ngram, 0)
        context_count = self.ngram_counts.get(context, 0)
        
        if context_count == 0:
            return self.word_counts.get(word, 1) / self.total
        
        return count / context_count
    
    def tokenize(self, text):
        """分词（最大概率路径）"""
        words = list(text)
        n = len(words)
        
        # 动态规划
        dp = [0] * (n + 1)
        prev = [-1] * (n + 1)
        
        for i in range(1, n + 1):
            for j in range(i, 0, -1):
                word = ''.join(words[j-1:i])
                prob = np.log(self.probability(word, words[:j-1]))
                
                if dp[j-1] + prob > dp[i]:
                    dp[i] = dp[j-1] + prob
                    prev[i] = j - 1
        
        # 回溯
        result = []
        i = n
        while i > 0:
            result.insert(0, ''.join(words[prev[i]:i]))
            i = prev[i]
        
        return result

# 测试
texts = [["自然", "语言", "处理"], ["人工", "智能", "是", "未来"]]
all_words = [w for text in texts for w in text]

tokenizer = NGramTokenizer(n=2)
tokenizer.train(texts, all_words)

result = tokenizer.tokenize(list("自然语言处理"))
print(f"分词结果: {result}")
```

### 8.3 BPE完整实现

```python
from collections import Counter

class BPEProcessor:
    """完整的BPE处理器"""
    
    def __init__(self):
        self.vocab = {}
        self.merges = {}
    
    def get_word_frequencies(self, texts):
        """获取词频"""
        freq = Counter()
        for text in texts:
            words = text.split()
            freq.update(words)
        return freq
    
    def get_vocab(self, texts):
        """获取词表"""
        vocab = Counter()
        for text in texts:
            for char in text:
                vocab[char] += 1
        
        for word in texts[0].split():
            for i in range(len(word) - 1):
                bigram = word[i] + word[i+1]
                vocab[bigram] += 1
        
        return vocab
    
    def train(self, texts, target_vocab_size=1000):
        """训练BPE"""
        # 初始化字符表
        vocab = Counter()
        for text in texts:
            for char in text:
                vocab[char] += 1
        
        # 迭代合并
        num_merges = target_vocab_size - len(vocab)
        
        for _ in range(num_merges):
            # 统计所有bigram
            bigram_counts = Counter()
            
            for text in texts:
                chars = list(text)
                for i in range(len(chars) - 1):
                    if chars[i] in vocab and chars[i+1] in vocab:
                        bigram_counts[chars[i] + chars[i+1]] += 1
            
            if not bigram_counts:
                break
            
            # 找到最频繁的
            most_common = bigram_counts.most_common(1)[0]
            merge = most_common[0]
            
            self.merges[merge] = len(self.merges)
            vocab[merge] = most_common[1]
            
            # 应用合并
            new_texts = []
            for text in texts:
                new_text = text[0]
                for char in text[1:]:
                    if new_text[-1] + char == merge:
                        new_text = new_text[:-1] + merge
                    else:
                        new_text += char
                new_texts.append(new_text)
            
            texts = new_texts
        
        self.vocab = vocab
        return self
    
    def encode(self, text):
        """编码为IDs"""
        tokens = list(text)
        
        # 应用所有合并
        for merge in sorted(self.merges.keys(), key=lambda x: -len(x)):
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] + tokens[i+1] == merge:
                    new_tokens.append(merge)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # 转换为IDs
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(list(self.vocab.keys()).index(token))
            else:
                ids.append(-1)  # OOV
        
        return tokens, ids
    
    def decode(self, ids):
        """解码为文本"""
        id2token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for id in ids:
            if id in id2token:
                tokens.append(id2token[id])
        
        return ''.join(tokens)

# 测试
texts = ["hello world", "good morning", "thank you very much"]
bpe = BPEProcessor()
bpe.train(texts, target_vocab_size=30)

tokens, ids = bpe.encode("hello")
print(f"Tokens: {tokens}")
print(f"IDs: {ids}")
```

### 8.4 完整分词Pipeline

```python
class FullTokenizer:
    """完整的分词流程"""
    
    def __init__(self, vocab_file=None):
        self.vocab = self._load_vocab(vocab_file)
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'
    
    def _load_vocab(self, vocab_file):
        """加载词表"""
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                vocab[line.strip()] = i
        return vocab
    
    def tokenize(self, text):
        """分词"""
        tokens = []
        sub_tokens = []
        
        for char in text:
            sub_tokens.append(char)
        
        for sub_token in sub_tokens:
            if sub_token in self.vocab:
                if sub_tokens:
                    tokens.extend(sub_tokens)
                    sub_tokens = []
                tokens.append(sub_token)
            else:
                sub_tokens.append(self.unk_token)
        
        if sub_tokens:
            tokens.extend(sub_tokens)
        
        return tokens
    
    def convert_tokens_to_ids(self, tokens):
        """转换为IDs"""
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab.get(self.unk_token)))
        return ids
    
    def convert_ids_to_tokens(self, ids):
        """ID转换为tokens"""
        id2token = {v: k for k, v in self.vocab.items()}
        return [id2token.get(id, self.unk_token) for id in ids]
    
    def encode(self, text, max_length=512):
        """完整编码"""
        tokens = self.tokenize(text)
        tokens = [self.cls_token] + tokens[:max_length-2] + [self.sep_token]
        ids = self.convert_tokens_to_ids(ids)
        
        # Padding
        while len(ids) < max_length:
            ids.append(self.vocab[self.pad_token])
        
        return ids
    
    def decode(self, ids):
        """解码"""
        tokens = self.convert_ids_to_tokens(ids)
        return ''.join(tokens)
```

---

## 9. 可视化与结果理解

### 9.1 分词结果可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_tokens():
    """可视化分词结果"""
    texts = [
        "unhappiness",
        "machine learning",
        "自然语言处理"
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    tokenizers = ['raw', 'char', 'wordpiece']
    
    for ax, text in zip(axes, texts):
        # 原始
        tokens = list(text)
        
        # 绘制
        colors = plt.cm.Set3(np.linspace(0, 1, len(tokens)))
        
        for i, (token, color) in enumerate(zip(tokens, colors)):
            ax.barh(0, 1, left=i, color=color)
            ax.text(i + 0.5, 0, token, ha='center', va='center')
        
        ax.set_xlim(0, len(tokens))
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(text)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_tokens()
```

### 9.2 词表分布可视化

```python
import matplotlib.pyplot as plt
from collections import Counter

def visualize_vocab_distribution():
    """词表分布可视化"""
    # 假设已有词频
    lengths = list(range(1, 20))
    frequencies = [np.random.randint(10, 100) for _ in lengths]
    
    plt.figure(figsize=(12, 6))
    plt.bar(lengths, frequencies)
    plt.xlabel('Token长度')
    plt.ylabel('频次')
    plt.title('词表分布')
    plt.grid(True, alpha=0.3)
    plt.show()

visualize_vocab_distribution()
```

### 9.3 OOV分析

```python
import matplotlib.pyplot as plt
import numpy as np

def analyze_oov():
    """OOV分析"""
    vocab_sizes = [1000, 5000, 10000, 30000, 50000]
    oov_rates = [0.5, 0.2, 0.1, 0.05, 0.03]
    
    plt.figure(figsize=(10, 6))
    plt.plot(vocab_sizes, oov_rates, 'o-')
    plt.xlabel('词表大小')
    plt.ylabel('OOV率')
    plt.title('词表大小vs OOV率')
    plt.grid(True, alpha=0.3)
    plt.show()

analyze_oov()
```

### 9.4 分词速度对比

```python
import matplotlib.pyplot as plt

def compare_tokenizer_speed():
    """分词速度对比"""
    methods = ['Dict', 'CRF', 'BPE', 'WordPiece', 'SentencePiece']
    speeds = [10000, 5000, 8000, 7500, 7000]  # tokens/sec
    
    plt.figure(figsize=(10, 6))
    plt.barh(methods, speeds)
    plt.xlabel('速度 (tokens/sec)')
    plt.title('分词方法速度对比')
    for i, v in enumerate(speeds):
        plt.text(v + 100, i, str(v), va='center')
    plt.grid(True, alpha=0.3)
    plt.show()

compare_tokenizer_speed()
```

### 9.5 分词粒度对比

```python
import matplotlib.pyplot as plt
import numpy as np

def compare_granularity():
    """分词粒度对比"""
    words = ["machine learning", "unhappiness", "自然语言处理"]
    granularities = [
        ["machine", "learning"],
        ["un", "happ", "i", "ness"],
        ["自然语言", "处理"]
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, word, tokens in zip(axes, words, granularities):
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(tokens)))
        
        for i, (token, color) in enumerate(zip(tokens, colors)):
            ax.barh(0, 1, left=i, color=color, edgecolor='black')
            ax.text(i + 0.5, 0, token, ha='center', va='center', fontsize=9)
        
        ax.set_xlim(0, len(tokens))
        ax.set_title(word)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

compare_granularity()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| 分词准确率 | 正确率 | 正确分词数/总数 |
| OOV召回率 | 新词识别 | 识别出的新词数/所有新词 |
| 速度 | 处理速度 | tokens/second |
| 词表大小 | 词汇量 | 唯一token数 |

### 10.2 评估代码

```python
def evaluate_tokenizer(tokenizer, test_data, ground_truth):
    """评估分词器"""
    results = {
        'accuracy': [],
        'oov_recall': [],
        'speed': []
    }
    
    for text, gt in zip(test_data, ground_truth):
        # 分词
        tokens = tokenizer.tokenize(text)
        
        # 准确率
        accuracy = sum(t == g for t, g in zip(tokens, gt)) / len(gt)
        results['accuracy'].append(accuracy)
        
        # OOV召回
        oov_count = sum(1 for t in tokens if t not in vocab)
        oov_recall = oov_count / len([t for t in gt if t not in vocab])
        results['oov_recall'].append(oov_recall)
    
    return {
        'accuracy': np.mean(results['accuracy']),
        'oov_recall': np.mean(results['oov_recall']),
        'speed': np.mean(results['speed'])
    }
```

---

## 11. 常见问题与易错点

### 11.1 问题诊断表

| 问题 | 原因 | 方案 |
|------|------|------|
| OOV问题 | 词表太小 | 增大词表 |
| 歧义问题 | 词典不全 | 统计分词 |
| 速度慢 | 算法复杂 | 优化实现 |
| 编码错误 | 特殊字符 | 预处理 |

### 11.2 常见错误

```python
# 错误1：不处理标点
# 解决方案：先去除或单独处理

# 错误2：不处理英文大小写
# 解决方案：小写化

# 错误3：不处理数字
# 解决方案：保留或替换
```

---

## 12. 学习总结

### 核心思想

分词将文本分解为可管理的最小单元，是NLP的基础步骤。子词分词通过BPE等方法解决OOV问题。

### 关键公式

BPE合并：$best = \arg\max_{pair} Count(pair)$

### 后续学习

- 词性标注
- 命名实体识别
- 预训练分词器

---

## 13. 练习题与思考题

**题目1**：为什么BPE能解决OOV？

**答案**：BPE将词分解为子词，可以通过子词组合表示新词。

**题目2**：如何选择分词粒度？

**答案**：根据任务需求，粒度越细越能处理新词，但序列越长。

---

## 14. 学习路径建议

### 前置知识

- 正则表达式
- 数据结构

### 推荐学习路线

1. **词级别分词**
2. **子词分词BPE**
3. **预训练模型分词器**

### 推荐资源

1. **HuggingFace Tokenizers**
2. **jieba**
3. **SentencePiece**

---

