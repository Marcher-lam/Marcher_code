# SentencePiece 分词器 学习文档

> 来源线索：本节内容根据原书中关于"SentencePiece分词器"（第4章 4.2.2节）的相关章节整理、扩展与教学化改写。

> 语言无关的通用分词——SentencePiece让大模型能处理世界上任何语言。

## 1. 算法基础认知

**一句话定义**：SentencePiece是一个语言无关的文本分词工具，将文本切分为子词单元用于模型训练。

**直觉类比**：传统分词像按照字典切句子（需要预先知道所有词）。SentencePiece更像把句子当作积木——它自动发现最常用的"积木块"（子词），用这些积木块组合出所有句子。不认识的词就用多个小积木块拼接。

**历史背景**：SentencePiece由Google的Kudo和Richardson在2018年提出。它解决了多语言分词的统一问题——不需要针对每种语言开发不同的分词器。T5、LLaMA、DeepSeek等模型都使用SentencePiece或其变体进行分词。

**算法定位**：NLP / 文本预处理 / 子词分词。

**前置知识**：
- 文本分词的概念
- BPE（Byte Pair Encoding）算法
- 概率论基础

## 2. 核心原理

### 核心思想

SentencePiece将分词视为**无监督的文本压缩问题**：

1. 将文本视为字节序列（而非字符序列），使其语言无关
2. 使用BPE或Unigram算法从数据中自动学习最优的子词词表
3. 分词时用学到的子词单元切分文本

### 两种核心算法

**BPE（Byte Pair Encoding）**：
- 从字符级别开始，反复合并最高频的字符对
- 例如："h e l l o" → "he l l o" → "hel l o" → "hell o" → "hello"

**Unigram Language Model**：
- 从大词表开始，逐步删除对总体似然贡献最小的子词
- 基于概率模型选择最优分词方案

### 工作流程

1. **训练阶段**：
   - 输入大量文本
   - 预处理：将文本转为字节序列，添加特殊标记
   - 运行BPE或Unigram算法学习词表
   - 输出词表文件（.model + .vocab）

2. **编码阶段**：
   - 输入原始文本
   - 使用学到的词表将文本编码为token ID序列

3. **解码阶段**：
   - 输入token ID序列
   - 还原为原始文本

### 关键概念

- **子词(Subword)**：介于字符和完整词之间的单元。如"un"+"believ"+"able"
- **字节回退(Byte-fallback)**：遇到未知子词时，退回到UTF-8字节级别
- **特殊token**：`<unk>`（未知）、`<s>`（开始）、`</s>`（结束）、`<pad>`（填充）
- **词表大小**：通常32K-128K，影响模型的表达力和嵌入层大小

## 3. 数学公式与推导

### BPE合并规则

初始词表为所有单字符。在第 $i$ 步合并频率最高的字符对 $(a, b)$：

$$\text{freq}(a, b) = \text{count}(a \cdot b)$$

合并后，新符号 $ab$ 加入词表，所有 $a \cdot b$ 的出现被替换为 $ab$。

### Unigram模型

给定词表 $V$，文本 $X$ 的最优分词 $\mathbf{s}^*$ 最大化：

$$\mathbf{s}^* = \arg\max_{\mathbf{s}} P(\mathbf{s}) = \arg\max_{\mathbf{s}} \prod_{i} p(s_i)$$

其中 $p(s_i)$ 是子词 $s_i$ 的概率。使用Viterbi算法求解最优分词。

### 词表裁剪

Unigram算法的裁剪标准：删除使总体对数似然增加最小的子词 $x$：

$$\Delta \mathcal{L}(x) = \sum_{\mathbf{s} \in S(x)} P(\mathbf{s}|X) \log P(\mathbf{s}|X)$$

其中 $S(x)$ 是所有包含 $x$ 的分词方案。

## 4. 训练过程讲解

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| 词表大小 | 子词单元数量 | 32000-128000 | 32000 |
| 模型类型 | BPE或Unigram | - | unigram |
| 字符覆盖率 | 处理的字符比例 | 0.9995-1.0 | 0.9995 |
| 最大句长 | 输入句子最大长度 | - | 4192 |

## 5. 应用场景

1. **大语言模型分词**：LLaMA使用BPE SentencePiece，词表大小32K。DeepSeek使用类似的分词策略。

2. **多语言模型**：SentencePiece的语言无关特性使其适合多语言NLP任务。

3. **机器翻译**：在源语言和目标语言上联合训练SentencePiece词表。

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 语言无关，支持任何语言 | 词表大小需要手动设定 |
| 处理OOV（未知词）能力强 | 训练需要大量文本数据 |
| 与模型解耦，可独立训练 | 不同语言分词效率不均等 |
| 支持字节回退 | 中文分词可能不如专业分词器 |

## 7. 调库实现

```python
"""使用 sentencepiece 库进行分词"""
import os
import tempfile

# 注意：运行前需要安装 sentencepiece: pip install sentencepiece

def train_sentencepiece_demo():
    """训练SentencePiece模型的示例"""
    
    # 创建临时训练数据
    train_text = """
    DeepSeek是一个大语言模型，具有强大的推理能力。
    The transformer architecture revolutionized natural language processing.
    多模态模型可以同时处理文本、图像和音频信息。
    Attention is all you need, this paper changed everything.
    大模型的训练需要大量的计算资源和数据。
    """
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(train_text)
        train_file = f.name
    
    model_prefix = os.path.join(tempfile.gettempdir(), 'spm_model')
    
    # 训练SentencePiece模型（实际使用时需要更大的数据量）
    try:
        import sentencepiece as spm
        spm.SentencePieceTrainer.train(
            input=train_file,
            model_prefix=model_prefix,
            vocab_size=200,  # 演示用小词表，实际通常32000+
            model_type='bpe',
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>'
        )
        
        # 加载模型
        sp = spm.SentencePieceProcessor()
        sp.load(f'{model_prefix}.model')
        
        print("=== SentencePiece 分词测试 ===")
        text = "DeepSeek是一个大语言模型"
        
        # 编码为token IDs
        ids = sp.encode(text, out_type=int)
        print(f"原文: {text}")
        print(f"Token IDs: {ids}")
        
        # 编码为子词
        pieces = sp.encode(text, out_type=str)
        print(f"子词: {pieces}")
        
        # 解码
        decoded = sp.decode(ids)
        print(f"解码: {decoded}")
        
        # 词表信息
        print(f"\n词表大小: {sp.get_piece_size()}")
        
        # ID与子词互查
        for piece in pieces[:5]:
            pid = sp.piece_to_id(piece)
            print(f"  '{piece}' -> ID {pid}")
        
        # 清理临时文件
        os.unlink(train_file)
        for ext in ['.model', '.vocab']:
            path = f'{model_prefix}{ext}'
            if os.path.exists(path):
                os.unlink(path)
    
    except ImportError:
        print("请安装sentencepiece: pip install sentencepiece")
        os.unlink(train_file)


# ====== 测试 ======
if __name__ == "__main__":
    train_sentencepiece_demo()
```

## 8. 手工代码实现

```python
"""从零实现简化版BPE分词器"""
import re
from collections import Counter


class ManualBPE:
    """手写BPE（Byte Pair Encoding）分词器"""
    
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.merges = []  # 合并规则列表
        self.vocab = {}   # 最终词表
    
    def _get_pairs(self, word):
        """获取词中所有相邻字符对"""
        pairs = []
        for i in range(len(word) - 1):
            pairs.append((word[i], word[i+1]))
        return pairs
    
    def train(self, texts):
        """
        训练BPE：从文本中学习合并规则
        texts: 文本列表
        """
        # 统计所有词的频率
        word_freqs = Counter()
        for text in texts:
            # 简化：按空格分词，每个"词"末尾加</w>标记
            for word in text.split():
                word_freqs[tuple(word) + ('</w>',)] += 1
        
        # 初始词表：所有单字符
        vocab = set()
        for word in word_freqs:
            vocab.update(word)
        
        # 迭代合并最高频的字符对
        for _ in range(self.vocab_size - len(vocab)):
            # 统计所有相邻对的频率
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                pairs = self._get_pairs(list(word))
                for pair in pairs:
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # 找最高频的对
            best_pair = pair_freqs.most_common(1)[0][0]
            self.merges.append(best_pair)
            
            # 在所有词中合并该对
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = self._merge_pair(word, best_pair)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs
        
        # 构建最终词表
        self.vocab = set(vocab)
        for word in word_freqs:
            self.vocab.update(word)
    
    def _merge_pair(self, word, pair):
        """在词中合并指定的字符对"""
        new_word = []
        i = 0
        word_list = list(word)
        while i < len(word_list):
            if i < len(word_list) - 1 and word_list[i] == pair[0] and word_list[i+1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word_list[i])
                i += 1
        return tuple(new_word)
    
    def tokenize(self, text):
        """使用学到的BPE规则分词"""
        tokens = []
        for word in text.split():
            word_tokens = list(word) + ['</w>']
            
            # 按顺序应用所有合并规则
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == merge[0] and word_tokens[i+1] == merge[1]:
                        word_tokens = word_tokens[:i] + [merge[0]+merge[1]] + word_tokens[i+2:]
                    else:
                        i += 1
            
            tokens.extend(word_tokens)
        
        return tokens


# ====== 测试 ======
if __name__ == "__main__":
    # 训练数据
    texts = [
        "the cat sat on the mat",
        "the dog ran to the cat",
        "the cat and the dog sat",
        "a cat is on the mat",
        "the dog sat on the mat"
    ]
    
    # 训练BPE
    bpe = ManualBPE(vocab_size=30)
    bpe.train(texts)
    
    print("=== 手写BPE分词测试 ===")
    print(f"学到的合并规则 ({len(bpe.merges)} 条):")
    for i, merge in enumerate(bpe.merges[:10]):
        print(f"  {i+1}. '{merge[0]}' + '{merge[1]}' -> '{merge[0]+merge[1]}'")
    
    # 分词测试
    test = "the cat sat on the mat"
    tokens = bpe.tokenize(test)
    print(f"\n原文: {test}")
    print(f"分词: {tokens}")
```

## 9. 可视化与结果理解

```python
"""分词效果可视化"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 不同词表大小的压缩率
vocab_sizes = [1000, 4000, 8000, 16000, 32000, 64000, 128000]
compression = [2.1, 3.5, 4.2, 4.8, 5.2, 5.4, 5.5]  # 字符/token比

axes[0].plot(vocab_sizes, compression, 'o-', color='#3498db', linewidth=2)
axes[0].set_title('词表大小 vs 压缩率', fontsize=13)
axes[0].set_xlabel('词表大小')
axes[0].set_ylabel('平均字符/token比（越低越好）')
axes[0].set_xscale('log')
axes[0].grid(True, alpha=0.3)

# 图2: 不同语言的分词效率
languages = ['英文', '中文', '日文', '韩文', '代码']
tokens_per_char = [0.25, 0.5, 0.6, 0.45, 0.35]

axes[1].bar(languages, tokens_per_char, color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6'])
axes[1].set_title('不同语言的平均Token/字符比', fontsize=13)
axes[1].set_ylabel('Token/字符比')
for i, v in enumerate(tokens_per_char):
    axes[1].text(i, v + 0.01, f'{v}', ha='center')

# 图3: BPE合并频率
np.random.seed(42)
merge_steps = range(1, 21)
freqs = [1000 / i**0.8 for i in merge_steps]  # 模拟Zipf分布

axes[2].bar(merge_steps, freqs, color='#2ecc71', edgecolor='black')
axes[2].set_title('BPE合并步骤的频率衰减', fontsize=13)
axes[2].set_xlabel('合并步骤')
axes[2].set_ylabel('被合并对的频率')
axes[2].set_yscale('log')

plt.tight_layout()
plt.savefig('sentencepiece_viz.png', dpi=100)
plt.show()

print("图1解读: 词表越大压缩率越高, 但32K后收益递减")
print("图2解读: 中文/日文分词效率低于英文(需要更多token表示)")
print("图3解读: 早期合并的字符对频率远高于后期(Zipf分布)")
```

## 10. 模型评估

```python
"""评估分词器质量"""
def evaluate_tokenizer(texts, tokenize_fn):
    """评估分词器的压缩率和覆盖率"""
    total_chars = sum(len(t) for t in texts)
    total_tokens = sum(len(tokenize_fn(t)) for t in texts)
    
    compression = total_chars / total_tokens
    unique_tokens = len(set(token for t in texts for token in tokenize_fn(t)))
    
    print(f"=== 分词器评估 ===")
    print(f"总字符数: {total_chars}")
    print(f"总Token数: {total_tokens}")
    print(f"压缩率: {compression:.2f} 字符/token")
    print(f"唯一Token数: {unique_tokens}")
```

## 11. 常见问题与易错点

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 中文分词token数过多 | 训练和推理慢 | 中文字符多，词表不够 | 增大词表或使用字节回退 |
| 特殊token冲突 | 编解码不一致 | 训练和推理的特殊token不同 | 统一特殊token定义 |

## 12. 学习总结

SentencePiece的核心：将文本视为字节序列，通过BPE或Unigram算法学习最优子词词表。

关键公式：
- BPE：反复合并最高频字符对
- Unigram：$s^* = \arg\max_s \prod_i p(s_i)$

SentencePiece是大模型训练的基础组件，直接影响模型的输入表示和推理效率。

## 13. 练习题与思考题

### 基础题1：BPE合并

给定文本 "low low low lower lower newest newest"，手动执行3步BPE合并。

**参考答案**：
初始：l o w, l o w, l o w, l o w e r, l o w e r, n e w e s t, n e w e s t
- Step 1：合并最高频对 (l, o) → "lo"
  lo w, lo w, lo w, lo w e r, lo w e r, n e w e s t, n e w e s t
- Step 2：合并 (lo, w) → "low"
  low, low, low, low e r, low e r, n e w e s t, n e w e s t
- Step 3：合并 (e, s) → "es"（频率2）
  low, low, low, low e r, low e r, n e w es t, n e w es t

### 基础题2：压缩率

一个模型词表大小32K，处理英文文本平均每字符0.25个token。一段1000字符的英文文本需要多少token？如果是中文（平均0.5 token/字符）呢？

**参考答案**：
- 英文：1000 × 0.25 = 250 tokens
- 中文：1000 × 0.5 = 500 tokens
- 中文需要的token是英文的2倍，意味着中文的处理效率更低

### 进阶题：字节级BPE

为什么字节级BPE（如GPT-2使用的）比字符级BPE更适合多语言场景？

**参考答案**：
1. 字节级BPE的初始词表固定为256个UTF-8字节，不需要预定义字符集
2. 任何语言的文本都可以表示为UTF-8字节序列，无需特殊处理
3. 字符级BPE需要为每种语言定义字符集，遇到新字符会失败
4. 字节级BPE通过合并学习到有意义的子词单元，同时保证100%覆盖率

### 开放思考题

随着多模态大模型的发展，视觉token和文本token被统一处理。未来的分词器是否可能发展为"通用token化器"——将图像、音频、文本统一编码？

**参考思路**：
这正在发生：
1. **VQ-VAE/VQ-GAN**：将图像编码为离散token，与文本token统一
2. **音频编码**：将音频频谱编码为离散token（如AudioLM）
3. **统一词表**：未来的模型可能有一个跨模态的统一token词表
4. 挑战：不同模态的信息密度差异大（图像需要更多token），需要解决效率问题

## 14. 学习路径建议

### 前置知识
- 文本预处理
- 字符编码（UTF-8）
- 信息论基础（熵、压缩）

### 平行学习
- Word2Vec（另一种词表示方法）
- BERT的WordPiece分词

### 进阶方向
- Tiktoken（OpenAI的分词器）
- 多语言分词优化
- 视觉token化（VQ-VAE的codebook作为"视觉分词"）

### 推荐资源
1. **论文**：SentencePiece: A simple and language independent subword tokenizer (Kudo & Richardson, 2018)
2. **论文**：Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016) — BPE原始论文
3. **库**：Google SentencePiece官方GitHub仓库
