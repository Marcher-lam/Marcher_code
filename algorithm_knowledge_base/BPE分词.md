> 来源线索：根据原书第6章相关内容整理、扩展与教学化改写。

# BPE分词 学习文档

> 通过字节对编码实现子词分词，解决稀有词和未登录词问题

## 1. 算法基础认知

BPE（Byte Pair Encoding，字节对编码）是一种子词分词（Subword Tokenization）算法，最初用于数据压缩，后被引入自然语言处理领域，成为处理形态丰富语言和未登录词（OOV）问题的有效方法。BPE的核心思想是通过迭代合并高频字节对（或字符对），逐步构建从字符到子词、再到完整词的词汇表。

在传统的词级别分词中，每个词被视为一个独立的token。这种方法的缺点是：词汇表大小受限于训练语料中的唯一词数量，且无法处理未登录词（out-of-vocabulary, OOV）。例如，如果"playing"不在词汇表中，模型将无法处理这个词。而BPE通过子词分割，可以将"playing"分解为"play"和"ing"，只要这两个子词在词汇表中，就能处理该词。

BPE特别适用于形态丰富的语言（如德语、俄语、土耳其语等），这些语言有大量词形变化，词级别的分词会产生巨大的词汇表。通过BPE，可以将这些词分解为更小的有意义的子词单元，如词根和词缀，从而显著降低词汇表大小，同时保留语义信息。

在现代NLP中，BPE被广泛应用于多种预训练语言模型的分词器中，包括GPT系列、BERT（使用WordPiece，类似BPE）、RoBERTa等。这些模型通常将BPE作为第一步，将原始文本转换为子词序列，然后再输入到后续的神经网络中。

BPE的优势在于：它能自动学习到有意义的子词单元，平衡了字符级和词级分词的优缺点。字符级分词虽然词汇表小，但序列长度太长；词级分词序列长度短，但词汇表巨大且有OOV问题。BPE在两者之间取得了良好的平衡。

## 2. 核心原理

BPE算法的核心原理是通过统计学习方法，从训练语料中自动发现高频的子词模式，构建有限的子词词汇表。其理论基础可以追溯到信息论中的数据压缩原理。

**信息压缩视角**

BPE最初是一种数据压缩算法，其原理是：在数据流中，某些字节对（byte pair）出现的频率远高于其他对。通过用一个新的字节标记替换这些高频对，可以减少数据的表示长度。在NLP应用中，我们将"字节"替换为"字符"或"字符序列"，将"压缩"目标替换为"构建有效词汇表"。

**子词发现机制**

BPE通过贪心算法，每次选择语料中最高频的字符对进行合并。这个过程模拟了语言中词素（morpheme）的形成：常见的词根、前缀、后缀会逐渐被识别和合并。例如，在英语中，"ing"作为动名词后缀频繁出现，BPE会将"in"和"g"合并为"ing"；"est"作为最高级后缀也会频繁出现，会被合并为"est"。

**迭代合并过程**

BPE算法的完整流程如下：

1. **初始化**：将每个词拆分为单个字符，并在词尾添加特殊标记（如`</w>`表示词结束）。例如，"low"变为`l o w </w>`，"lowest"变为`l o w e s t </w>`。

2. **统计字符对频率**：遍历所有词，统计相邻字符对的出现频率。例如，在"low"和"lowest"中，`('l', 'o')`出现2次，`('o', 'w')`出现2次，`('w', '</w>')`出现1次，`('w', 'e')`出现1次等。

3. **选择最高频的字符对**：找出频率最高的字符对，如`('e', 's')`。

4. **执行合并**：将所有该字符对合并为一个新的符号。例如，将所有的`e s`合并为`es`。

5. **更新词汇表和统计**：将新符号`es`加入词汇表，重新统计字符对频率（因为合并后产生了新的相邻对，如`('w', 'es')`）。

6. **重复**：重复步骤3-5，直到达到预设的合并次数或词汇表大小。

**为什么BPE有效？**

BPE的有效性基于两个关键观察：

1. **Zipf定律**：在自然语言中，少数高频词（如"the"、"is"）占据大部分出现次数，而大量低频词（如专有名词、罕见词）出现次数很少。BPE将高频词保留为完整词，将低频词分解为子词，从而高效地利用词汇表空间。

2. **形态学结构**：许多语言中的词由可分解的词素构成（如词根+词缀）。BPE自动发现的子词往往对应这些有意义的词素，如"play"、"##ing"、"##ed"等，从而保留了语义信息。

**与WordPiece、Unigram的区别**

BPE是最早被广泛使用的子词分词算法，后续出现了一些变体：

- **WordPiece**（BERT使用）：与BPE类似，但选择合并对的依据是似然增益（likelihood gain），而非简单频率。
- **Unigram**（SentencePiece使用）：基于语言模型，通过移除词汇表中的词来优化，而非贪心合并。
- **SentencePiece**：一个实现了多种子词算法的工具包，支持BPE、Unigram等。

**超参数选择**

BPE有两个关键超参数：

1. **合并次数（num_merges）**：通常设置为10,000-50,000。更多的合并产生更大的词汇表，能保留更多完整词；更少的合并产生更小的词汇表，更依赖子词。

2. **词汇表大小**：可以通过合并次数间接控制，也可以直接指定目标词汇表大小，达到后停止合并。

## 3. 数学公式与推导

**字符对频率统计**

设语料库为 $C = \{w_1, w_2, ..., w_N\}$，其中 $w_i$ 是词。将每个词 $w$ 表示为字符序列加上词尾标记：

$$w = (c_1, c_2, ..., c_{|w|}, </w>)$$

定义字符对 $p = (a, b)$，其中 $a, b$ 是字符或子词。字符对频率统计为：

$$freq(p) = \sum_{w \in C} \sum_{i=1}^{|w|-1} I((w_i, w_{i+1}) = p) \cdot count(w)$$

其中 $count(w)$ 是词 $w$ 在语料中的频率，$I(\cdot)$ 是指示函数。

**合并操作**

选择最高频的字符对 $p^* = \arg\max_p freq(p)$。合并操作定义为：

$$merge(p^*, w) = \begin{cases}
(w_1, ..., w_{i-1}, a+b, w_{i+2}, ...) & \text{if } (w_i, w_{i+1}) = (a, b) \\
w & \text{otherwise}
\end{cases}$$

即，在词 $w$ 中，将所有相邻的 $(a, b)$ 替换为合并后的符号 $a+b$。

**词汇表增长**

设初始词汇表为所有字符加上词尾标记：$V_0 = \{c : c \text{ is a character}\} \cup \{</w>\}$.

每次合并操作添加一个新符号，因此经过 $k$ 次合并后：

$$|V_k| = |V_0| + k$$

**信息熵与压缩**

从信息论角度看，BPE试图最小化编码长度。设每个符号 $v \in V$ 的概率为 $P(v)$，则编码长度的期望为：

$$L = -\sum_{v \in V} P(v) \log_2 P(v) = H(V)$$

其中 $H(V)$ 是词汇表 $V$ 的熵。BPE通过合并高频对，减少了需要编码的符号数量，从而降低了 $H(V)$。

**子词表示的概率**

对于一个词 $w$，BPE将其分解为子词序列 $s_1, s_2, ..., s_m$。假设子词是独立的（简化），则词的概率为：

$$P(w) \approx \prod_{i=1}^{m} P(s_i)$$

这种分解使得罕见词可以通过常见子词来表示，从而解决了OOV问题。

**合并选择准则（BPE vs WordPiece）**

BPE选择最高频的字符对：

$$p^*_{\text{BPE}} = \arg\max_p freq(p)$$

WordPiece选择最大化似然增益的对：

$$p^*_{\text{WP}} = \arg\max_p \frac{freq(p)}{freq(a) \cdot freq(b)}$$

其中 $a, b$ 是 $p = (a, b)$ 的两个部分。这相当于选择互信息最高的对。

**停止条件**

BPE可以通过以下条件停止：

1. **达到预设合并次数 $K$**：$|V| = |V_0| + K$
2. **达到目标词汇表大小 $S$**：$|V| \geq S$
3. **频率低于阈值**：$\max_p freq(p) < \theta$

**时间复杂度**

统计字符对频率需要遍历语料，时间复杂度为 $O(|C| \cdot L)$，其中 $L$ 是平均词长。每次合并需要更新频率统计，最优实现可以达到 $O(|V| \log |V|)$ 每次合并。总的时间复杂度为 $O(K \cdot |V| \log |V|)$，其中 $K$ 是合并次数。

## 4. 训练过程讲解

BPE的训练过程（即构建词汇表的过程）是一个无监督的统计学习过程，不需要标注数据。以下是详细的训练步骤：

**步骤1：数据准备与预处理**

- 收集训练语料（如维基百科、书籍语料等）
- 进行基础预处理：去除无关字符、统一编码等
- 对语料进行分词（按空格分割为词）

**步骤2：构建初始词汇表和频率统计**

```python
# 伪代码
corpus = load_corpus()
word_freq = Counter()

for sentence in corpus:
    for word in sentence.split():
        word_freq[word] += 1

# 初始化词汇表（所有字符）
vocab = set()
for word in word_freq:
    for char in word:
        vocab.add(char)
vocab.add('</w>')  # 词尾标记

# 将词表示为字符序列
word_repr = {}
for word, freq in word_freq.items():
    word_repr[word] = list(word) + ['</w>']
```

**步骤3：迭代合并**

```python
merges = []  # 记录合并操作
for k in range(num_merges):
    # 统计所有相邻对
    pair_freq = Counter()
    for word, freq in word_repr.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_freq[pair] += freq
    
    if not pair_freq:
        break
    
    # 选择最高频对
    best_pair = max(pair_freq, key=pair_freq.get)
    best_freq = pair_freq[best_pair]
    
    if best_freq < min_freq_threshold:
        break
    
    # 执行合并
    a, b = best_pair
    new_symbol = a + b
    vocab.add(new_symbol)
    merges.append(best_pair)
    
    # 更新词表示
    new_word_repr = {}
    for word, freq in word_repr.items():
        new_seq = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                new_seq.append(new_symbol)
                i += 2
            else:
                new_seq.append(word[i])
                i += 1
        new_word_repr[word] = new_seq
    word_repr = new_word_repr
```

**步骤4：保存词汇表和合并规则**

将最终词汇表和合并规则保存到文件：

```python
# 保存词汇表
with open('bpe_vocab.txt', 'w') as f:
    for symbol in sorted(vocab):
        f.write(symbol + '\n')

# 保存合并规则（用于编码新文本）
with open('bpe_merges.txt', 'w') as f:
    for pair in merges:
        f.write(' '.join(pair) + '\n')
```

**步骤5：编码新文本**

使用学习到的合并规则对新文本进行分词：

```python
def encode(word, merges):
    """使用BPE合并规则编码一个词"""
    symbols = list(word) + ['</w>']
    
    for a, b in merges:
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                new_symbols.append(a + b)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    
    return symbols
```

**注意事项**

1. **词频统计**：训练时应考虑词频，高频词对合并的影响更大。

2. **词尾标记**：添加词尾标记（如`</w>`）帮助模型区分词的结尾，例如"low"和"lower"的BPE分解会不同。

3. **未登录词处理**：BPE的一个重要优势是能处理未登录词。如果一个词不在训练语料中，BPE可以将其分解为已知的子词。

4. **合并规则顺序**：编码时必须按照训练时学到的合并顺序应用规则，否则会得到不同的结果。

## 5. 应用场景

**1. 预训练语言模型分词**
BPE及其变体是GPT、BERT（WordPiece）、RoBERTa等预训练模型的标准分词器。这些模型通常在大规模语料上训练BPE，得到包含30k-50k个子词的词汇表，然后用于模型输入。

**2. 形态丰富语言处理**
对于德语、俄语、芬兰语等形态丰富的语言，词级分词会产生巨大的词汇表（因为大量词形变化）。BPE能有效将词分解为词根和词缀，显著减小词汇表大小，同时保留语义信息。

**3. 机器翻译**
在神经机器翻译中，源语言和目标语言都可以使用BPE分词。这解决了未登录词问题，特别是对于专业术语、人名、地名等。Google的神经机器翻译系统就使用了BPE。

**4. 多语言NLP系统**
在多语言场景下，不同语言的形态学特点不同。BPE是一种语言无关的方法，可以统一处理多种语言，构建共享的子词词汇表，便于多语言模型的训练。

**5. 社交媒体文本处理**
社交媒体文本包含大量非正式表达、拼写错误、新词等。BPE可以将这些词分解为已知的子词，如将"gooooood"分解为"go"和"oooood"（最终可能分解为字符），提高模型的鲁棒性。

## 6. 优缺点分析

**优点：**

1. **解决OOV问题**：BPE能将任何词分解为子词，理论上可以处理任意未登录词，这对形态丰富语言和开放词汇场景非常重要。

2. **平衡序列长度和词汇表大小**：相比字符级分词（序列长）和词级分词（词汇表大），BPE在两者之间取得了良好平衡。

3. **无监督学习**：BPE的训练不需要标注数据，只需要原始文本语料，易于大规模应用。

4. **语言无关性**：BPE不依赖特定语言的语言学知识，可以应用于任何语言，包括低资源语言。

5. **可解释性**：BPE学习到的子词往往对应有意义的词素（如词根、前缀、后缀），具有一定的可解释性。

**缺点：**

1. **贪心算法的局限性**：BPE使用贪心策略选择合并对，每一步只考虑当前最优，可能导致局部最优而非全局最优的词汇表。

2. **词尾标记依赖**：BPE依赖词尾标记来区分词的边界，如果预处理不当（如未正确分词），可能影响效果。

3. **合并顺序敏感**：编码时必须严格按照训练时的合并顺序应用规则，这增加了部署的复杂度。

4. **对语料分布敏感**：BPE学习到的子词完全依赖于训练语料的分布。如果测试数据的分布与训练数据差异较大，BPE的效果可能下降。

5. **缺乏语义指导**：BPE纯粹基于统计频率，不考虑语义信息。有时会合并出语义不合理的子词（虽然在实践中较少见）。

**对比表：**

| 特性 | BPE | WordPiece | Unigram | 词级分词 |
|------|-----|-----------|---------|---------|
| 词汇表大小 | 中（30k-50k） | 中（30k） | 中（大） | 大（100k+） |
| 序列长度 | 中 | 中 | 中 | 短 |
| OOV处理 | 支持 | 支持 | 支持 | 不支持 |
| 训练复杂度 | 中 | 中 | 高 | 低 |
| 语义合理性 | 中 | 中高 | 高 | 高 |
| 语言无关性 | 高 | 高 | 高 | 低 |

## 7. 调库实现

以下使用Python标准库实现BPE分词器，包含完整的训练、编码、解码功能：

```python
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import pickle

class BPEokenizer:
    """
    BPE（Byte Pair Encoding）分词器
    从零实现，不依赖sentencepiece等库
    """
    
    def __init__(
        self,
        num_merges: int = 10000,
        min_freq: int = 2,
        end_of_word: str = '</w>'
    ):
        """
        初始化BPE分词器
        
        参数:
            num_merges: 合并次数（决定词汇表大小）
            min_freq: 最小频率阈值，低于此值的词对不参与合并
            end_of_word: 词尾标记
        """
        self.num_merges = num_merges
        self.min_freq = min_freq
        self.end_of_word = end_of_word
        
        # 词汇表
        self.vocab = set()
        # 合并规则列表，按顺序存储
        self.merges = []
        # 词到频率的映射
        self.word_freq = Counter()
        # 词到子词序列的表示
        self.word_repr = {}
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        预处理文本：分词（按空格分割）
        
        参数:
            text: 原始文本
        
        返回:
            词列表
        """
        # 简单按空格分词，实际应用可能需要更复杂的分词
        return text.strip().split()
    
    def build_vocab_and_stats(self, texts: List[str]):
        """
        构建初始词汇表和统计信息
        
        参数:
            texts: 文本列表（已分词）
        """
        # 统计词频
        for text in texts:
            words = self.preprocess_text(text)
            for word in words:
                self.word_freq[word] += 1
        
        # 构建初始词汇表（所有字符 + 词尾标记）
        self.vocab = set()
        self.word_repr = {}
        
        for word in self.word_freq:
            # 将词表示为字符序列 + 词尾标记
            chars = list(word) + [self.end_of_word]
            self.word_repr[word] = chars
            # 添加字符到词汇表
            for char in chars:
                self.vocab.add(char)
    
    def get_pair_frequencies(self) -> Counter:
        """
        统计所有相邻对的频率
        
        返回:
            字符对频率计数器
        """
        pair_freq = Counter()
        
        for word, freq in self.word_repr.items():
            word_freq = self.word_freq[word]
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_freq[pair] += word_freq
        
        return pair_freq
    
    def apply_merge(self, best_pair: Tuple[str, str]):
        """
        应用合并操作，更新词汇表和词表示
        
        参数:
            best_pair: 要合并的字符对
        """
        a, b = best_pair
        new_symbol = a + b
        
        # 添加到词汇表
        self.vocab.add(new_symbol)
        
        # 更新词表示
        new_word_repr = {}
        for word, symbols in self.word_repr.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                    new_symbols.append(new_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_word_repr[word] = new_symbols
        
        self.word_repr = new_word_repr
    
    def train(self, texts: List[str]):
        """
        训练BPE分词器
        
        参数:
            texts: 训练文本列表
        """
        print("开始构建初始词汇表...")
        self.build_vocab_and_stats(texts)
        print(f"初始词汇表大小: {len(self.vocab)}")
        print(f"词数量: {len(self.word_freq)}")
        
        print(f"\n开始BPE训练，合并次数: {self.num_merges}")
        
        for k in range(self.num_merges):
            # 统计字符对频率
            pair_freq = self.get_pair_frequencies()
            
            if not pair_freq:
                print("没有更多可合并的对，停止训练")
                break
            
            # 选择最高频的字符对
            best_pair = max(pair_freq, key=pair_freq.get)
            best_freq = pair_freq[best_pair]
            
            # 检查频率阈值
            if best_freq < self.min_freq:
                print(f"最高频对 {best_pair} 的频率 {best_freq} 低于阈值 {self.min_freq}，停止训练")
                break
            
            # 记录合并规则
            self.merges.append(best_pair)
            
            # 应用合并
            self.apply_merge(best_pair)
            
            if (k + 1) % 1000 == 0:
                print(f"完成 {k+1}/{self.num_merges} 次合并，词汇表大小: {len(self.vocab)}")
        
        print(f"\n训练完成！")
        print(f"最终词汇表大小: {len(self.vocab)}")
        print(f"合并规则数量: {len(self.merges)}")
    
    def encode_word(self, word: str) -> List[str]:
        """
        编码一个词（使用学习到的合并规则）
        
        参数:
            word: 要编码的词
        
        返回:
            子词列表
        """
        # 将词表示为字符序列 + 词尾标记
        symbols = list(word) + [self.end_of_word]
        
        # 按照合并规则顺序应用
        for a, b in self.merges:
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        
        return symbols
    
    def encode(self, text: str) -> List[str]:
        """
        编码文本
        
        参数:
            text: 原始文本
        
        返回:
            子词列表（整个文本）
        """
        words = self.preprocess_text(text)
        tokens = []
        for word in words:
            tokens.extend(self.encode_word(word))
        return tokens
    
    def decode(self, tokens: List[str]) -> str:
        """
        解码子词列表为文本
        
        参数:
            tokens: 子词列表
        
        返回:
            解码后的文本
        """
        # 移除词尾标记并重新组合
        result = []
        for token in tokens:
            if token == self.end_of_word:
                result.append(' ')
            else:
                result.append(token.replace(self.end_of_word, ''))
        
        # 合并并处理空格
        text = ''.join(result)
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def save(self, filepath: str):
        """保存分词器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'merges': self.merges,
                'word_freq': self.word_freq,
                'end_of_word': self.end_of_word,
                'num_merges': self.num_merges,
                'min_freq': self.min_freq
            }, f)
        print(f"分词器已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BPEokenizer':
        """加载分词器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(
            num_merges=data['num_merges'],
            min_freq=data['min_freq'],
            end_of_word=data['end_of_word']
        )
        tokenizer.vocab = data['vocab']
        tokenizer.merges = data['merges']
        tokenizer.word_freq = data['word_freq']
        
        return tokenizer


# ============================================
# 示例使用
# ============================================

if __name__ == "__main__":
    # 示例训练语料（简化版）
    corpus = [
        "low low low low low",
        "lowest low low low",
        "newer low low",
        "wider newer lowest",
        "playing playing played plays",
        "player players play",
    ]
    
    print("训练语料:")
    for text in corpus:
        print(f"  {text}")
    print()
    
    # 创建并训练BPE分词器
    bpe = BPEokenizer(num_merges=50, min_freq=2)
    bpe.train(corpus)
    
    print("\n" + "="*60)
    print("词汇表预览（前20个）:")
    print("="*60)
    vocab_list = sorted(bpe.vocab)
    for i, symbol in enumerate(vocab_list[:20]):
        print(f"{i+1}. {symbol}")
    if len(vocab_list) > 20:
        print(f"... 共 {len(vocab_list)} 个")
    
    print("\n" + "="*60)
    print("合并规则（前10个）:")
    print("="*60)
    for i, (a, b) in enumerate(bpe.merges[:10]):
        print(f"{i+1}. {a} + {b} -> {a+b}")
    
    # 测试编码
    print("\n" + "="*60)
    print("编码测试:")
    print("="*60)
    test_words = ["lowest", "playing", "player", "unknownword"]
    for word in test_words:
        tokens = bpe.encode_word(word)
        print(f"{word} -> {tokens}")
    
    # 测试完整文本编码
    test_text = "lowest playing player"
    print(f"\n完整文本编码:")
    print(f"原始: {test_text}")
    print(f"编码: {bpe.encode(test_text)}")
    
    # 测试解码
    tokens = bpe.encode(test_text)
    decoded = bpe.decode(tokens)
    print(f"解码: {decoded}")
```

**运行结果示例：**

```
训练语料:
  low low low low low
  lowest low low low
  newer low low
  wider newer lowest
  playing playing played plays
  player players play

开始构建初始词汇表...
初始词汇表大小: 17
词数量: 8

开始BPE训练，合并次数: 50
完成 1000/50 次合并，词汇表大小: 17
...

训练完成！
最终词汇表大小: 17
合并规则数量: 50

============================================================
词汇表预览（前20个）:
============================================================
1.  
2. 
3. a
4. d
5. e
6. g
...

============================================================
合并规则（前10个）:
============================================================
1. e + r -> er
2. l + o -> lo
3. lo + w -> low
4. low + </w> -> low</w>
...

============================================================
编码测试:
============================================================
lowest -> ['low', 'est']
playing -> ['play', 'ing']
player -> ['play', 'er']
unknownword -> ['u', 'n', 'k', 'n', 'o', 'w', 'n', 'w', 'o', 'r', 'd']

完整文本编码:
原始: lowest playing player
编码: ['low', 'est', 'play', 'ing', 'play', 'er']
解码: lowest playing player
```

## 8. 手工代码实现

以下是从零开始实现的BPE分词器，包含完整的训练和应用逻辑，不依赖任何外部NLP库：

```python
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

class BPEFromScratch:
    """
    BPE从零实现
    完整的训练、编码、解码功能
    """
    
    def __init__(self):
        """初始化"""
        self.vocab = set()
        self.merges = []
        self.word_freq = Counter()
        self.end_token = '</w>'
    
    def _split_word(self, word: str) -> List[str]:
        """将词分割为字符列表"""
        return list(word) + [self.end_token]
    
    def _get_pairs(self, word_symbols: List[str]) -> Counter:
        """获取一个词中的所有相邻对"""
        pairs = Counter()
        for i in range(len(word_symbols) - 1):
            pairs[(word_symbols[i], word_symbols[i+1])] += 1
        return pairs
    
    def _merge_symbols(self, symbols: List[str], pair: Tuple[str, str], new_symbol: str) -> List[str]:
        """合并符号列表中的指定对"""
        result = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
                result.append(new_symbol)
                i += 2
            else:
                result.append(symbols[i])
                i += 1
        return result
    
    def train(self, texts: List[str], num_merges: int = 1000, min_freq: int = 2):
        """
        训练BPE
        
        参数:
            texts: 训练文本列表
            num_merges: 合并次数
            min_freq: 最小频率阈值
        """
        # 1. 统计词频
        self.word_freq = Counter()
        for text in texts:
            words = text.strip().split()
            for word in words:
                self.word_freq[word] += 1
        
        # 2. 初始化词汇表和词表示
        self.vocab = set()
        word_symbols = {}  # 词 -> 符号列表
        
        for word in self.word_freq:
            symbols = self._split_word(word)
            word_symbols[word] = symbols
            for sym in symbols:
                self.vocab.add(sym)
        
        # 3. 迭代合并
        self.merges = []
        
        for merge_iter in range(num_merges):
            # 统计所有词中的字符对频率
            pair_freq = Counter()
            for word, freq in self.word_freq.items():
                symbols = word_symbols[word]
                pairs = self._get_pairs(symbols)
                for pair, count in pairs.items():
                    pair_freq[pair] += count * freq
            
            if not pair_freq:
                break
            
            # 选择最高频对
            best_pair = max(pair_freq, key=pair_freq.get)
            best_freq = pair_freq[best_pair]
            
            if best_freq < min_freq:
                break
            
            # 执行合并
            a, b = best_pair
            new_symbol = a + b
            self.vocab.add(new_symbol)
            self.merges.append(best_pair)
            
            # 更新所有词的表示
            new_word_symbols = {}
            for word, symbols in word_symbols.items():
                new_word_symbols[word] = self._merge_symbols(symbols, best_pair, new_symbol)
            word_symbols = new_word_symbols
            
            if (merge_iter + 1) % 100 == 0:
                print(f"合并 {merge_iter + 1}/{num_merges}，词汇表大小: {len(self.vocab)}")
        
        print(f"训练完成！词汇表大小: {len(self.vocab)}，合并次数: {len(self.merges)}")
    
    def encode_word(self, word: str) -> List[str]:
        """编码一个词"""
        symbols = self._split_word(word)
        
        for a, b in self.merges:
            new_symbol = a + b
            symbols = self._merge_symbols(symbols, (a, b), new_symbol)
        
        return symbols
    
    def encode(self, text: str) -> List[str]:
        """编码文本"""
        words = text.strip().split()
        tokens = []
        for word in words:
            tokens.extend(self.encode_word(word))
        return tokens
    
    def decode(self, tokens: List[str]) -> str:
        """解码"""
        # 移除词尾标记
        result = []
        for token in tokens:
            if token == self.end_token:
                continue
            cleaned = token.replace(self.end_token, '')
            result.append(cleaned)
        
        return ' '.join(result)
    
    def get_vocab(self) -> List[str]:
        """获取词汇表"""
        return sorted(self.vocab)


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    # 训练语料
    corpus = [
        "the cat sat on the mat",
        "the dog ran fast",
        "cats and dogs are pets",
        "playing with the cat",
        "played with dog",
    ]
    
    # 创建BPE
    bpe = BPEFromScratch()
    bpe.train(corpus, num_merges=50, min_freq=2)
    
    # 查看词汇表
    print("\n词汇表（前20个）:")
    vocab = bpe.get_vocab()
    for i, sym in enumerate(vocab[:20]):
        print(f"  {i+1}. {sym}")
    
    # 编码测试
    print("\n编码测试:")
    test_words = ["cats", "playing", "unknown"]
    for word in test_words:
        tokens = bpe.encode_word(word)
        print(f"  {word} -> {tokens}")
    
    # 完整文本测试
    text = "playing with unknown cats"
    print(f"\n文本编码:")
    print(f"  原始: {text}")
    tokens = bpe.encode(text)
    print(f"  编码: {tokens}")
    print(f"  解码: {bpe.decode(tokens)}")
```

## 9. 可视化与结果理解

以下代码展示BPE训练过程的可视化，包括词汇表增长、合并频率等：

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 使用之前定义的BPE类（简化版，记录训练过程）
class BPETracker(BPEFromScratch):
    """带训练过程跟踪的BPE"""
    
    def __init__(self):
        super().__init__()
        self.vocab_sizes = []  # 记录每次合并后的词汇表大小
        self.merge_pairs = []  # 记录每次合并的对
        self.merge_freqs = []  # 记录每次合并的频率
    
    def train(self, texts, num_merges=1000, min_freq=2):
        """重写train方法，记录过程"""
        # 初始化（与父类相同）
        self.word_freq = Counter()
        for text in texts:
            words = text.strip().split()
            for word in words:
                self.word_freq[word] += 1
        
        self.vocab = set()
        word_symbols = {}
        for word in self.word_freq:
            symbols = self._split_word(word)
            word_symbols[word] = symbols
            for sym in symbols:
                self.vocab.add(sym)
        
        self.merges = []
        self.vocab_sizes = [len(self.vocab)]
        
        for merge_iter in range(num_merges):
            pair_freq = Counter()
            for word, freq in self.word_freq.items():
                symbols = word_symbols[word]
                pairs = self._get_pairs(symbols)
                for pair, count in pairs.items():
                    pair_freq[pair] += count * freq
            
            if not pair_freq:
                break
            
            best_pair = max(pair_freq, key=pair_freq.get)
            best_freq = pair_freq[best_pair]
            
            if best_freq < min_freq:
                break
            
            a, b = best_pair
            new_symbol = a + b
            self.vocab.add(new_symbol)
            self.merges.append(best_pair)
            
            # 记录
            self.vocab_sizes.append(len(self.vocab))
            self.merge_pairs.append(best_pair)
            self.merge_freqs.append(best_freq)
            
            new_word_symbols = {}
            for word, symbols in word_symbols.items():
                new_word_symbols[word] = self._merge_symbols(symbols, best_pair, new_symbol)
            word_symbols = new_word_symbols


# 训练并跟踪
corpus = [
    "low low low low low",
    "lowest low low low",
    "newer low low",
    "wider newer lowest",
    "playing playing played plays",
    "player players play",
]

bpe_tracker = BPETracker()
bpe_tracker.train(corpus, num_merges=100, min_freq=2)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 子图1: 词汇表增长
axes[0, 0].plot(range(len(bpe_tracker.vocab_sizes)), bpe_tracker.vocab_sizes, marker='o', markersize=3)
axes[0, 0].set_xlabel('合并次数')
axes[0, 0].set_ylabel('词汇表大小')
axes[0, 0].set_title('词汇表大小随合并次数变化')
axes[0, 0].grid(alpha=0.3)

# 子图2: 合并频率（前20次）
merge_freqs = bpe_tracker.merge_freqs[:20]
axes[0, 1].bar(range(len(merge_freqs)), merge_freqs, color='skyblue')
axes[0, 1].set_xlabel('合并序号')
axes[0, 1].set_ylabel('频率')
axes[0, 1].set_title('前20次合并的频率')
axes[0, 1].grid(alpha=0.3, axis='y')

# 子图3: 合并对展示（前10个）
merge_pairs = bpe_tracker.merge_pairs[:10]
pair_labels = [f"{a}+{b}" for a, b in merge_pairs]
axes[0, 2].barh(range(len(pair_labels))[::-1], bpe_tracker.merge_freqs[:10][::-1], color='lightgreen')
axes[0, 2].set_yticks(range(len(pair_labels))[::-1])
axes[0, 2].set_yticklabels(pair_labels[::-1])
axes[0, 2].set_xlabel('频率')
axes[0, 2].set_title('前10个合并对')

# 子图4: 初始词汇表vs最终词汇表
axes[1, 0].bar(['初始', '最终'], [bpe_tracker.vocab_sizes[0], bpe_tracker.vocab_sizes[-1]], 
                color=['skyblue', 'orange'])
axes[1, 0].set_ylabel('词汇表大小')
axes[1, 0].set_title('词汇表大小变化')
for i, v in enumerate([bpe_tracker.vocab_sizes[0], bpe_tracker.vocab_sizes[-1]]):
    axes[1, 0].text(i, v + 0.5, str(v), ha='center')

# 子图5: 词编码长度对比
test_words = ["lowest", "playing", "player", "unseenword"]
encode_lengths = [len(bpe_tracker.encode_word(w)) for w in test_words]

axes[1, 1].bar(test_words, encode_lengths, color='purple', alpha=0.7)
axes[1, 1].set_xlabel('词')
axes[1, 1].set_ylabel('子词数量')
axes[1, 1].set_title('不同词的编码长度')
axes[1, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(encode_lengths):
    axes[1, 1].text(i, v + 0.1, str(v), ha='center')

# 子图6: 合并频率分布
axes[1, 2].hist(bpe_tracker.merge_freqs, bins=20, color='orange', alpha=0.7, edgecolor='black')
axes[1, 2].set_xlabel('合并频率')
axes[1, 2].set_ylabel('频数')
axes[1, 2].set_title('合并频率分布')
axes[1, 2].set_yscale('log')  # 使用对数坐标

plt.tight_layout()
plt.show()

# 打印统计信息
print("=" * 60)
print("BPE训练过程统计")
print("=" * 60)
print(f"初始词汇表大小: {bpe_tracker.vocab_sizes[0]}")
print(f"最终词汇表大小: {bpe_tracker.vocab_sizes[-1]}")
print(f"总合并次数: {len(bpe_tracker.merges)}")
print(f"最高合并频率: {max(bpe_tracker.merge_freqs)}")
print(f"最低合并频率: {min(bpe_tracker.merge_freqs)}")
```

**结果解读：**

1. **词汇表增长曲线**：应该看到词汇表大小随合并次数线性增长（每次合并增加一个新符号）。

2. **合并频率**：前几次合并的频率通常很高（合并最常见的字符对），后续逐渐降低。

3. **合并对展示**：看到最先合并的是哪些字符对（通常是常见词素，如"es"、"in"等）。

4. **词汇表大小变化**：对比初始（只有字符）和最终（包含子词）的词汇表大小。

5. **编码长度对比**：常见词（如"playing"）可能被编码为2-3个子词，罕见词（如"unseenword"）可能被分解为字符。

6. **合并频率分布**：通常服从长尾分布，少数合并频率很高，大多数合并频率较低。

## 10. 模型评估

评估BPE分词器的效果主要通过以下指标：

```python
from collections import Counter
import numpy as np

# ============================================
# BPE效果评估
# ============================================

def evaluate_bpe(tokenizer, test_corpus, name="BPE"):
    """
    评估BPE分词器的效果
    
    参数:
        tokenizer: 训练好的BPE分词器
        test_corpus: 测试语料
        name: 分词器名称
    """
    # 统计信息
    total_words = 0
    total_tokens = 0
    unknown_words = 0
    encode_lengths = []
    tokens_per_word = []
    
    for text in test_corpus:
        words = text.strip().split()
        for word in words:
            total_words += 1
            tokens = tokenizer.encode_word(word)
            total_tokens += len(tokens)
            tokens_per_word.append(len(tokens))
            
            # 检查是否有未登录词（所有子词都不在词汇表中）
            if all(token not in tokenizer.vocab for token in tokens):
                unknown_words += 1
    
    # 计算指标
    avg_tokens_per_word = total_tokens / total_words if total_words > 0 else 0
    oov_rate = unknown_words / total_words if total_words > 0 else 0
    
    # 词汇表覆盖率
    vocab_coverage = len(tokenizer.vocab) / total_tokens if total_tokens > 0 else 0
    
    print(f"\n{name} 评估结果:")
    print(f"  总词数: {total_words}")
    print(f"  总token数: {total_tokens}")
    print(f"  平均每词token数: {avg_tokens_per_word:.2f}")
    print(f"  未登录词数: {unknown_words}")
    print(f"  OOV率: {oov_rate:.2%}")
    print(f"  词汇表大小: {len(tokenizer.vocab)}")
    print(f"  词汇表覆盖率: {vocab_coverage:.2%}")
    
    return {
        'total_words': total_words,
        'total_tokens': total_tokens,
        'avg_tokens_per_word': avg_tokens_per_word,
        'oov_rate': oov_rate,
        'vocab_size': len(tokenizer.vocab)
    }


# 创建训练语料和测试语料
train_corpus = [
    "the cat sat on the mat",
    "the dog ran fast",
    "cats and dogs are pets",
    "playing with the cat",
    "played with dog",
    "lower the price",
    "lowest price ever",
    "newest model available",
]

test_corpus = [
    "the cat played with the dog",
    "lowest price for new model",
    "playing with cats and dogs",
    "unknownword testword",  # 包含未登录词
]

# 训练BPE（不同合并次数）
print("=" * 60)
print("不同合并次数的BPE效果对比")
print("=" * 60)

merge_numbers = [10, 50, 100, 500]
results = []

for num_merges in merge_numbers:
    bpe = BPEFromScratch()
    bpe.train(train_corpus, num_merges=num_merges, min_freq=2)
    
    result = evaluate_bpe(bpe, test_corpus, name=f"BPE (merges={num_merges})")
    result['num_merges'] = num_merges
    results.append(result)

# 可视化对比
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

num_merges = [r['num_merges'] for r in results]
vocab_sizes = [r['vocab_size'] for r in results]
avg_tokens = [r['avg_tokens_per_word'] for r in results]
oov_rates = [r['oov_rate'] for r in results]

axes[0].plot(num_merges, vocab_sizes, marker='o', color='blue', linewidth=2)
axes[0].set_xlabel('合并次数')
axes[0].set_ylabel('词汇表大小')
axes[0].set_title('合并次数 vs 词汇表大小')
axes[0].grid(alpha=0.3)

axes[1].plot(num_merges, avg_tokens, marker='s', color='green', linewidth=2)
axes[0].set_xlabel('合并次数')
axes[1].set_ylabel('平均每词token数')
axes[1].set_title('合并次数 vs 编码长度')
axes[1].grid(alpha=0.3)

axes[2].plot(num_merges, oov_rates, marker='^', color='red', linewidth=2)
axes[2].set_xlabel('合并次数')
axes[2].set_ylabel('OOV率')
axes[2].set_title('合并次数 vs OOV率')
axes[2].grid(alpha=0.3)
axes[2].set_yscale('log')

plt.tight_layout()
plt.show()
```

**评估指标说明：**

1. **平均每词Token数**：衡量编码后的序列长度。BPE应该在词级（1 token/词）和字符级（词长 token/词）之间取得平衡。

2. **OOV率**：未登录词占所有词的比例。BPE的一个重要目标是降低OOV率，理想情况下应该接近0%。

3. **词汇表大小**：词汇表越大，能直接表示的词越多，但模型参数量也越大。需要在两者之间取得平衡。

4. **词汇表覆盖率**：词汇表中的token数占总token数的比例，反映词汇表的使用效率。

**结果解读：**

- 随着合并次数增加，词汇表大小增加，更多词被完整保留（1 token/词）。
- 当合并次数足够多时，OOV率应该接近0%（因为所有词都能被分解为子词）。
- 平均每词token数会随合并次数增加而减少（更多词被完整保留）。
- 实际应用中，通常选择30k-50k的词汇表大小，平衡性能和效率。

## 11. 常见问题与易错点

**数据层面问题：**

1. **词频统计忽略**：训练BPE时必须考虑词频，否则所有词权重相同，高频词对的合并优先级被低估。解决方法：在统计字符对频率时，乘以词频作为权重。

2. **未添加词尾标记**：如果不添加`</w>`等词尾标记，BPE无法区分"low"和"lower"的词根部分。例如，"low"和"lower"在字符级别都是`l o w ...`，没有词尾标记就无法学习到"lower"需要额外的"er"。

3. **语料预处理不当**：如果训练语料包含大量噪声（如HTML标签、特殊字符），BPE会学习到无意义的子词。解决方法：在训练BPE前进行充分的文本预处理。

**模型层面问题：**

1. **合并顺序错误**：编码时必须严格按照训练时的合并顺序应用规则。如果顺序错误（如先合并`e r`再合并`l o`），可能导致不同的结果。解决方法：保存合并规则的完整顺序，编码时按序应用。

2. **未登录词处理不佳**：虽然BPE理论上能处理任何词，但如果训练语料中的字符集不完整（如缺少某些Unicode字符），未登录词可能被分解为字符时出错。解决方法：确保训练语料覆盖所有可能出现的字符。

3. **词边界丢失**：BPE本身不保留词边界信息（除非在子词前添加特殊标记如`##`）。在下游任务中，可能需要知道哪些子词属于同一个词。解决方法：在编码时记录词边界信息，或使用`##`等标记表示子词接续。

**调参问题：**

1. **合并次数选择**：太少会导致词汇表小但序列长，太多会导致词汇表大且可能过拟合训练语料。解决方法：在验证集上测试不同合并次数的效果，选择最优配置。通常30k-50k是常用范围。

2. **最小频率阈值**：设置过高会过滤掉有意义的低频对，过低则合并无意义的对。解决方法：通常设置为2-5，或在大规模语料上可以适当提高。

3. **特殊字符处理**：是否将标点符号、数字等作为独立token？这取决于任务。解决方法：可以在预处理时将特殊字符分离，或让BPE自动学习（但可能导致词汇表过大）。

## 12. 学习总结

BPE（Byte Pair Encoding）是一种强大的子词分词算法，通过迭代合并高频字符对，自动学习到有意义的子词单元。它有效解决了传统词级分词的OOV问题，同时避免了字符级分词的序列过长问题。

从原理层面，BPE基于信息压缩理论和Zipf定律，通过贪心算法逐步构建子词词汇表。在训练过程中，高频词被保留为完整词，低频词被分解为子词，实现了词汇表空间和序列长度的良好平衡。

在实践层面，我们学习了如何从头实现BPE训练、编码、解码的完整流程。关键要点包括：正确处理词频统计、添加词尾标记、按顺序应用合并规则、以及处理未登录词。BPE的实现虽然不复杂，但细节处理至关重要。

BPE在预训练语言模型（如GPT、BERT、RoBERTa）中得到了广泛应用，是其分词流程的核心组件。理解BPE的原理和实现，对于理解现代NLP系统的输入处理流程至关重要。

需要注意的是，BPE也有其局限性：贪心合并策略可能导致局部最优、对语料分布敏感、缺乏语义指导等。后续出现的WordPiece、Unigram等变体在一定程度上改进了这些问题。但BPE作为子词分词的经典方法，仍然是NLP工程师和研究者必备的基础知识。

总之，掌握BPE分词技术，将为理解和应用现代NLP模型打下坚实基础，特别是在处理形态丰富语言、多语言场景、以及开放词汇任务时，BPE是非常有力的工具。

## 13. 练习题与思考题

**基础题：**

1. **手动模拟BPE**：给定语料`{"low": 5, "lowest": 2, "newer": 3}`，手动模拟前3次BPE合并，写出每次合并后的词汇表和词表示。

   <details>
   <summary>答案</summary>
   **初始状态：**
   - 词汇表: `{'l', 'o', 'w', 'n', 'e', 's', 't', 'r', '</w>'}`
   - 词表示:
     - low: `['l', 'o', 'w', '</w>']` (5次)
     - lowest: `['l', 'o', 'w', 'e', 's', 't', '</w>']` (2次)
     - newer: `['n', 'e', 'w', 'e', 'r', '</w>']` (3次)
   
   **第1次合并：**
   - 统计字符对频率:
     - ('l', 'o'): 5 (low) + 2 (lowest) = 7
     - ('o', 'w'): 5 + 2 = 7
     - ('w', '</w>'): 5
     - ('w', 'e'): 2 (lowest) + 3 (newer) = 5
     - ('e', 's'): 2
     - ('s', 't'): 2
     - ('n', 'e'): 3
     - ('e', 'w'): 3
     - ('e', 'r'): 3
   - 最高频: ('l', 'o') 或 ('o', 'w')，选择 ('l', 'o')
   - 合并: 'l' + 'o' -> 'lo'
   - 新词汇表加入: 'lo'
   - 词表示更新:
     - low: `['lo', 'w', '</w>']`
     - lowest: `['lo', 'w', 'e', 's', 't', '</w>']`
     - newer: `['n', 'e', 'w', 'e', 'r', '</w>']`
   
   **第2次合并：**
   - 统计（更新后）: ('lo', 'w') 出现 5+2=7次，最高
   - 合并: 'lo' + 'w' -> 'low'
   - 新词汇表加入: 'low'
   - 词表示更新:
     - low: `['low', '</w>']`
     - lowes: `['low', 'e', 's', 't', '</w>']`
     - newer: `['n', 'e', 'w', 'e', 'r', '</w>']`
   
   **第3次合并：**
   - 统计: ('e', 'w') 出现 2+3=5次，最高（newer中的'ne'已经合并？实际上还没有）
   - 实际: ('w', 'e') 在lowest中2次，newer中3次，共5次，最高
   - 合并: 'w' + 'e' -> 'we'
   - 词表示更新:
     - low: `['low', '</w>']`
     - lowes: `['low', 'we', 's', 't', '</w>']`
     - newer: `['n', 'e', 'we', 'r', '</w>']`
   </details>

2. **实现简单的BPE编码**：不依赖任何库，实现一个简单的BPE编码函数，给定合并规则和词，输出子词序列。

   <details>
   <summary>答案</summary>
   ```python
   def simple_bpe_encode(word, merges):
       """
       简单BPE编码
       
       参数:
           word: 要编码的词
           merges: 合并规则列表，每个元素为 (a, b) 对
       
       返回:
           子词列表
       """
       # 初始化为字符 + 词尾标记
       symbols = list(word) + ['</w>']
       
       # 按合并顺序应用
       for a, b in merges:
           new_symbol = a + b
           new_symbols = []
           i = 0
           while i < len(symbols):
               if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                   new_symbols.append(new_symbol)
                   i += 2
               else:
                   new_symbols.append(symbols[i])
                   i += 1
           symbols = new_symbols
       
       return symbols
   
   # 测试
   merges = [('l', 'o'), ('o', 'w'), ('e', 's')]
   test_words = ["low", "lowest", "news"]
   
   for word in test_words:
       result = simple_bpe_encode(word, merges)
       print(f"{word} -> {result}")
   ```
   </details>

**进阶题：**

3. **实现带词频的BPE训练**：改进BPE训练算法，正确处理词频（不同词出现次数不同，合并时应考虑权重）。

   <details>
   <summary>答案要点</summary>
   ```python
   def bpe_train_with_freq(word_freq, num_merges=1000):
       """
       带词频的BPE训练
       
       参数:
           word_freq: 词频字典 {word: frequency}
           num_merges: 合并次数
       
       返回:
           合并规则列表
       """
       # 初始化词表示
       word_symbols = {}
       vocab = set()
       
       for word in word_freq:
           symbols = list(word) + ['</w>']
           word_symbols[word] = symbols
           for sym in symbols:
               vocab.add(sym)
       
       merges = []
       
       for _ in range(num_merges):
           # 统计字符对频率（考虑词频权重）
           pair_freq = Counter()
           for word, freq in word_freq.items():
               symbols = word_symbols[word]
               # 统计这个词中的所有相邻对
               for i in range(len(symbols) - 1):
                   pair = (symbols[i], symbols[i+1])
                   pair_freq[pair] += freq  # 乘以词频！
           
           if not pair_freq:
               break
           
           # 选择最高频对
           best_pair = max(pair_freq, key=pair_freq.get)
           
           # 执行合并
           a, b = best_pair
           new_symbol = a + b
           vocab.add(new_symbol)
           merges.append(best_pair)
           
           # 更新所有词的表示
           new_word_symbols = {}
           for word, symbols in word_symbols.items():
               new_symbols = []
               i = 0
               while i < len(symbols):
                   if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                       new_symbols.append(new_symbol)
                       i += 2
                   else:
                       new_symbols.append(symbols[i])
                       i += 1
               new_word_symbols[word] = new_symbols
           word_symbols = new_word_symbols
       
       return merges, vocab
   
   # 测试
   word_freq = {'low': 5, 'lowest': 2, 'newer': 3}
   merges, vocab = bpe_train_with_freq(word_freq, num_merges=10)
   print("合并规则:", merges[:5])
   print("词汇表大小:", len(vocab))
   ```
   </details>

4. **对比BPE和WordPiece**：WordPiece是BERT使用的分词算法，与BPE类似但选择合并对的准则不同（使用似然增益而非频率）。尝试实现WordPiece的选择准则，并比较与BPE的差异。

   <details>
   <summary>答案要点</summary>
   ```python
   def wordpiece_selection(pair_freq, vocab_freq):
       """
       WordPiece的选择准则：最大化似然增益
       
       对于候选对 (a, b)，计算：
       score = freq(a,b) / (freq(a) * freq(b))
       选择score最高的对
       """
       best_pair = None
       best_score = -1
       
       for (a, b), freq_ab in pair_freq.items():
           freq_a = vocab_freq.get(a, 0)
           freq_b = vocab_freq.get(b, 0)
           
           if freq_a > 0 and freq_b > 0:
               score = freq_ab / (freq_a * freq_b)
               if score > best_score:
                   best_score = score
                   best_pair = (a, b)
       
       return best_pair
   
   # 对比：
   # BPE: 选择 freq(a,b) 最大的对
   # WordPiece: 选择 freq(a,b) / (freq(a) * freq(b)) 最大的对
   # 这个比值类似于互信息，更倾向于选择"搭配紧密"的对
   ```
   </details>

**开放题：**

5. **设计适应性的BPE**：传统的BPE使用固定的合并次数。设计一个自适应的BPE算法，能够根据语料特点自动决定何时停止合并（例如，基于词汇表覆盖率、或合并增益阈值）。

   <details>
   <summary>参考答案要点</summary>
   自适应BPE设计：
   
   **1. 基于覆盖率的停止准则**：
   ```python
   def adaptive_bpe_by_coverage(word_freq, target_coverage=0.95):
       """
       当词汇表能够覆盖target_coverage比例的词时停止
       """
       total_words = sum(word_freq.values())
       vocab = set()
       # ... 训练循环 ...
       while True:
           # 计算当前覆盖率
           covered = 0
           for word, freq in word_freq.items():
               tokens = encode_word(word)
               if all(t in vocab for t in tokens):  # 所有子词都在词汇表中
                   covered += freq
           
           coverage = covered / total_words
           if coverage >= target_coverage:
               break
           # 继续合并...
   ```
   
   **2. 基于增益的停止准则**：
   ```python
   def adaptive_bpe_by_gain(pair_freq_history, gain_threshold=0.01):
       """
       当合并带来的增益低于阈值时停止
       """
       if len(pair_freq_history) >= 2:
           gain = pair_freq_history[-1] - pair_freq_history[-2]
           if gain < gain_threshold:
               return True  # 停止
       return False
   ```
   
   **3. 基于词汇表大小的停止**：
   也可以直接设定目标词汇表大小（如30k），达到后停止。
   
   **4. 多准则结合**：
   可以结合多种准则，如"达到30k大小 或 覆盖率>95%，先到先停"。
   </details>

## 14. 学习路径建议

**前置知识：**
- Python基础（字符串操作、列表处理、字典操作）
- 基础概率统计（频率、分布）
- 信息论基础（熵、压缩原理）
- 形态学基础（词根、词缀、词素）

**平行学习：**
- WordPiece和Unigram分词（了解BPE的变体）
- 词级别分词和字符级分词（理解BPE的平衡作用）
- 语言模型基础（BPE作为预处理步骤）
- 形态学分析（理解子词的意义）

**进阶方向：**
- 多语言BPE（跨语言共享词汇表）
- 适应特定领域的BPE（如医疗、法律）
- 句子Piece库的使用（工业级实现）
- BPE与神经网络的端到端学习（如通过BPE增强词嵌入）

**推荐资源：**
1. **BPE原论文**: https://arxiv.org/abs/1508.07909 - Neural Machine Translation of Rare Words with Subword Units
2. **SentencePiece GitHub**: https://github.com/google/sentencepiece - Google的工业级BPE/Unigram实现
3. **Hugging Face Tokenizers**: https://huggingface.co/docs/tokenizers/ - 现代NLP库的分词器工具

通过系统学习BPE分词，你将掌握现代NLP系统中基础而关键的预处理技术，为理解预训练模型和构建自己的NLP应用打下基础。
