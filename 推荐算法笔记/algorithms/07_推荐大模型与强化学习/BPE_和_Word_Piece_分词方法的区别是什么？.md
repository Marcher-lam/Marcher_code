# 面试题：BPE 和 Word Piece 分词方法的区别是什么？

# 面试题：BPE 和 Word Piece 分词方法的区别是什么？

Byte-Pair Encoding (BPE) 和 WordPiece 是现代自然语言处理中两种主流的子词分词算法，它们通过将单词拆分为更小的、有意义的子词单元，巧妙地平衡了词表大小与未登录词（OOV）问题。

<table><tr><td>特性</td><td>Byte-Pair Encoding (BPE)</td><td>WordPiece</td></tr><tr><td>核心思想</td><td>基于频率的贪婪合并</td><td>基于语言模型似然的合并</td></tr><tr><td>合并策略</td><td>迭代合并出现频率最高的相邻符号对</td><td>迭代合并能最大化训练数据似然的符号对</td></tr><tr><td>合并准则</td><td>频率驱动，选择最常出现的组合</td><td>利用互信息，选择关联性最强的组合</td></tr><tr><td>词表构建</td><td>自底向上，从字符开始合并</td><td>自底向上，从字符开始合并</td></tr><tr><td>典型应用</td><td>GPT 系列、RoBERTa</td><td>BERT 及其变体</td></tr><tr><td>子词表示</td><td>不强制使用特殊标记</td><td>常使用##前缀标记非词首子词</td></tr></table>

# BPE (Byte-Pair Encoding) 原理

BPE 的核心运作机制可以概括为 "合并频率最高的相邻符号对"。其本质是一种数据压缩算法，后被成功应用于 NLP领域。

 初始化：将训练语料中的每个单词分割成最基本的单元（例如字符或字节），并在单词末尾添加特殊的结束符（如</w>）以标记单词边界。此时，初始词表就是所有这些基本单元。  
 统计与合并：

 统计文本中所有相邻符号对（一开始是字符对）出现的频率。  
 找到出现频率最高的那一对符号（例如，连续的字符 "e"和 "s"）。  
 将语料中所有出现的这个符号对合并成一个新的、更大的符号（例如，将 "e"和 "s"合并为 "es"），并将这个新符号加入到词表中。

 迭代：不断重复"统计-合并"的过程，直到达到预设的词表大小或合并次数。

通过这个过程，像 "low"、"lower"、"newest"这样的单词，经过多轮合并后，可能会产生 "low"、"er"、"est"等有意义的子词单元。常见的单词（如 "the"）可能会被保留为完整 token，而罕见词（如 "unfamiliar"）则会被拆分成如 "un","fam", "iliar"这样的子词。

WordPiece 不再简单地选择最频繁的符号对，而是选择那个能最大程度提升语言模型在训练数据上似然概率的符号对进行合并。具体来说，它会计算每对相邻符号的互信息（点互信息 PMI）。互信息越高，表明这两个符号的关联性越强，合并后对语言模型似然值的提升就越大。

# 主要 Steps：

# 1. 初始化词汇表 (Initialization)

将训练语料中的每个单词拆分成更小的单元。最常见的做法是将单词拆分为字符（对于拉丁语系）或更小的单位（如字节），并在非词首的单元前添加一个特殊前缀（如 ##）以标识其在单词中的位置。所有这些基本单元构成了初始词汇表。

# 2. 迭代合并 (Iterative Merging)

 统计频率：基于当前词汇表和分词结果，统计语料中所有相邻符号对的出现频率。

 计算得分：对于每一个相邻符号对，使用公式 $\text{score} = \frac{\text{freq\_of\_pair}}{\text{freq\_of\_first\_element} \times \text{freq\_of\_second\_element}}$ 计算其合并得分（即互信息 PMI）。  
 选择与合并：选择得分最高的符号对进行合并。这个新的合并单元会被加入到词汇表中，并在语料中的所有出现位置用这个新单元替换原来的符号对。

# 3. 终止条件 (Termination Condition)

重复迭代合并步骤，直到词汇表大小达到预设的目标值，或者没有更多有意义的合并可以进行（例如，得分低于某个阈值）。

# 4 选型考量

综上所述，BPE 和 WordPiece 的主要区别在于合并准则：BPE 是频率驱动，而 WordPiece 是似然驱动。这使得WordPiece 在理论上更能捕捉到有意义的子词单元。在选择使用哪种方法时，可以考虑以下几点：

 任务类型：由于 BERT 类模型普遍使用 WordPiece，若需构建双向上下文理解模型或进行迁移学习，WordPiece 可能是更自然的选择。而对于自回归生成式任务（如文本生成），BPE 系列（尤其是 BBPE）应用更广。  
 语言特性：对于英语等空格分隔语言，两者都适用。但对于多语言混合或需要处理特殊符号（如代码、表情）的场景，BPE 的变种 Byte-level BPE (BBPE) 因其基于字节构建，具有更好的通用性。  
 实践建议：在大多数实际应用中，我们通常直接使用预训练模型（如 BERT、GPT）自带的分词器，而非从头开始训练。

# 5 BPE 完整实现代码

```python
import re
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}

    def _get_pairs(self, word_freqs):
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair, word_freqs):
        new_word_freqs = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word, freq in word_freqs.items():
            new_word = pattern.sub(''.join(pair), word)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, corpus):
        word_freqs = Counter()
        for text in corpus:
            words = text.strip().split()
            for word in words:
                word_freqs[' '.join(list(word)) + ' </w>'] += 1

        base_vocab = set()
        for word in word_freqs:
            for char in word.split():
                base_vocab.add(char)

        self.vocab = {char: idx for idx, char in enumerate(sorted(base_vocab))}

        for _ in range(self.vocab_size - len(base_vocab)):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)

        return self

    def tokenize(self, word):
        word = ' '.join(list(word)) + ' </w>'
        tokens = word.split()
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + [pair[0] + pair[1]] + tokens[i + 2:]
                else:
                    i += 1
        return tokens

corpus = [
    "low lower lowest",
    "new newer newest",
    "the theme is low",
]
bpe = BPETokenizer(vocab_size=50).train(corpus)
print("BPE词表:", list(bpe.vocab.keys())[:20])
print("分词 'lowest':", bpe.tokenize("lowest"))
print("分词 'newest':", bpe.tokenize("newest"))
```

# 6 WordPiece 完整实现代码

```python
class WordPieceTokenizer:
    def __init__(self, vocab_size=1000, max_word_len=100):
        self.vocab_size = vocab_size
        self.max_word_len = max_word_len
        self.vocab = {}

    def _compute_pair_scores(self, splits, word_freqs):
        letter_freqs = Counter()
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split)):
                letter_freqs[split[i]] += freq
                if i < len(split) - 1:
                    pair_freqs[(split[i], split[i + 1])] += freq

        scores = {}
        for pair, freq in pair_freqs.items():
            first, second = pair
            score = freq / (letter_freqs[first] * letter_freqs[second])
            scores[pair] = score
        return scores

    def train(self, corpus):
        word_freqs = Counter()
        for text in corpus:
            for word in text.strip().split():
                word_freqs[word] += 1

        splits = {word: [c if i == 0 else '##' + c for i, c in enumerate(word)]
                  for word in word_freqs}

        base_chars = set()
        for split in splits.values():
            for token in split:
                base_chars.add(token)
        self.vocab = {char: idx for idx, char in enumerate(sorted(base_chars))}

        for _ in range(self.vocab_size - len(base_chars)):
            scores = self._compute_pair_scores(splits, word_freqs)
            if not scores:
                break
            best_pair = max(scores, key=scores.get)
            for word in word_freqs:
                split = splits[word]
                i = 0
                while i < len(split) - 1:
                    if split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
                        merged = best_pair[0] + best_pair[1].lstrip('##') \
                            if best_pair[1].startswith('##') \
                            else best_pair[0] + best_pair[1]
                        if not merged.startswith('##') and i > 0:
                            pass
                        merged = best_pair[0] + best_pair[1][2:] \
                            if best_pair[1].startswith('##') \
                            else best_pair[0] + best_pair[1]
                        split = split[:i] + [merged] + split[i + 2:]
                    else:
                        i += 1
                splits[word] = split
            new_token = best_pair[0] + best_pair[1].lstrip('##')
            self.vocab[new_token] = len(self.vocab)

        return self

    def tokenize(self, word):
        if word in self.vocab:
            return [word]
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            found = False
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = '##' + substr
                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break
                end -= 1
            if not found:
                tokens.append('##' + word[start] if start > 0 else word[start])
                end = start + 1
            start = end
        return tokens

wp_corpus = [
    "low lower lowest",
    "new newer newest",
    "the theme is low",
]
wp = WordPieceTokenizer(vocab_size=50).train(wp_corpus)
print("WordPiece词表:", list(wp.vocab.keys())[:20])
print("分词 'lowest':", wp.tokenize("lowest"))
print("分词 'newer':", wp.tokenize("newer"))
```

# 7 SentencePiece 对比

| 特性 | BPE | WordPiece | SentencePiece |
|------|-----|-----------|---------------|
| 训练数据要求 | 预分词文本 | 预分词文本 | 原始文本（无需预分词） |
| 语言依赖 | 依赖分词器 | 依赖分词器 | 语言无关 |
| 实现方式 | 字符级合并 | PMI合并 | BPE/Unigram 均支持 |
| 多语言支持 | 需要额外处理 | 需要额外处理 | 原生支持 |
| 代表模型 | GPT-2/3/4 | BERT | T5, ALBERT, LLaMA |
| 空格处理 | 特殊标记 | 特殊标记 | 将空格转为▁符号 |

# 8 常见问题与易错点

1. **词表大小选择**：词表太小导致 OOV 多，太大则模型参数增多，中文一般 3-5 万，英文 3 万左右。
2. **子词正则化**：训练时随机使用多种分词结果可提升鲁棒性，推理时用最贪心的分词。
3. **中文分词特殊性**：BPE/WordPiece 对中文通常以字符为基本单位，不需要额外分词器。
4. **预训练与微调词表不一致**：微调时应使用预训练模型的词表，不要重新训练分词器。

# 9 学习路径建议

1. 理解子词分词的动机（OOV 问题）
2. 手动实现 BPE 和 WordPiece 算法
3. 对比不同分词方法的效果差异
4. 学习 SentencePiece 的 Unigram 模型
5. 研究大模型中的分词策略对性能的影响
