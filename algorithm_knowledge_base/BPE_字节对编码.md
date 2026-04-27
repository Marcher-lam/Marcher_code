# BPE 字节对编码 学习文档

> 将文本智能地切分为子词单元，使LLM能高效处理任意输入文本。

> 来源线索：本节内容根据原书第2章2.4节关于tokenizer/BPE的讲解整理、扩展与教学化改写。

## 1. 算法基础认知

### 一句话定义
BPE (Byte Pair Encoding) 是一种从数据中自动学习子词切分规则的分词算法。

### 直觉类比
把英文单词"unfortunately"拆成"un"+"fortun"+"ate"+"ly"——就像把长单词拆成几个有意义的小块。BPE就是自动找出这种"最优的拆法"。

### 历史背景
BPE最初是1994年提出的数据压缩算法。2016年Sennrich等人将其应用于机器翻译中的子词切分。GPT-2的发布使BPE在LLM领域广为人知，此后几乎所有现代LLM（GPT系列、Llama系列、Qwen系列等）都使用BPE或其变体。

### 算法定位
- **类型**：分词算法 / 数据预处理 / 子词切分
- **性质**：无监督学习（从数据中学习合并规则），在训练LLM之前独立训练

### 前置知识
- 理解token概念
- 基本的字符串操作
- 了解Unicode编码基础

## 2. 核心原理

### 核心思想
BPE的核心思想是**迭代合并**：从字符级开始，反复找出训练文本中最常出现的相邻符号对，将它们合并成一个新符号。最终得到一个从字符到常见子词再到完整单词的"词汇阶梯"。

### 工作流程
1. 将训练语料中每个单词拆分为字符序列，每个字符后加词尾标记
2. 统计所有相邻符号对的出现频率
3. 找出出现最频繁的一对（如"e"+"r"），将其合并为新符号"er"
4. 更新语料（所有"e r"序列变为"er"）
5. 重复步骤2-4，直到达到目标词汇量或合并次数
6. 得到最终的合并规则表 + 词汇表

### 关键概念解释
- **合并规则 (Merge Rules)**：按优先级排序的"哪些符号应该合并"的列表。编码时从头到尾应用这些规则。
- **词汇表 (Vocabulary)**：所有原子符号和合并后符号的集合。LLM只会生成词汇表内的token。
- **子词 (Subword)**：大于字符、小于完整单词的文本单元，如"ing""tion""pre"
- **词尾标记**：标记单词结束的特殊符号（通常用`</w>`），确保BPE区分"est"(单词中)和"est</w>"(单词结尾)

### 直观解释
```
文本: "lower lower lower"

初始: l o w e r </w> (×3)

第1步统计: 
  (l,o)=3, (o,w)=3, (w,e)=3, (e,r)=3, (r,</w>)=3
  最高频率都=3，按规则选第一个: 合并 lo

第2步:
  lo w e r </w>  (×3)
  统计: (w,e)=3, (e,r)=3 → 合并 er

第3步:
  lo w er </w>  (×3)
  合并 lo w → low
  → low er </w>

最终词汇: l, o, w, e, r, lo, er, low
```

## 3. 数学公式与推导

### 符号约定
| 符号 | 含义 |
|------|------|
| $\mathcal{D}$ | 训练语料 |
| $V$ | 当前词汇表 |
| $|V|$ | 词汇表大小 |
| $(a, b)$ | 相邻符号对 |
| $f(a, b)$ | 符号对 $(a,b)$ 在语料中的出现次数 |

### 核心算法

每一步中，选择合并的符号对为：

$$(a^*, b^*) = \arg\max_{a,b \in V} f(a, b)$$

合并操作 $a^* b^* \to ab^*$ 后：

- 新符号 $ab^*$ 加入词汇表 $V' = V \cup \{ab^*\}$
- 更新所有出现次数的统计

重复至 $|V| = \text{target\_vocab\_size} + |\text{base\_characters}|$。

### 编码过程（应用合并规则）

给定输入文本和学到的合并规则，编码过程按照合并规则的先后依次应用：

```python
# 伪代码: 应用BPE合并规则
def bpe_encode(word, merge_rules):
    symbols = list(word)
    for (a, b) in merge_rules:  # 按学习顺序
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i+1] == b:
                symbols[i] = a + b  # 合并
                symbols.pop(i+1)
            i += 1
    return symbols
```

### 与分词质量相关的度量

**压缩率**: token数量越少越好（每个token携带更多信息）

$$\text{压缩率} = \frac{\text{字符数}}{\text{token数}}$$

## 4. 训练过程讲解

### 数据预处理
- 文本规范化（Unicode标准化，NFKC/NFKD）
- 按空格预分词（对于英文等空格分隔的语言）
- 字节级BPE变体直接操作字节流，无需预分词

### 训练过程
1. 从256个基础字节（或字符集）开始初始化词汇表
2. 在所有训练文本上统计字符对频率——这是最耗时的步骤
3. 选最高频对合并，写入合并规则列表
4. 更新符号序列和频率统计
5. 重复直到达到目标词汇量：常见值32k-150k

### 关键超参数
| 参数 | 作用 | 推荐范围 | 默认建议 |
|------|------|----------|----------|
| vocab_size | 最终词汇量 | 32k ~ 150k | 50k-151k |
| min_frequency | 符号对合并的最低频率 | 1 ~ 10 | 2 |
| max_token_length | 单个token最大长度 | 无限制 | 无限制 |

## 5. 应用场景

### 典型应用
1. **LLM分词**：几乎所有现代LLM都使用BPE或类似子词分词。Qwen3 用151k词汇量，Llama 3 用128k。
2. **多语言处理**：BPE不需要语言特定的规则，同一种分词器可以处理多种语言混合输入。
3. **OOV处理**：通过子词组合，没有真正的"未登录词"——每个未知词都能拆解为子词。
4. **代码分词**：代码中的变量名、函数名等会被BPE合理拆分。

### 适用数据特征
- 大规模多语言文本
- 包含大量罕见词或新词的领域
- 需要固定大小词汇表的场景

### 不适用场景
- 中文等无空格语言：BPE对中文不太理想（常需配合预分词）
- 需要精确字符级处理的场景

## 6. 优缺点分析

### 优点
| 优点 | 说明 | 成立条件 |
|------|------|----------|
| 自动学习 | 无需人工设计规则 | 有足够训练数据 |
| 平衡粒度 | 常见词完整，罕见词拆子词 | — |
| 无OOV | 所有词都可表示为子词序列 | 初始字符覆盖输入语言 |
| 多语言支持 | 同一模型处理多种语言 | 训练数据多语言 |
| 可逆性 | 无损编解码 | — |

### 缺点
| 缺点 | 说明 | 缓解思路 |
|------|------|----------|
| 训练数据偏差 | 训练语料中不常见的语言分词质量差 | 多语言均衡数据 |
| 中文表现不佳 | 无空格分隔，BPE合并不自然 | 使用Unigram/sentencepiece |
| 词汇量大 | 为覆盖多语言需大词汇量，增加模型参数 | 字节级BPE+静态词汇压缩 |

### 与同类方法对比
| 方法 | 粒度 | 多语言 | OOV处理 | 训练方式 |
|------|------|--------|---------|----------|
| BPE | 子词 | 好 | 字符级回退 | 统计合并 |
| WordPiece | 子词 | 好 | 字符级回退 | 基于似然 |
| Unigram | 子词 | 很好 | 字符级回退 | 概率模型 |
| SentencePiece | 子词 | 很好 | — | BPE/Unigram |

## 7. 调库实现

```python
"""
BPE 分词的调库实现
使用 HuggingFace tokenizers 库
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import os


def train_bpe_tokenizer(
    training_files: list,
    vocab_size: int = 50000,
    save_path: str = "./bpe_tokenizer.json",
):
    """
    训练一个BPE分词器
    """
    # 初始化BPE模型
    tokenizer = Tokenizer(models.BPE())

    # 预分词器：按空格和标点分割（对英文重要）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 解码器：对应byte-level BPE
    tokenizer.decoder = decoders.ByteLevel()

    # BPE训练器配置
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
        min_frequency=2,
        show_progress=True,
    )

    # 在训练文件上训练
    tokenizer.train(files=training_files, trainer=trainer)

    # 保存分词器
    tokenizer.save(save_path)
    print(f"BPE分词器已保存到 {save_path}")
    print(f"词汇大小: {tokenizer.get_vocab_size()}")

    return tokenizer


def use_bpe_tokenizer(tokenizer_path: str, text: str):
    """使用训练好的BPE分词器"""
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 编码：文本→token IDs
    encoded = tokenizer.encode(text)
    print(f"原始文本: {text}")
    print(f"Token IDs: {encoded.ids}")
    print(f"Token数量: {len(encoded.ids)}")

    # 解码：token IDs→文本
    decoded = tokenizer.decode(encoded.ids)
    print(f"解码: {decoded}")

    # 查看每个token对应的文本
    print("\nToken分解:")
    for i, token_id in enumerate(encoded.ids):
        token_text = tokenizer.decode([token_id])
        print(f"  [{i}] ID {token_id} → '{token_text}'")

    return encoded


# ===== 使用示例 =====
# 创建示例训练数据文件
sample_text = (
    "Machine learning is a subset of artificial intelligence. "
    "Deep learning uses neural networks with many layers. "
    "Natural language processing helps computers understand human language.\n"
) * 100  # 重复以产生足够数据

with open("sample_corpus.txt", "w") as f:
    f.write(sample_text)

# 训练BPE分词器
print("训练BPE分词器...")
# tokenizer = train_bpe_tokenizer(["sample_corpus.txt"], vocab_size=500)

# 如果训练文件不存在或不想实际训练，这里是模拟输出：
print("预期输出:")
print("BPE分词器已保存到 ./bpe_tokenizer.json")
print("词汇大小: ~500 (实际会因vocab_size参数而异)")
```

## 8. 手工代码实现

```python
"""
BPE 的手工实现
从零实现BPE训练和编码过程
"""

from collections import defaultdict
from typing import List, Dict, Tuple
import re


class BPETokenizer:
    """手工实现BPE分词器"""

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], int] = {}  # 合并规则及其优先级
        self.vocab: Dict[int, str] = {}  # token ID → token text
        self.inverse_vocab: Dict[str, int] = {}  # token text → token ID

    def _get_word_frequencies(self, corpus: str) -> Dict[str, int]:
        """统计语料中每个词的出现频率"""
        # 按空格分割单词，每个单词后加 </w> 标记词尾
        words = re.findall(r'\w+|\S', corpus.lower())
        word_freqs = defaultdict(int)
        for word in words:
            word_freqs[word + '</w>'] += 1
        return dict(word_freqs)

    def _initialize_splits(self, word_freqs: Dict[str, int]) -> Dict[str, List[str]]:
        """将每个词初始化拆为字符序列"""
        splits = {}
        for word in word_freqs:
            # 每个字符成为一个独立的符号
            splits[word] = list(word)
        return splits

    def _get_pair_frequencies(
        self,
        splits: Dict[str, List[str]],
        word_freqs: Dict[str, int],
    ) -> Dict[Tuple[str, str], int]:
        """统计所有相邻符号对的频率"""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = splits[word]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] += freq
        return dict(pair_freqs)

    def _merge(self, splits: Dict[str, List[str]], pair: Tuple[str, str]):
        """在所有的字分割中，将pair合并为一个符号"""
        a, b = pair
        for word in splits:
            symbols = splits[word]
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols[i] = a + b  # 合并为一个符号
                    symbols.pop(i + 1)
                else:
                    i += 1

    def train(self, corpus: str):
        """
        训练BPE分词器

        核心算法:
        1. 将所有词初始化为字符序列
        2. 一遍遍地找出最频繁的相邻符号对
        3. 将这对符号合并成新符号
        4. 记录合并规则
        """
        # 获取词频并初始化为字符序列
        word_freqs = self._get_word_frequencies(corpus)
        splits = self._initialize_splits(word_freqs)

        # 基础字符加入词汇表
        base_chars = set()
        for word in word_freqs:
            for char in word:
                base_chars.add(char)

        # 为所有基础字符分配 token ID
        for i, char in enumerate(sorted(base_chars)):
            self.vocab[i] = char
            self.inverse_vocab[char] = i

        num_merges = self.vocab_size - len(base_chars)
        print(f"基础字符数: {len(base_chars)}")
        print(f"计划合并次数: {num_merges}")

        for step in range(num_merges):
            # 统计当前所有相邻符号对的频率
            pair_freqs = self._get_pair_frequencies(splits, word_freqs)

            if not pair_freqs:
                break  # 没有更多可合并的对

            # 找到频率最高的对
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]

            if best_freq < 2:  # 频率太低，不再有意义
                break

            # 记录合并规则（顺序很重要！）
            self.merges[best_pair] = step

            # 新token
            new_token = best_pair[0] + best_pair[1]
            token_id = len(self.vocab)
            self.vocab[token_id] = new_token
            self.inverse_vocab[new_token] = token_id

            # 应用合并
            self._merge(splits, best_pair)

            if step % 50 == 0:
                print(f"  合并 #{step}: '{best_pair[0]}'+'{best_pair[1]}'→'{new_token}' (频率:{best_freq})")

        print(f"训练完成。词汇表大小: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """
        使用学到的合并规则编码文本

        按照合并规则的优先顺序，逐对合并
        """
        # 转为小写并标记词尾
        text = text.lower()
        symbols = list(text + '</w>')

        # 按合并规则的优先级顺序进行合并
        # 排序规则：先学到的规则优先应用（step越小越优先）
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])

        for (a, b), _ in sorted_merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols[i] = a + b
                    symbols.pop(i + 1)
                else:
                    i += 1

        # 将符号转为 token ID
        token_ids = []
        for symbol in symbols:
            if symbol in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[symbol])
            else:
                # 未知符号：拆回字符
                for char in symbol:
                    token_ids.append(
                        self.inverse_vocab.get(char, 0)
                    )

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """将token ID序列解码回文本"""
        tokens = []
        for tid in token_ids:
            if tid in self.vocab:
                tokens.append(self.vocab[tid])

        text = ''.join(tokens)
        # 去掉词尾标记
        text = text.replace('</w>', ' ')
        return text.strip()


# ===== 测试手工BPE =====
corpus = (
    "lower lower lower higher higher newest widest "
    "low lower lowest low low low low "
) * 20

print("=== 训练手工BPE分词器 ===")
bpe = BPETokenizer(vocab_size=100)
bpe.train(corpus)

print("\n=== 编码测试 ===")
test_word = "lowest"
tokens = bpe.encode(test_word)
print(f"'{test_word}' → tokens: {tokens}")
print(f"解码: '{bpe.decode(tokens)}'")

# 显示每个token对应的文本
print("\nToken分解:")
for tid in tokens:
    print(f"  ID {tid} → '{bpe.vocab[tid]}'")
```

## 9. 可视化与结果理解

```python
"""
BPE分词的可视化
"""
import matplotlib.pyplot as plt
import numpy as np

# 展示不同词汇量下的分词效果
text_examples = [
    ("understanding", [9, 6, 4, 3, 2, 2]),
    ("unfortunately", [10, 8, 6, 4, 3, 2]),
    ("internationalization", [19, 14, 10, 6, 4, 3]),
    ("misunderstanding", [12, 9, 7, 5, 4, 3]),
    ("reinforcement", [10, 7, 5, 4, 3, 2]),
]

fig, ax = plt.subplots(figsize=(10, 5))
vocab_sizes = ["1k", "5k", "10k", "50k", "100k", "150k"]
x = np.arange(len(vocab_sizes))

for i, (word, token_counts) in enumerate(text_examples):
    ax.plot(x, token_counts, marker='o', linewidth=2, markersize=8, label=word)

ax.set_xticks(x)
ax.set_xticklabels(vocab_sizes)
ax.set_xlabel("词汇表大小", fontsize=12, fontweight="bold")
ax.set_ylabel("分词后的token数", fontsize=12, fontweight="bold")
ax.set_title("BPE: 词汇量越大 = token越少（压缩越高效）\n但模型参数也越多", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
print("""
解读：词汇量越大，常见词越容易被完整保留（不拆分），
token数减少。但词汇量增加意味着embedding矩阵变大，
训练和推理成本都上升。50k-150k是现代LLM的常见平衡点。
""")
```

## 10. 模型评估

BPE分词器的评估主要关注压缩效率和多语言覆盖。

```python
"""BPE分词器评估"""
import numpy as np

def evaluate_bpe_tokenizer(tokenizer, test_texts):
    stats = []
    for text in test_texts:
        tokens = tokenizer.encode(text)
        stats.append({
            "text_len_chars": len(text),
            "num_tokens": len(tokens),
            "compression_ratio": len(text) / max(1, len(tokens)),
        })

    ratios = [s["compression_ratio"] for s in stats]
    print(f"测试样本: {len(stats)}")
    print(f"平均token数: {np.mean([s['num_tokens'] for s in stats]):.1f}")
    print(f"平均压缩率: {np.mean(ratios):.2f} 字符/token")
    return stats

# 测试
test_texts = [
    "Machine learning is fascinating.",
    "自然语言处理是一个有趣的领域。",
    "AI can understand and generate text.",
]
print("BPE评估示例:")
for t in test_texts:
    print(f"  \"{t}\" → 约{len(t)//4}-{len(t)//3}个tokens")
```

## 11. 常见问题与易错点

### 数据层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 中文分词差 | "我喜欢学习"四个汉字被逐字拆开 | BPE训练语料中中文占比低 | 使用Unigram或在多语言数据上训练 |
| 特殊字符误处理 | emoji、URL被不合理拆分 | 训练数据中这些字符太少 | 添加URL/emoji的预分词规则 |

### 模型层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 词汇表太大 | 模型embedding层占太多参数 | vocab_size太大 | 降词汇量或使用嵌入压缩技术 |
| 词汇表太小 | 太多词被拆分，序列变长影响性能 | vocab_size太小 | 增大词汇量 |

### 调参层面
| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 最做BPE很慢 | 训练半天没完成 | 简单实现O(n³)复杂度 | 用优化库(HuggingFace tokenizers) |

## 12. 学习总结

BPE通过迭代合并最频繁的符号对，自动学习子词切分规则。它是现代LLM的标准分词方案，平衡了表达能力和计算效率。理解BPE有助于理解为什么同样的文本在不同模型中token数不同，以及为什么某些语言（中文）在英文主导的模型中分词效率较低。

## 13. 练习题与思考题

**题1**：为什么 BPE 要给每个单词末尾加 `</w>` 标记？

**参考答案**：区分"单词中间的字符对"和"词尾的字符对"。例如"est"在"estimate"中间和"smallest</w>"词尾是不同的token——后者只在词尾出现。不加标记的话，两者的token会混乱，解码后无法还原单词边界。

**题2**：Qwen3的词汇量是151k，GPT-2是50k。更大的词汇量有什么优劣？

**参考答案**：优势——更多常用词不变拆分，序列更短（推理更快）。劣势——embedding矩阵太大（151k×emb_dim），模型参数变多，训练和推理的开销增加。Qwen3选择151k是对多语言+长序列效率综合权衡的结果。

## 14. 学习路径建议

- **前置**: 基础Python文本处理
- **进阶**: Unigram/SentencePiece分词器原理
- **推荐资源**: Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016)
