# 自然语言处理基础

自然语言处理（NLP）是AI的重要应用领域，处理人类语言的理解和生成。

## 🎯 学习目标

### 1. 基础技术
- 文本预处理（分词、清洗、归一化）
- 词向量（Word2Vec、GloVe、FastText）
- 语言模型
- 序列标注（NER、POS Tagging）

### 2. 经典模型
- RNN、LSTM、GRU
- Seq2Seq模型
- Attention机制
- Transformer

### 3. 预训练模型
- BERT系列（BERT、RoBERTa、ALBERT）
- GPT系列（GPT-2、GPT-3、GPT-4）
- T5、BART
- 中文模型（ERNIE、ChatGLM）

## 📚 主要任务

### 文本理解
- **文本分类**：情感分析、主题分类
- **命名实体识别（NER）**：提取人名、地名等
- **关系抽取**：识别实体间关系
- **依存句法分析**：分析句子结构

### 文本生成
- **机器翻译**
- **文本摘要**
- **对话系统**
- **故事生成**

### 问答系统
- **阅读理解**
- **知识图谱问答**
- **开放域问答**

## 🛠️ 技术栈

```python
# 传统NLP
import nltk
import jieba  # 中文分词
import spacy

# 现代NLP
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline
```

## 📖 学习资源

### 书籍
- 《Speech and Language Processing》- Jurafsky & Martin
- 《自然语言处理综论》
- 《神经自然语言处理》

### 课程
- Stanford CS224n: NLP with Deep Learning
- 《自然语言处理》- 哈工大

### 预训练模型
- Hugging Face Transformers
- ModelScope（中文模型）

## 💡 实践项目

### 初级
- [ ] 文本分类（情感分析）
- [ ] 关键词提取
- [ ] 简单聊天机器人

### 中级
- [ ] 命名实体识别
- [ ] 机器翻译
- [ ] 文本摘要

### 高级
- [ ] 问答系统
- [ ] 知识图谱
- [ ] 大语言模型微调

## 🔗 核心概念

### Word Embedding
```
词 → 向量
"国王" - "男人" + "女人" ≈ "女王"
```

### Transformer
```
Self-Attention → 并行计算
Position Encoding → 位置信息
Multi-Head → 多角度关注
```

### BERT
```
预训练 + 微调
Masked LM + Next Sentence Prediction
```

## 📝 学习路径

```
1. 传统NLP方法
   ↓
2. 词向量和语言模型
   ↓
3. RNN/LSTM
   ↓
4. Attention机制
   ↓
5. Transformer和BERT
   ↓
6. 大语言模型（LLM）
   ↓
7. 实际项目
```

## 💻 编程实践

### 文本预处理流程
1. 文本清洗（去除HTML、特殊字符）
2. 分词（中英文不同）
3. 去停用词
4. 词干提取/词形还原
5. 向量化

### 现代NLP流程
1. 选择预训练模型
2. 加载Tokenizer
3. 数据预处理
4. 模型微调
5. 评估和部署
