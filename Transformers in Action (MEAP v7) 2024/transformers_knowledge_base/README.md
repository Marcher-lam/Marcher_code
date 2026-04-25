# Transformers in Action 学习知识库

> 本知识库基于《Transformers in Action (MEAP v7) 2024》全书内容整理，将各章节和关键技术拆分为独立学习文档，方便按需查阅和学习。

## 目录结构

| 文件 | 主题 | 核心技术 |
|------|------|----------|
| [01_Transformer的诞生与需求.md](01_Transformer的诞生与需求.md) | Transformer 为什么被发明 | RNN, LSTM, Attention, Multi-head Attention |
| [02_深入理解Transformer架构.md](02_深入理解Transformer架构.md) | Transformer 内部机制 | Seq2Seq, Encoder-Decoder, Self-Attention, Positional Encoding, FFN |
| [03_文本摘要.md](03_文本摘要.md) | 文本摘要任务 | TextRank, BART, T5, ProphetNet, Pegasus, Longformer, BigBird, ROUGE, BLEU |
| [04_机器翻译.md](04_机器翻译.md) | 机器翻译任务 | mBART, mBART-50, XLM, XLM-RoBERTa, M-BERT, mT5, BPE, SentencePiece, METEOR |
| [05_文本分类.md](05_文本分类.md) | 文本分类任务 | Naive Bayes, BERT, RoBERTa, ALBERT, DistilBERT, DeBERTa, ELECTRA, Confusion Matrix, F1 |
| [06_文本生成.md](06_文本生成.md) | 文本生成任务 | GPT-1/2/3, InstructGPT, GPT-NeoX, Llama, Alpaca, Dolly, Falcon, Greedy/Beam/Top-k/Nucleus/Temperature |
| [07_控制生成文本.md](07_控制生成文本.md) | 控制 LLM 输出 | RLHF, DPO, Prompt Engineering (Zero-shot, Few-shot, CoT, ToT) |
| [08_多模态模型.md](08_多模态模型.md) | 多模态 Transformer | BLIP, BLIP-2, CLIP, X-CLIP, Flamingo, GPT-4V, LLaVA |
| [09_优化与评估大语言模型.md](09_优化与评估大语言模型.md) | 模型优化与评估 | Hyperparameter Tuning, Pruning, Distillation, LoRA, DoRA, Quantization, QLoRA, Sharding |
| [附录_运行代码指南.md](附录_运行代码指南.md) | 代码运行指南 | Python, PyTorch, Hugging Face, GPU |

## 学习路径建议

### 入门路线（Part 1）
1. **第1章** - 了解 Transformer 的诞生背景，理解为什么需要 Attention 机制
2. **第2章** - 深入理解 Transformer 架构的核心组件

### 基础应用（Part 2）
3. **第3章** - 文本摘要：从 TextRank 基线到 Transformer 模型
4. **第4章** - 机器翻译：理解多语言模型和分词技术
5. **第5章** - 文本分类：掌握 BERT 系列模型和评估方法

### 进阶应用（Part 3）
6. **第6章** - 文本生成：从 GPT 系列到各种解码策略
7. **第7章** - 控制 LLM：RLHF、DPO 和 Prompt Engineering
8. **第8章** - 多模态：视觉-语言模型
9. **第9章** - 优化部署：LoRA、量化、蒸馏等高效技术

## 技术全景图

```
Transformer 生态系统
├── 基础架构
│   ├── Attention 机制
│   ├── Multi-Head Attention
│   ├── Positional Encoding
│   └── Encoder-Decoder
├── 摘要模型
│   ├── BART (双向+自回归)
│   ├── T5 (统一文本到文本)
│   ├── ProphetNet (未来n-gram预测)
│   ├── Pegasus (间隙句子生成)
│   ├── Longformer (长文档窗口注意力)
│   └── BigBird (稀疏注意力)
├── 翻译模型
│   ├── mBART / mBART-50
│   ├── XLM / XLM-RoBERTa
│   ├── M-BERT
│   └── mT5
├── 分类模型
│   ├── BERT
│   ├── RoBERTa
│   ├── ALBERT
│   ├── DistilBERT
│   ├── DeBERTa
│   └── ELECTRA
├── 生成模型
│   ├── GPT-1/2/3
│   ├── InstructGPT
│   ├── Llama / Alpaca
│   ├── Dolly / Falcon
│   └── GPT-NeoX-20B
├── 多模态模型
│   ├── BLIP / BLIP-2
│   ├── CLIP / X-CLIP
│   ├── Flamingo / OpenFlamingo
│   ├── GPT-4 Vision
│   └── LLaVA
├── 对齐与控制
│   ├── RLHF (人类反馈强化学习)
│   ├── DPO (直接偏好优化)
│   └── Prompt Engineering
└── 优化技术
    ├── LoRA / DoRA
    ├── Quantization
    ├── QLoRA / QA-LoRA
    ├── Pruning
    ├── Distillation
    └── Sharding
```

## 使用说明

1. 按学习路径顺序阅读效果最佳
2. 每个文件顶部包含该章学习要点
3. 文中代码可直接在 Python 环境中运行
4. 建议配合原书仓库使用：https://github.com/Nicolepcx/Transformers-in-Action
5. 推荐使用 GPU 环境运行代码（Google Colab 等）

## 前置知识要求

- 中级 Python 编程能力
- 基础机器学习概念
- 深度学习基础
- 线性代数、统计学、微积分基础
- 了解 NLP 基本概念
- PyTorch、NumPy、Pandas 基础使用
