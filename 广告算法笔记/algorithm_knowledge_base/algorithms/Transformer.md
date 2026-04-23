# Transformer 学习文档

## 1. 算法基础认知

Transformer 是基于 Self-Attention 机制的序列建模架构，在广告系统中被广泛用于用户行为序列建模、特征交互统一建模和生成式推荐。

## 2. 核心原理

### Multi-Head Self-Attention

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

### 在广告系统中的应用

1. **行为序列建模**：BST（Behavior Sequence Transformer）用 Transformer 编码用户行为序列
2. **统一排序架构**：OneTrans 用单个 Transformer 同时处理序列建模和特征交互
3. **生成式推荐**：Decision Transformer 将出价过程建模为序列生成任务
4. **超长序列建模**：LONGER 通过令牌压缩降低 Transformer 的二次计算复杂度

## 3. 关键模型

| 模型 | 核心创新 | 应用 |
|------|---------|------|
| BST | Transformer 编码行为序列 | 用户兴趣建模 |
| OneTrans | 统一序列建模+特征交互 | CTR/CVR 预估 |
| Decision Transformer | 序列生成式出价 | 自动出价 |
| RankMixer | 无参数特征交互 | 排序模型 |

## 4. 工程优化

- FlashAttention-2：IO-Aware 的 tiling 分块计算，2~4× 加速
- GQA（Grouped Query Attention）：减少 KV 头数
- Linear Attention：线性复杂度替代
- 跨请求 KV 缓存

## 5. 学习总结

Transformer 已成为广告推荐系统的核心架构，从行为序列建模到统一排序模型，再到生成式推荐和出价，Transformer 的应用范围不断扩大。
