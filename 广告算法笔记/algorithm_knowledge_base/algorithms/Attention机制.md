# Attention 机制 学习文档

## 1. 算法基础认知

Attention 机制是深度学习中的一种加权聚合方法，允许模型对输入的不同部分赋予不同权重。在广告系统中广泛用于用户行为序列建模。

## 2. 核心原理

### 通用 Attention 公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 在广告系统中的三种 Attention 类型

| 类型 | 说明 | 代表模型 |
|------|------|---------|
| Target Attention (DIN) | 对候选广告相关的历史行为加权 | DIN/DIEN |
| Multi-Head Self-Attention | Transformer 风格，捕捉序列内依赖 | BST/DSIN/OneTrans |
| Cross-Attention | User-Ad 交叉注意力，增强匹配建模 | InterFormer |

### DIN 中的 Attention

$$
\alpha_i = \frac{\exp(\mathbf{e}_i^\top \mathbf{W} \mathbf{e}_{ad})}{\sum_j \exp(\mathbf{e}_j^\top \mathbf{W}\mathbf{e}_{ad})}
$$

## 3. 应用场景

- CTR 预估中的用户兴趣建模（DIN/DIEN）
- 序列行为建模（BST/DSIN/OneTrans）
- 重排中的上下文建模（PRM/DLCM）
- 多模态特征融合（Cross-Modal Attention）

## 4. 学习总结

Attention 机制是广告推荐模型的核心组件，从 DIN 的 Target Attention 到 Transformer 的 Self-Attention，再到 OneTrans 的统一注意力，不断演进。
