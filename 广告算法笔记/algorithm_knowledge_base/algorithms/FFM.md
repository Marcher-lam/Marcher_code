# FFM（Field-aware Factorization Machine）学习文档

## 1. 算法基础认知

FFM 是 FM 的扩展，引入了"域"（Field）的概念。不同域的特征交互使用不同的隐向量，比 FM 更精细地建模特征交叉。

## 2. 核心原理

### FM 公式

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

### FFM 公式

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_{i,f_j}, \mathbf{v}_{j,f_i}\rangle x_i x_j
$$

其中 $f_j$ 是特征 j 所属的域。每个特征对每个域维护一个独立的隐向量。

## 3. 应用场景

- 稀疏特征交叉（比 FM 更精细）
- 推荐系统 baseline
- 广告 CTR 预估

## 4. 与 FM 对比

| 特性 | FM | FFM |
|------|----|----|
| 隐向量数量 | 每特征 1 个 | 每特征×每域 1 个 |
| 参数量 | O(nk) | O(nfk) |
| 表达能力 | 中等 | 较强 |
| 训练速度 | 快 | 较慢 |

## 5. 学习总结

FFM 是 FM/FFM 系列中处理稀疏特征交叉的经典方法，在业界广泛作为 baseline。在广告排序中，FM/FFM 常作为特征交叉的基础模块。
