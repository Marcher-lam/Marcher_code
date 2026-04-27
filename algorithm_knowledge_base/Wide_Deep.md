# Wide & Deep 学习文档

> 推荐系统中结合线性模型与深度学习的经典架构

---

## 1. 算法基础认知

**一句话定义**：Wide & Deep（宽深结合）是由Google研究员于2016年提出的推荐系统模型，同时学习模型的"记忆能力"（Wide部分）和"泛化能力"（Deep部分），在Google Play商店推荐中获得巨大提升。

**直觉类比**：Wide & Deep就像一个既有经验又有想象力的店员。Wide部分是老店员，记得所有顾客过去买过什么——你买过手机壳？推荐更多手机壳！Deep部分是学习能力强的年轻人，能推理如果你买过手机，可能也需要屏幕保护膜。两者结合，既精准推荐你熟悉的，又推荐你可能感兴趣的。

**历史背景**：
- 2016年，Google的Cheng等人在论文"Wide & Deep Learning for Recommender System"中提出
- 用于Google Play商店百万级APP推荐
- 后续演进出Wide & Deep Learning for Ad等

**核心定位**：
- 类型：推荐系统 → 排序模型
- 输出：点击/购买概率
- 模型类型：混合模型

**前置知识**：
- [必备]：推荐系统基础（协同过滤）
- [必备]：深度学习（全连接层）
- [推荐]：Embedding

---

## 2. 核心原理

### 2.1 传统LR的问题

**LR**（逻辑回归）：
- 优点：可解释、记忆强
- 缺点：泛化差，需要人工特征

**DNN**：
- 优点：自动特征泛化
- 缺点：记忆弱，稀疏数据上表现差

### 2.2 Wide & Deep的核心思想

**同时保留两者优点！**

| 部分 | 作用 | 特点 |
|------|------|------|
| Wide | 记忆 | 直接记住历史模式 |
| Deep | 泛化 | 推断潜在兴趣 |

### 2.3 架构图

```
    输入特征
       │
  ┌────┴────┐
  │         │
Wide     Deep
  │         │
  │  ┌──────┴──────┐
  │  ▼             ▼
  │ Embeddings    Embeddings
  │   │             │
  │   └──────┬──────┘
  │          ▼
  │    拼接 Concatenate
  │          │
  ▼          ▼
FC层 → FC层 → Output
  │          │
  └────┬────┘
       ▼
    Sigmoid
       │
    预测概率
```

---

## 3. 数学公式与推导

### 3.1 Wide部分

线性模型（逻辑回归）：

$$y = \sigma(W_{wide}^T x + b)$$

其中x是稀疏特征（one-hot或计数）。

### 3.2 Deep部分

深度神经网络：

$$y = \sigma(W_{deep}^T \cdot ReLU(W_{d-1} \cdot ... ReLU(W_1 \cdot E(x)) + b) + b)$$

其中E是Embedding层。

### 3.3 联合输出

$$\hat{y} = \sigma(W_{wide}^T [x_w, x_d] + W_{deep}^T h_{deep} + b)$$

### 3.4 损失函数

对数损失（CTR预测）：

$$L = -\frac{1}{N} \sum [y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

加上L2正则：

$$L_{total} = L + \lambda \|W\|^2$$

---

## 4. 训练过程讲解

### 4.1 特征处理

**稀疏特征**：
- 类别特征 → Embedding
- 数值特征 → 归一化

**Embedding维度**：通常8-64

### 4.2 训练流程

```
    特征输入
        │
        ▼
    ┌───────────┐
    │ Wide部分  │ ← LR层
    └─────┬─────┘
        │
        ▼
    ┌───────────┐
    │ Deep部分  │ ← DNN层
    └─────┬─────┘
        │
        ▼
    ┌───────────┐
    │ 联合输出  │ ← Sigmoid
    └─────┬─────┘
        │
        ▼
    ┌───────────┐
    │ 计算损失  │
    └─────┬─────┘
        │
        ▼
    ┌───────────┐
    │ 反向传播  │
    └───────────┘
```

### 4.3 超参数

| 参数 | 典型值 |
|------|--------|
| embedding_dim | 32 |
| hidden_layers | [128, 64, 32] |
| learning_rate | 0.001 |
| batch_size | 256 |

### 4.4 优化器

通常使用Adam或Adagrad。

---

## 5. 应用场景

### 5.1 APP推荐

Google Play商店推荐：

- 特征：用户安装的APP、类别、安装时间
- 目标：CTR

### 5.2 电商推荐

商品推荐：

- 特征：浏览、点击、购买历史
- 目标：购买率

### 5.3 搜索排序

搜索结果排序：

- 特征：query、用户历史
- 目标：相关性

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **记忆+泛化** | 同时具备两者优势 |
| **端到端** | 联合训练 |
| **可扩展** | 特征和结构可扩展 |
| **工业级** | 已在Google验证 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **实现复杂** | 需要特征工程 |
| **调参难** | 多个超参数 |
| **不如DIN** | 序列行为建模弱 |

### 6.3 改进方案

| 改进 | 方法 |
|------|------|
| DeepFM | Wide → FM |
| DCN | 交叉网络 |
| DIN | 注意力机制 |

---

## 7. 调库实现

### 7.1 PyTorch实现

```python
import torch
import torch.nn as nn


class WideComponent(nn.Module):
    """Wide部分"""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)


class DeepComponent(nn.Module):
    """Deep部分"""
    def __init__(self, embed_dim, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        input_dim = embed_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = h_dim
            
        self.dnn = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.dnn(x)


class WideAndDeep(nn.Module):
    """Wide & Deep模型"""
    
    def __init__(self, wide_dim, embed_dim, deep_dims=[64, 32]):
        super().__init__()
        
        self.wide = WideComponent(wide_dim)
        self.deep = DeepComponent(embed_dim, deep_dims)
        
        # 输出层
        total_dim = 1 + deep_dims[-1]
        self.output = nn.Linear(total_dim, 1)
        
    def forward(self, wide_x, deep_x):
        wide_out = self.wide(wide_x)
        deep_out = self.deep(deep_x)
        
        combined = torch.cat([wide_out, deep_out], dim=-1)
        
        output = torch.sigmoid(self.output(combined))
        
        return output


# 使用示例
if __name__ == "__main__":
    # 假设特征
    wide_features = torch.randn(32, 10)  # 宽特征
    deep_features = torch.randn(32, 8, 32)  # 深特征(批量, 序列, embedding)
    
    # 展平深度特征
    deep_features = deep_features.reshape(32, -1)
    
    model = WideAndDeep(wide_dim=10, embed_dim=8*32)
    output = model(wide_features, deep_features)
    
    print(f"输出形状: {output.shape}")
```

### 7.2 TensorFlow实现

```python
# TensorFlow实现
import tensorflow as tf

class WideAndDeepModel(tf.keras.Model):
    def __init__(self, wide_dim, embed_dim, deep_dim):
        super().__init__()
        
        # Wide
        self.wide_layer = tf.keras.layers.Dense(1)
        
        # Deep
        self.embed = tf.keras.layers.Embedding(1000, embed_dim)
        self.deep_layers = [
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ]
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, wide_x, deep_x):
        # Wide
        wide_out = self.wide_layer(wide_x)
        
        # Deep
        deep_out = self.embed(deep_x)
        deep_out = tf.reshape(deep_out, [tf.shape(deep_out)[0], -1])
        
        for layer in self.deep_layers:
            deep_out = layer(deep_out)
        
        # 合并
        combined = tf.concat([wide_out, deep_out], axis=-1)
        
        return self.output_layer(combined)
```

### 7.3 DeepTables库

```python
# 使用DeepTables
from deeptables.models import WideDeepModel

model = WideDeepModel(
    sets=['wide', 'deep'],
    early_stopping=False
)
model.fit(df)
```

---

## 8. 手工代码实现

### 8.1 简化实现

```python
import numpy as np
import torch
import torch.nn as nn


class WideAndDeep:
    """简化版Wide & Deep"""
    
    def __init__(self, wide_dim, embed_dim, dnn_dim=[64, 32]):
        self.wide_dim = wide_dim
        self.embed_dim = embed_dim
        
        # Wide
        self.W_wide = np.random.randn(wide_dim, 1) * 0.01
        self.b_wide = 0.0
        
        # Deep简化为矩阵乘法
        self.W_deep = [np.random.randn(embed_dim, dnn_dim[0]) * 0.01]
        for i in range(len(dnn_dim)-1):
            self.W_deep.append(np.random.randn(dnn_dim[i], dnn_dim[i+1]) * 0.01)
            
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, wide_x, deep_x):
        # Wide部分
        wide_out = np.dot(wide_x, self.W_wide) + self.b_wide
        
        # Deep部分
        deep_out = deep_x
        for W in self.W_deep:
            deep_out = self.relu(np.dot(deep_out, W))
        
        # 合并
        combined = np.concatenate([wide_out, deep_out], axis=-1)
        
        # 输出
        output = self.sigmoid(np.mean(combined))
        
        return output


def train_wide_deep():
    """训练示例"""
    np.random.seed(42)
    
    # 生成数据
    n_samples = 1000
    wide_dim = 10
    embed_dim = 32
    
    X_wide = np.random.randn(n_samples, wide_dim)
    X_deep = np.random.randn(n_samples, embed_dim)
    y = (X_wide[:, 0] + np.random.randn(n_samples)*0.1 > 0).astype(float)
    
    # 模型
    model = WideAndDeep(wide_dim, embed_dim)
    
    # 训练
    lr = 0.01
    for epoch in range(100):
        total_loss = 0
        for i in range(n_samples):
            pred = model.forward(X_wide[i:i+1], X_deep[i:i+1])
            loss = -(y[i] * np.log(pred+1e-8) + (1-y[i])*np.log(1-pred+1e-8))
            
            # 简单梯度下降
            # (简化)
            total_loss += loss
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/n_samples:.4f}")


if __name__ == "__main__":
    train_wide_deep()
```

---

## 9. 特征工程

### 9.1 特征类型

| 特征 | Wide | Deep |
|------|------|------|
| 用户ID | ✔ | Embedding |
| 类别特征 | ✔ | Embedding |
| 数值特征 | 归一化 | Embedding |
| 历史行为 | - | RNN/Embedding |

### 9.2 特征处理

```python
# 类别特征
category_feature = pd.Categorical(df['category'])
category_codes = category_feature.codes

# 数值特征
numeric_feature = (df['value'] - mean) / std

# 历史行为
user_history = df.groupby('user_id')['item_id'].apply(list)
```

---

## 10. 模型评估

### 10.1 指标

| 指标 | 说明 |
|------|------|
| AUC | 排序质量 |
| LogLoss | 预测精度 |
| CTR | 点击率 |

### 10.2 对比

| 模�� | AUC |
|------|-----|
| LR | 0.75 |
| DNN | 0.78 |
| **Wide&Deep** | **0.80** |

---

## 11. 常见问题与易错点

### 11.1 特征维度

问题：Wide特征和Deep特征维度不匹配

解决：分别定义，合并前对齐

### 11.2 过拟合

解决：Dropout、L2正则

### 11.3 稀疏特征

解决：使用特征Hash

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| Wide | 记忆，直接特征 |
| Deep | 泛化，Embedding |
| 联合 | 端到端 |

### 12.2 扩展

- DeepFM
- DCN
- xDeepFM

---

## 13. 练习题

### 13.1 基础

1. Wide和Deep的区别？
2. 为什么需要两者结合？

### 13.2 进阶

1. vs DeepFM
2. 序列行为如何建模？

---

## 14. 学习路径

1. 推荐系统基础
2. 特征工程
3. Wide & Deep
4. 进阶模型

---

## 附录

### 参考

- 论文：Cheng et al., 2016
- 库：TensorFlow Recommenders

---

**文档结束**