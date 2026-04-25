# DIEN（深度兴趣演化网络）学习文档

> 阿里巴巴提出的序列推荐模型，捕捉用户兴趣的演化过程

---

## 1. 算法基础认知

**一句话定义**：DIEN（Deep Interest Evolution Network，深度兴趣演化网络）是由阿里巴巴的Zhou等于2018年提出的序列推荐模型，通过 GRU 和注意力机制捕捉用户兴趣随时间的演化过程，解决推荐中用户兴趣动态变化的问题。

**直觉类比**：DIEN就像一个"记住你变化心路"的海报。回想你买手机的过程：2个月前你开始搜索（这时可能只是好奇）→1个月前开始比较几款手机（有了明确目标）→上周开始频繁看评测视频（进入决策阶段）→今天终于下单。DIEN捕捉的就是这个兴趣变化过程，不是单纯记住你看过什么，而是理解你"从好奇到下单"的心路历程。

**历史背景**：
- 2018年，阿里巴巴Zhou等人在论文"Deep Interest Evolution Network for Click-Through Rate Prediction"中提出
-应用于淘宝商品推荐系统
- 在DIN基础上增加时间序列建模

**核心定位**：
- 类型：推荐系统 → 序列建模
- 输出：CTR预测
- 模型类型：GRU + Attention

**前置知识**：
- [必备]：推荐系统基础（协同过滤、DIN）
- [必备]：RNN/LSTM基础
- [推荐]：注意力机制

---

## 2. 核心原理

### 2.1 之前模型的问题

| 模型 | 问题 |
|------|------|
| DIN | 只考虑当前兴趣，忽略时序 |
| WDL | 无序列建模范 |
| RNN | 简单RNN不够精确 |

**核心局限**：无法建模兴趣随时间的变化！

### 2.2 DIEN核心创新

**双层 GRU + 注意力**：

```
用户行为序列 (时间顺序)
    │
    ▼
┌────────────────────────┐
│ Interest Extractor    │  ← 第一层GRU
│ 提取用户短期兴趣       │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Interest Evolver       │  ← 第二层GRU+注意力
│ 建模兴趣演化方向      │
└───────────┬────────────┘
            │
            ▼
          预测
```

### 2.3 整体架构

```
             ┌────────────────────────────────┐
             │        Embedding Layer       │
             └─────────────┬───────────────┘
                           │
        ┌─────────────────┴──────────────────┐
        ▼                                    ▼
┌───────────────────┐            ┌───────────────────┐
│   Interest       │            │   Attention Hub   │
│   Extractor GRU  │            │    (AUGRU)       │
└───────┬─────────┘            └────────┬────────┘
        │                             │
        └────────────┬────────────────┘
                     ▼
              ┌─────────────────┐
              │    Output       │
              │  (Sigmoid)     │
              └─────────────────┘
```

---

## 3. 数学公式与推导

### 3.1 兴趣提取层

GRU更新门：
$$u_t = \sigma(W_u \cdot [h_{t-1}, x_t])$$

重置门：
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

候选隐藏：
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$

最终隐藏：
$$h_t = (1 - u_t) \odot h_{t-1} + u_t \odot \tilde{h}_t$$

### 3.2 兴趣演化层（AUGRU）

**核心**：在GRU中加入注意力！

$$\tilde{u}_t' = \frac{exp(score)}{\sum exp(score)}$$

其中score是当前候选和目标的相关性。

最终：
$$h_t^E = (1 - \tilde{u}_t') \odot h_{t-1}^E + \tilde{u}_t' \odot \tilde{h}_t^E$$

### 3.3 损失函数

$$\mathcal{L} = -\frac{1}{N} \sum [y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

---

## 4. 训练过程讲解

### 4.1 训练流程

```
      用户行为序列
            │
            ▼
    ┌───────────────┐
    │ Embedding    │ ← ID转向量
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ GRU Layer 1   │ ← 提取兴趣
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ AUGRU Layer  │ ← 演化+注意力
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ 拼接+输出    │
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │   BCE Loss   │
    └───────────────┘
```

### 4.2 特征处理

| 特征 | 处理 |
|------|------|
| 用户ID | Embedding |
| 行为序列 | GRU/Embedding |
| 上下文 | 直接拼接 |

### 4.3 超参数

| 参数 | 典型值 |
|------|--------|
| embedding_dim | 18 |
| hidden_size | 100 |
| layer_num | 2 |
| attention_dim | 36 |

---

## 5. 应用场景

### 5.1 电商推荐

- 淘宝商品推荐
- 搜索排序

### 5.2 其他序列推荐

- 新闻推荐
- 视频推荐

---

## 6. 优缺点分析

### 6.1 优点

| 优点 |
|------|
| 捕捉兴趣演化 |
| 序列建模 |
| 自动注意力 |

### 6.2 缺点

| 缺点 |
|------|
| 计算重 |
| 需要序列数据 |
| 调参难 |

### 6.3 改进

- DIEN+
- BIM（兴趣记忆）

---

## 7. 调库实现

### 7.1 TensorFlow实现

```python
import tensorflow as tf

class DIENModel(tf.keras.Model):
    def __init__(self, feature_dim, embed_dim, hidden_size):
        super().__init__()
        
        self.embedding = tf.keras.layers.Embedding(feature_dim, embed_dim)
        
        #兴趣提取层GRU
        self.gru_interest = tf.keras.layers.GRU(
            hidden_size, return_sequences=True)
        
        #演化层AUGRU  
        self.gru_evolution = tf.keras.layers.GRU(
            hidden_size, return_sequences=True)
        
        self.attention = tf.keras.layers.Dense(1)
        self.output = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, target_item):
        #嵌入
        embed = self.embedding(inputs)
        
        #兴趣提取
        interest = self.gru_interest(embed)
        
        #兴趣演化（带注意力）
        attention_scores = self.attention(
            tf.concat([interest, self.embedding(target_item)], axis=-1))
        
        evolution = self.gru_evolution(embed)
        
        #拼接输出
        combined = tf.concat([interest[:, -1], evolution[:, -1]], axis=-1)
        
        return self.output(combined)
```

### 7.2 训练示例

```python
#伪代码
model = DIENModel(num_features=10000, embed_dim=18, hidden_size=100)

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.BinaryCrossentropy()

for epoch in range(10):
    for batch in train_data:
        inputs, target, label = batch
        
        with tf.GradientTape() as tape:
            pred = model(inputs, target)
            loss = loss_fn(label, pred)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
    print(f"Epoch {epoch}: Loss {loss:.4f}")
```

---

## 8. 手工代码实现

### 8.1 PyTorch实现

```python
import torch
import torch.nn as nn


class InterestExtractorGRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        # x: [batch, seq, embed]
        output, hidden = self.gru(x)
        return output


class AUGRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x, target_embed):
        # x: [batch, seq, hidden]
        # target_embed: [batch, embed]
        
        gru_out, _ = self.gru(x)
        
        #注意力分数
        target_expand = target_embed.unsqueeze(1).expand_as(gru_out)
        attention = torch.sigmoid(self.attention(
            torch.cat([gru_out, target_expand], dim=-1)))
        
        #加权
        weighted = attention * gru_out
        
        return weighted[:, -1]


class DIEN(nn.Module):
    def __init__(self, num_features, embed_dim, hidden_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(num_features, embed_dim)
        
        self.interest_extractor = InterestExtractorGRU(embed_dim, hidden_dim)
        self.interest_evolution = AUGRU(embed_dim, hidden_dim)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, seq, target):
        # seq: [batch, seq_len]
        # target: [batch]
        
        embed = self.embedding(seq)
        target_embed = self.embedding(target)
        
        interest = self.interest_extractor(embed)
        evolution = self.interest_evolution(interest, target_embed)
        
        combined = torch.cat([interest[:, -1], evolution], dim=-1)
        
        return self.output(combined)


#训练
if __name__ == "__main__":
    model = DIEN(num_features=1000, embed_dim=32, hidden_dim=64)
    
    seq = torch.randint(0, 1000, (32, 10))
    target = torch.randint(0, 1000, (32,))
    label = torch.rand(32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    for epoch in range(5):
        pred = model(seq, target)
        loss = criterion(pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: {loss.item():.4f}")
```

---

## 9. 可视化与结果理解

### 9.1 兴趣演化可视化

```python
import matplotlib.pyplot as plt

def plot_evolution(attention_weights):
    # attention_weights: [seq_len]
    plt.figure(figsize=(10, 5))
    plt.plot(attention_weights)
    plt.xlabel('Time Step')
    plt.ylabel('Attention')
    plt.title('Interest Evolution')
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| AUC | 排序质量 |
| LogLoss | 精度 |
| Recall | 召回 |

### 10.2 对比

| 模型 | AUC |
|------|-----|
| DIN | 0.78 |
| **DIEN** | **0.80** |
| WDL | 0.75 |

---

## 11. 常见问题与易错点

### 11.1 序列长度

问题：序列太短，GRU效果差

解决：填充或截断到固定长度

### 11.2 注意力目标

问题：注意力计算不准确

解决：正确设置target item

### 11.3 序列顺序

问题：序列顺序错误

解决：确保按时间顺序排列

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 双层GRU | 提取+演化 |
| 注意力 | AUGRU |
| 时序建模 | 捕捉兴趣变化 |

### 12.2 扩展

- DIEN+
- MIMN

---

## 13. 练习题

### 13.1 基础

1. DIEN和DIN的区别？
2. AUGRU的作用？

### 13.2 进阶

1. 序列长度如何设置？
2. 注意力如何计算？

---

## 14. 学习路径

1. 推荐系统基础
2. DIN模型
3. DIEN原理
4. 序列建模
5. 实战

---

## 附录

### 参考

- 论文：Zhou et al., 2018
- 库：TensorFlow Recommenders, DeepRec

---

**文档结束**