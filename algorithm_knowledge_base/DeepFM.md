# DeepFM（深度因子分解机）学习文档

> 结合FM与深度学习的推荐模型，自动学习特征交互

---

## 1. 算法基础认知

### 1.1 一句话定义

DeepFM是2017年由华为诺亚实验室提出的推荐模型，结合了FM（一阶+二阶特征交互）和深度神经网络（高阶特征交互），无需特征工程就能学习所有阶数的特征交互。

### 1.2 直觉类比

DeepFM = Wide&Deep的升级版！Wide&Deep的Wide部分需要人工选定"记忆"哪些特征，但DeepFM自动学习！它同时用FM捕捉简单的特征组合，用DNN捕捉复杂的特征组合，一个模型搞定一切。

想象你在做电商推荐：
- Wide&Deep：需要人工告诉模型"用户买过A也可能买B"（Wide部分）
- DeepFM：自动从数据中发现用户-商品、类别-品牌之间的复杂关系！

### 1.3 发展背景

- 2017年，Guo等人在arxiv发表"DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
- 继承Wide&Deep思想但用FM替代Wide部分
- 解决特征工程繁琐的问题

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 推荐系统 → CTR预测 |
| 输出 | 点击概率 |
| 模型 | FM + DNN |
| 特点 | 端到端自动特征交互 |

### 1.5 前置知识

- [必备]：推荐系统基础
- [必备]：FM模型
- [推荐]：Wide&Deep

---

## 2. 核心原理

### 2.1 为什么需要DeepFM？

**CTR预测问题**：
- 输入：大量特征（用户ID、商品ID、类别、品牌、年龄等）
- 输出：点击概率

**现有方法问题**：
- 线性模型：需要大量特征工程
- Wide&Deep：需要人工设计Wide部分
- DeepFM：自动学习所有阶数的特征交互！

### 2.2 架构对比

| 模型 | Wide部分 | Deep部分 | 特征交互阶数 |
|------|---------|---------|--------------|
| 线性 | 人工 | 无 | 一阶 |
| FM | 自动 | 无 | 二阶 |
| Wide&Deep | 人工 | MLP | 混合 |
| **DeepFM** | **FM** | **MLP** | **全部阶数** |

### 2.3 核心公式

$$\hat{y} = \sigma(w_0 + \underbrace{\sum w_i x_i}_{一阶} + \underbrace{\sum_{i<j} <v_i, v_j> x_i x_j}_{二阶FM} + \underbrace{DNN(x)}_{高阶})$$

### 2.4 架构图

```
特征输入 [稀疏]
    │
    ├──→ Embedding ──→ 共享 ──▶ 输出
    │    (共享嵌入)      │ (拼接)
    ▼                  ▼
FM组件 ────────────────┼─────────▶ Sigmoid
    │ (线性+二阶)     │
    │                  ▼
DNN组件 ─────────────┘
    │ (高阶MLP)
    ▼
```

---

## 3. 数学公式与推导

### 3.1 符号定义

- $\mathcal{X}$ = 特征空间
- $x_i \in \mathcal{X}$ = 特征值
- $w_i$ = 一阶权重
- $v_i$ = 二阶Embedding向量

### 3.2 一阶部分（Linear）

$$y_{linear} = w_0 + \sum_{i=1}^{n} w_i x_i$$

这相当于LR模型，直接学习特征权重。

### 3.3 二阶部分（FM）

$$y_{FM} = \sum_{i=1}^{n} \sum_{j=i+1}^{n} <v_i, v_j> x_i x_j$$

展开为：
$$<v_i, v_j> = \sum_{k=1}^{K} v_{ik} \cdot v_{jk}$$

使用公式简化（避免O(n²)）：
$$y_{FM} = \frac{1}{2} \sum_{k=1}^{K} ((\sum_i v_{ik} x_i)^2 - \sum_i v_{ik}^2 x_i^2)$$

推导：
- 原始：$y_{FM} = sum_{i<j} v_i \cdot v_j = \frac{1}{2}((sum_i v_i)^2 - sum_i v_i^2)$
- 复杂度从O(n²)降到O(n)

### 3.4 深度部分（DNN）

```
输入: x → Embedding层
    ↓
MLP: [d] → [128] → [64] → [1]
    ↓
输出: y_DNN
```

前向传播：
$$h^{(1)} = \sigma(W^{(1)} x + b^{(1)})$$
$$h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})$$

### 3.5 总输出

$$\hat{y} = \sigma(y_{linear} + y_{FM} + y_{DNN})$$

---

## 4. 训练过程讲解

### 4.1 特征处理

```python
# 特征类别
features = {
    'user_id': 'categorical',
    'item_id': 'categorical',
    'age': 'numerical',
    'category': 'categorical',
    'brand': 'categorical',
}

# 稀疏编码 (one-hot)
# user_id: [0, 3, 0, 0, ...]
# item_id: [1, 0, 5, 0, ...]
```

### 4.2 Embedding共享

FM和DNN共享同一个Embedding层：

```python
class SharedEmbeddings(nn.Module):
    def __init__(self, num_features, embed_dim):
        self.embedding = nn.Embedding(num_features, embed_dim)
```

优势：
- 减少参数量
- 避免重复学习

### 4.3 训练配置

| 参数 | 典型值 |
|------|--------|
| lr | 0.001 |
| batch_size | 256 |
| epochs | 10 |
| embed_dim | 10 |
| hidden_layers | [256, 128, 64] |
| dropout | 0.3 |

### 4.4 损失函数

使用Binary Cross-Entropy：

$$L = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

---

## 5. 应用场景

### 5.1 CTR预测

```
输入特征:
- user_id: 12345
- item_id: 67890
- category: electronics
- brand: Apple

输出:
点击概率 = 0.73
```

### 5.2 推荐排序

```python
# 对多个候选排序
candidates = [...]
scores = []
for item in candidates:
    score = deepfm.predict(user_features, item_features)
    scores.append((item, score))

# 排序输出
ranked = sorted(scores, key=lambda x: x[1], reverse=True)
```

### 5.3 对比其他模型

| 模型 | AUC | LogLoss |
|------|-----|--------|
| LR | 0.76 | 0.45 |
| FM | 0.78 | 0.42 |
| Wide&Deep | 0.80 | 0.39 |
| **DeepFM** | **0.81** | **0.38** |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **自动特征交互** | 无需特征工程 |
| **共享Embedding** | 减少参数量 |
| **端到端** | 一个模型 |
| **全部阶数** | 一阶+二阶+高阶 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **计算重** | DNN部分开销 |
| **调参难** | 多个超参数 |
| **可解释性** | 较弱 |

### 6.3 注意事项

- embed_dim不宜太大，10-50足够
- DNN深度2-3层即可
- 需要大量训练数据
- 特征需要稀疏编码

---

## 7. 调库实现（Python）

### 7.1 DeepTables

```python
from deeptables.models import DeepTable
from deeptables.models.selectors import *
from sklearn.model_selection import train_test_split

# 准备数据
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 创建模型
dt = DeepTable(
   nets=[
        WideNumeric(),
        LinearNumeric(),
        FactorizationMachine(name="fm"),
        DenseNet()
   ]
)

# 训练
dt.fit(X_train, y_train)

# 预测
predictions = dt.predict(X_test)
```

### 7.2 TensorFlow Recommenders

```python
import tensorflow_recommenders as tfrs
import tensorflow as tf

class DeepFMModel(tfrs.Model):
    def __init__(self, embedding_dim, vocab_sizes):
        super().__init__()
        
        # Embedding
        self.embeddings = tf.keras.layers.Embedding(
            sum(vocab_sizes) + 1, 
            embedding_dim
        )
        
        # FM部分
        self.fm_weights = tf.keras.layers.Dense(1)
        
        # DNN部分
        self.dnn = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    def call(self, features):
        embedded = self.embeddings(features)
        
        # FM
        fm = tf.reduce_sum(embedded, axis=1)
        
        # DNN
        dnn = self.dnn(embedded)
        
        return tf.sigmoid(fm + dnn)
```

### 7.3 PyTorch实现

```python
import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, feature_fields, embed_dim=10, hidden_dims=[128, 64]):
        super().__init__()
        
        # Embedding
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num, embed_dim)
            for name, num in feature_fields.items()
        })
        
        # Linear权重
        self.linear = nn.Embedding(len(feature_fields), 1)
        
        # DNN
        input_dim = len(feature_fields) * embed_dim
        layers = []
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = dim
        self.dnn = nn.Sequential(*layers)
        
        # 输出
        self.output = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, features):
        # Linear
        linear_out = self.linear(features).sum(dim=1)
        
        # Embedding
        embedded = torch.stack([
            self.embeddings[name](features[:, i])
            for i, name in enumerate(self.embeddings.keys())
        ], dim=1)
        
        # FM
        sum_square = (embedded ** 2).sum(dim=1)
        square_sum = (embedded.sum(dim=1) ** 2)
        fm_out = 0.5 * (square_sum - sum_square).sum(dim=1)
        
        # DNN
        dnn_input = embedded.reshape(embedded.size(0), -1)
        dnn_out = self.dnn(dnn_input)
        dnn_out = self.output(dnn_out).squeeze(-1)
        
        # 总输出
        logit = linear_out + fm_out + dnn_out
        return torch.sigmoid(logit)
```

---

## 8. 手工代码实现（理解原理）

### 8.1 简化版DeepFM

```python
import numpy as np

class DeepFMManual:
    """简化版DeepFM - 理解原理"""
    
    def __init__(self, n_features, embed_dim=10, lr=0.01):
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.lr = lr
        
        # 权重初始化
        self.w = np.random.randn(n_features) * 0.01
        self.V = np.random.randn(n_features, embed_dim) * 0.01
        
        # DNN权重 (简化)
        self.W1 = np.random.randn(embed_dim * 2, 16) * 0.01
        self.b1 = np.zeros(16)
        self.W2 = np.random.randn(16, 1) * 0.01
        self.b2 = np.zeros(1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        n = X.shape[0]
        
        # Linear (一阶)
        linear_out = np.zeros(n)
        for i in range(n):
            for j, val in enumerate(X[i]):
                if val > 0:
                    linear_out[i] += self.w[j] * val
        
        # FM (二阶)
        v = self.V[X]
        sum_v = v.sum(axis=1)
        sum_square = (v ** 2).sum(axis=1)
        fm_out = 0.5 * ((sum_v ** 2).sum(axis=1) - sum_square.sum(axis=1))
        
        # DNN (高阶)
        dnn_input = np.concatenate([linear_out.reshape(-1, 1), fm_out.reshape(-1, 1)], axis=1)
        h = np.maximum(0, dnn_input @ self.W1 + self.b1)
        dnn_out = h @ self.W2 + self.b2
        
        # 总输出
        logit = linear_out + fm_out.flatten() + dnn_out.flatten()
        return self.sigmoid(logit)
    
    def fit(self, X, y, epochs=10):
        """训练"""
        for epoch in range(epochs):
            pred = self.forward(X)
            loss = -np.mean(y * np.log(pred + 1e-8) + (1-y) * np.log(1-pred + 1e-8))
            print(f"Epoch {epoch}: Loss={loss:.4f}")


if __name__ == "__main__":
    np.random.seed(42)
    
    # 模拟数据
    n = 1000
    X = np.random.randint(0, 10, (n, 5))
    y = np.random.randint(0, 2, n)
    
    # 训练
    model = DeepFMManual(n_features=X.max()+1)
    model.fit(X, y)
```

---

## 9. 可视化与结果理解

### 9.1 特征交互可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_interactions():
    """可视化特征交互"""
    
    np.random.seed(42)
    
    # 模拟特征重要性
    features = ['user_age', 'item_price', 'category', 'brand', 'history']
    importance = np.random.rand(len(features))
    importance = importance / importance.sum()
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance)
    plt.xlabel('Importance')
    plt.title('DeepFM Feature Importance')
    plt.savefig('deefm_importance.png', dpi=100)
    plt.show()


def visualize_embedding():
    """可视化Embedding"""
    
    np.random.seed(42)
    n_features = 20
    embed_dim = 2
    
    embeddings = np.random.randn(n_features, embed_dim)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    
    for i in range(n_features):
        plt.annotate(f'F{i}', (embeddings[i, 0], embeddings[i, 1]))
    
    plt.title('DeepFM Feature Embeddings')
    plt.savefig('deefm_embedding.png', dpi=100)
    plt.show()


if __name__ == "__main__":
    visualize_feature_interactions()
    visualize_embedding()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| AUC | 排序效果 | >0.8 |
| LogLoss | 损失 | <0.4 |
| CTR@K | Top-K点击率 | 高 |
| Recall@K | Top-K召回 | 高 |

### 10.2 评估代码

```python
from sklearn.metrics import roc_auc_score, log_loss

def evaluate_deepfm(y_true, y_pred):
    """评估DeepFM"""
    auc = roc_auc_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    
    print(f"AUC: {auc:.4f}")
    print(f"LogLoss: {ll:.4f}")
    
    return {'AUC': auc, 'LogLoss': ll}
```

### 10.3 调参建议

```python
# 最优参数范围
params = {
    'embed_dim': [10, 15, 20],
    'hidden_dims': [[256, 128], [128, 64]],
    'lr': [0.001, 0.0005],
    'dropout': [0.2, 0.3, 0.5]
}
```

---

## 11. 常见问题与易错点

### Q1: 特征太多怎么办？

**答��**：用特征哈希或合并low-frequency特征。

### Q2: 训练太慢？

**答案**：减小embed_dim或用Adam优化器。

### Q3: 过拟合？

**答案**：加Dropout、L2正则。

### Q4: Embedding不收敛？

**答案**：检查特征编码，增加正则。

### Q5: AUC低？

**答案**：增加特征交互，检查数据质量。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心公式 | $\hat{y} = \sigma(w_0 + y_{linear} + y_{FM} + y_{DNN})$ |
| FM | 二阶特征交互 |
| DNN | 高阶特征交互 |
| 共享 | Embedding |

### 12.2 公式汇总

一阶：
$$y_{linear} = w_0 + \sum w_i x_i$$

二阶：
$$y_{FM} = \frac{1}{2} \sum_k ((\sum_i v_{ik}x_i)^2 - \sum_i v_{ik}^2 x_i^2)$$

损失：
$$L = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. DeepFM的FM部分学习几阶特征交互？
   - A) 一阶
   - B) 二阶
   - C) 高阶

2. DeepFM的DNN部分学习几阶特征交互？
   - A) 一阶
   - B) 二阶
   - C) 高阶

3. FM和DNN共享什么？
   - A) 权重
   - B) Embedding
   - C) 输出

### 13.2 简答题

1. 解释FM如何简化二阶计算？
2. 比较DeepFM和Wide&Deep的区别？
3. 为什么DeepFM不需要特征工程？

### 13.3 编程题

1. 用PyTorch实现DeepFM。
2. 比较不同embed_dim的效果。
3. 用真实数据集训练并评估。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
推荐系统基础
    ↓
LR模型
    ↓
FM模型
    ↓
Wide&Deep
    ↓
DeepFM
    ↓
实战项目
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| FM | 二阶特征交互 |
| Wide&Deep | Wide+Deep架构 |
| DCN | 交叉网络 |
| AutoInt | 注意力特征交互 |

### 14.3 扩展阅读

1. Guo et al. (2017). DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
2. Cheng et al. (2016). Wide & Deep Learning for Recommender Systems

---

## 附录

### A. 超参数速查

| 参数 | 推荐值 |
|------|--------|
| embed_dim | 10-20 |
| hidden_dims | [128, 64] |
| lr | 0.001 |
| dropout | 0.2-0.3 |

### B. 参考

1. Guo et al. (2017). DeepFM. arXiv:1703.04247
2. DeepTables库

---

**文档结束**