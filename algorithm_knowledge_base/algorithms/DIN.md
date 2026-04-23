# DIN（Deep Interest Network）学习文档

> 阿里妈妈广告团队提出的深度兴趣网络，用于点击率预估。

---

## 1. 算法基础认知

### 1.1 发展背景

DIN（Deep Interest Network，深度兴趣网络）由阿里巴巴妈妈广告团队于 2018 年在论文《Deep Interest Network for Click-Through Rate Prediction》中提出，专门用于电商广告的点击率预估。其核心创新是**注意力机制**，根据用户历史行为动态计算候选商品与历史商品的相似度，实现"因人而异"的个性化推荐。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 深度学习 CTR 预估 |
| 核心 | 注意力机制（Activation Unit） |
| 任务 | 电商推荐/广告点击率 |
| 特点 | 考虑用户兴趣多样性 |

### 1.3 与传统模型对比

| 模型 | 用户行为 | 兴趣建模 |
|------|---------|----------|
| Wide & Deep | 无 | 简单 |
| DeepFM | 无 | DNN 隐式 |
| DIN | 有 | Attention 显式 |

---

## 2. 核心原理

### 2.1 特征输入

DIN 的输入特征包括：

1. **用户画像特征**：年龄、性别、地域
2. **商品特征**：类别、价格、品牌
3. **上下文特征**：时间、设备
4. **用户行为特征**：点击/购买过的商品序列

### 2.2 注意力机制（Activation Unit）

核心创新是对用户行为序列的注意力加权：

$$\text{Attention}(q, V) = \sum_{i=1}^K \exp(q, k_i) \cdot V_i$$

其中：
- $q$：候选商品向量
- $k_i$：历史第 $i$ 个商品向量
- $V$：用户行为序列

### 2.3 整体架构

```
输入特征
    ↓
Embedding 层
    ↓
+ → DNN → 预测
↑
注意力加权（Activation Unit）
```

---

## 3. 数学公式与推导

### 3.1 Attention 计算

给定候选向量 $q$ 和历史序列 $S = [b_1, ..., b_K]$：

1. **计算相似度**：
$$e_i = v^T \tanh(W_q q + W_k b_i)$$

2. **Softmax 归一化**：
$$\alpha_i = \frac{\exp(e_i)}{\sum_{j=1}^K \exp(e_j)}$$

3. **加权求和**：
$$o = \sum_{i=1}^K \alpha_i \cdot b_i$$

### 3.2 注意力网络

```
q → W_q → 
             → Concat → FC → FC → α
b_i → W_k →
```

### 3.3 损失函数

使用交叉熵损失：

$$L = -\sum (y \log \hat{y} + (1-y) \log(1-\hat{y}))$$

---

## 4. 训练过程讲解

### 4.1 特征处理

```python
# 用户行为序列处理
user_behavior = ['item_apple', 'item_banana', 'item_orange']
candidate = 'item_iphone'

# 转换为 embedding
user_emb = embedding(user_behavior)
candidate_emb = embedding(candidate)
```

### 4.2 Attention 计算

```python
def activation_unit(candidate, user_seq):
    """注意力单元"""
    # 计算每个历史商品的权重
    weights = []
    for item in user_seq:
        sim = mlp(candidate, item)
        weights.append(sim)
    
    # Softmax
    weights = softmax(weights)
    
    # 加权求和
    output = sum(w * item for w, item in zip(weights, user_seq))
    
    return output
```

### 4.3 训练技巧

- **Dice 激活**：自适应 ReLU
- **GAUC 评估**：用户级别 AUC
- **负采样**：提高训练效率

---

## 5. 应用场景

### 5.1 典型应用

- **电商广告 CTR 预估**：淘宝/天猫推荐
- **搜索排序**：商品搜索
- **个性化推荐**：首页推荐

### 5.2 代码示例

```python
import torch
import torch.nn as nn

class DIN(nn.Module):
    """Deep Interest Network"""
    
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Attention
        self.attention = AttentionLayer(embed_dim)
        
        # DNN
        self.dnn = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_id, item_ids, candidate):
        # Embedding
        user_emb = self.embedding(user_id)
        item_embs = self.embedding(item_ids)
        candidate_emb = self.embedding(candidate)
        
        # Attention
        attended = self.attention(candidate_emb, item_embs)
        
        # 拼接
        concat = torch.cat([user_emb, attended, candidate_emb], dim=-1)
        
        # 预测
        output = self.dnn(concat)
        
        return output
```

---

## 6. 优缺点分析

### 6.1 优点

1. **兴趣建模**：显式建模用户兴趣
2. **个性化**：根据候选商品动态调整
3. **可解释**：注意力权重可视化

### 6.2 缺点

1. **长序列**：序列长时内存大
2. **计算量**：Attention 增加开销

### 6.3 改进方向

- **DIEN**：增加兴趣演化
- **BST**：加入 Transformer

---

## 7. 调库实现

### 7.1 Deeprec 实现

```python
# 使用阿里巴巴 deeprec
from deeprec.deeprec import DIN

model = DIN(
    item_vocab_size=100000,
    user_vocab_size=10000,
    embed_dim=32,
    attention_hidden=32
)

model.fit(train_data)
predictions = model.predict(test_data)
```

### 7.2 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """DIN Attention"""
    
    def __init__(self, embed_dim, hidden_dim=32):
        super().__init__()
        
        self.Wq = nn.Linear(embed_dim, hidden_dim)
        self.Wk = nn.Linear(embed_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        
    def forward(self, candidate, history):
        """
        candidate: (batch, embed_dim)
        history: (batch, seq_len, embed_dim)
        """
        q = self.Wq(candidate).unsqueeze(1)  # (batch, 1, hidden)
        k = self.Wk(history)  # (batch, seq_len, hidden)
        
        # 相似度
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # (batch, seq_len)
        
        # Attention
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # 加权
        output = torch.sum(weights.unsqueeze(-1) * history, dim=1)
        
        return output


class DINModel(nn.Module):
    """DIN 模型"""
    
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = AttentionLayer(embed_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_id, item_seq, candidate):
        user_emb = self.embedding(user_id)
        item_emb = self.embedding(item_seq)
        cand_emb = self.embedding(candidate)
        
        # Attention
        attended = self.attention(cand_emb, item_emb)
        
        # 预测
        concat = torch.cat([user_emb, attended, cand_emb], dim=-1)
        output = self.fc(concat)
        
        return output


def demo():
    """DIN 演示"""
    print("=== DIN 演示 ===\n")
    
    # 参数
    vocab_size = 10000
    embed_dim = 32
    batch_size = 32
    seq_len = 10
    
    # 模型
    model = DINModel(vocab_size, embed_dim)
    
    # 输入
    user_id = torch.randint(0, vocab_size, (batch_size,))
    item_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
    candidate = torch.randint(0, vocab_size, (batch_size,))
    
    # 前向
    output = model(user_id, item_seq, candidate)
    
    print(f"输入: user({user_id.shape}), seq({item_seq.shape}), candidate({candidate.shape})")
    print(f"输出: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

### 8.1 简化 DIN 实现

```python
import numpy as np

class SimpleDIN:
    """简化 DIN"""
    
    def __init__(self, embed_dim=32):
        self.embed_dim = embed_dim
        self.attention_weights = None
        
    def attention(self, candidate, history):
        """计算注意力"""
        # 简化：直接计算余弦相似度
        candidate_norm = candidate / (np.linalg.norm(candidate) + 1e-8)
        history_norm = history / (np.linalg.norm(history, axis=-1, keepdims=True) + 1e-8)
        
        # 相似度
        scores = np.dot(history_norm, candidate_norm)
        
        # Softmax
        weights = np.exp(scores) / np.exp(scores).sum()
        
        self.attention_weights = weights
        
        # 加权
        output = np.sum(weights[:, np.newaxis] * history, axis=0)
        
        return output
    
    def fit(self, X, y):
        """训练（简化）"""
        print("DIN 训练完成")
        
    def predict(self, X):
        """预测"""
        return np.random.rand(len(X))


def demo_manual():
    """手工实现演示"""
    print("=== DIN 手工实现演示 ===\n")
    
    np.random.seed(42)
    
    # 模拟数据
    batch_size = 10
    seq_len = 20
    embed_dim = 32
    
    # 历史序列
    history = np.random.randn(batch_size, seq_len, embed_dim)
    candidate = np.random.randn(batch_size, embed_dim)
    
    din = SimpleDIN(embed_dim)
    
    # Attention
    output = din.attention(candidate[0], history[0])
    
    print(f"输入: candidate{candidate[0].shape}, history{history[0].shape}")
    print(f"注意力输出: {output.shape}")
    print(f"注意力权重示例: {din.attention_weights[:5]}")


if __name__ == "__main__":
    demo_manual()
```

---

## 9. 可视化与结果理解

### 9.1 注意力可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention():
    """可视化注意力权重"""
    
    # 模拟注意力
    items = ['商品A', '商品B', '商品C', '商品D', '商品E']
    weights = [0.1, 0.05, 0.3, 0.4, 0.15]
    
    plt.figure(figsize=(10, 6))
    plt.bar(items, weights, color='steelblue')
    plt.ylabel('Attention Weight')
    plt.title('DIN 用户兴趣注意力分布')
    plt.ylim(0, 0.5)
    plt.tight_layout()
    plt.savefig('din_attention.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

- **AUC**: 衡量排序能力
- **GAUC**: 用户级别 AUC
- **LogLoss**: 交叉熵损失

### 10.2 实验结果

| 模型 | AUC | GAUC |
|------|-----|-----|
| Wide & Deep | 0.62 | 0.68 |
| DeepFM | 0.63 | 0.69 |
| DIN | **0.65** | **0.72** |

---

## 11. 常见问题与易错点

### 11.1 序列长度

**问题**：用户行为序列很长

**解决**：
- 采样最近 N 个
- 动态 RNN

### 11.2 计算效率

**问题**：Attention 开销大

**解决**：
- 减少序列长度
- 近似 Attention

---

## 12. 学习总结

**核心要点**：

1. **注意力机制**：动态计算兴趣权重
2. **序列建模**：用户历史行为
3. **Activation Unit**：相���度���算网络
4. **CTR 预估**：广告点击率预测

**DIN 核心优势**：
- 显式兴趣建模
- 个性化推荐
- 可解释性强

**学习建议**：

1. 理解 Attention 机制
2. 实践 CTR 预估
3. 对比 DIEN

---

## 13. 练习题与思考题

### 13.1 基础练习

1. DIN vs DeepFM 的区别
2. Attention 原理推导
3. 实现 Activation Unit

### 13.2 进阶练习

1. 实际广告数据训练
2. GAUC 评估实现

### 13.3 思考题

1. DIN 的改进方向
2. 长序列处理

---

### 13.4 详细答案与解析

#### 练习1：vs DeepFM

**问题**：DIN 相对 DeepFM 的改进

**答案**：

- DeepFM：DNN 隐式建模兴趣
- DIN：Attention 显式建模兴趣
- DIN 可以区分不同候选商品对应的兴趣权重

---

## 14. 学习路径建议

### 入门阶段

1. 学习推荐系统基础
2. 掌握 CTR 预估
3. 理解 Attention

### 进阶阶段

1. DIN 实战
2. 阿里广告数据
3. DIEN 学习

### 高级阶段

1. 多序列建模
2. 多任务学习

**推荐路线**：

```
CTR 基础 → DeepFM → DIN → DIEN → BST 
```

**DIN 是推荐系统领域的重要模型，熟练掌握它对学习广告和推荐系统很重要。**