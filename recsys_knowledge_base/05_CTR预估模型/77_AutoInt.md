# AutoInt 学习文档

## 1. 算法基础认知

### 1.1 什么是 AutoInt？

AutoInt（Automatic Feature Interaction Learning）是 2019 年提出的 CTR 预估模型，使用 **多头自注意力机制** 来自动学习特征的高阶交叉。

### 1.2 核心思想

- **自动交叉学习**：不需要人工设计交叉特征
- **注意力机制**：使用 Transformer 的自注意力学习特征重要性
- **可解释性**：注意力权重可以解释特征交叉的重要性

### 1.3 与其他模型的关系

```
FM (二阶) → DeepFM (二阶+高阶隐式) → DCN (有界阶显式)
                                        ↓
                              AutoInt (注意力学习高阶)
```

## 2. 模型架构

### 2.1 整体结构

```
输入特征
    ↓
Embedding Layer
    ↓
┌──────────────────────────┐
│   Multi-Head Self-Attention  │ ← 多层
│   - Query, Key, Value     │
│   - Residual Connection   │
│   - Layer Normalization   │
└──────────────────────────┘
    ↓
Interaction Layer (可选)
    ↓
Output Layer
```

### 2.2 自注意力交叉

自注意力计算：

$$\alpha_{ij}^{(h)} = \frac{\exp(\phi(Q_i^{(h)}, K_j^{(h)}))}{\sum_{l=1}^{m} \exp(\phi(Q_i^{(h)}, K_l^{(h)}))}$$

$$\tilde{e}_i^{(h)} = \sum_{j=1}^{m} \alpha_{ij}^{(h)} V_j^{(h)}$$

其中 $\phi$ 是注意力函数（如点积）。

## 3. PyTorch 完整实现

### 3.1 AutoInt 模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力层
    用于学习特征之间的交互
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        """
        参数:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout 比例
        """
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V 投影
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.output = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        """
        前向传播

        参数:
            x: (batch, field_num, embed_dim)
            mask: (batch, field_num) 可选的 mask

        返回:
            output: (batch, field_num, embed_dim)
            attention_weights: (batch, num_heads, field_num, field_num)
        """
        batch_size, field_num, _ = x.shape

        # Q, K, V 投影
        Q = self.query(x).view(batch_size, field_num, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, field_num, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, field_num, self.num_heads, self.head_dim)

        # 转置为 (batch, num_heads, field_num, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 注意力分数: (batch, num_heads, field_num, field_num)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用 mask
        if mask is not None:
            # mask: (batch, field_num) -> (batch, 1, 1, field_num)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        # (batch, num_heads, field_num, head_dim)
        context = torch.matmul(attention_weights, V)

        # 重塑回去
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, field_num, self.embed_dim)

        # 输出投影
        output = self.output(context)

        return output, attention_weights


class AutoIntLayer(nn.Module):
    """
    AutoInt 的交互层

    包含:
    - Multi-Head Self-Attention
    - Residual Connection
    - Layer Normalization
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, use_residual=True):
        super().__init__()

        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.use_residual = use_residual
        if use_residual:
            self.residual = nn.Linear(embed_dim, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播

        参数:
            x: (batch, field_num, embed_dim)
            mask: 可选的 mask

        返回:
            output: (batch, field_num, embed_dim)
            attention_weights: 注意力权重
        """
        # Self-Attention
        attn_output, attention_weights = self.attention(x, mask)

        # Residual Connection
        if self.use_residual:
            residual = self.residual(x)
            output = residual + attn_output
        else:
            output = attn_output

        # Dropout + Layer Norm
        output = self.dropout(output)
        output = self.layer_norm(output)

        return output, attention_weights


class AutoInt(nn.Module):
    """
    AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks

    论文: AutoInt: Automatic Feature Interaction Learning via
          Self-Attentive Neural Networks for CTR Prediction (CIKM 2019)
    """

    def __init__(self, feature_configs, embed_dim=16,
                 num_heads=8, num_layers=3, dropout=0.1,
                 use_residual=True, use_linear=True):
        """
        参数:
            feature_configs: dict, 特征配置
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: AutoInt 层数
            dropout: Dropout 比例
            use_residual: 是否使用残差连接
            use_linear: 是否使用线性部分
        """
        super().__init__()

        self.feature_configs = feature_configs
        self.embed_dim = embed_dim
        self.use_linear = use_linear

        # ========== Embedding 层 ==========
        self.embeddings = nn.ModuleDict()
        self.field_num = 0

        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                self.embeddings[name] = nn.Embedding(
                    config['vocab_size'],
                    config.get('embed_dim', embed_dim)
                )
                self.field_num += 1

        # 数值特征处理
        self.numerical_proj = nn.ModuleDict()
        for name, config in feature_configs.items():
            if config['type'] == 'numerical':
                self.numerical_proj[name] = nn.Linear(1, embed_dim)
                self.field_num += 1

        # ========== AutoInt 层 ==========
        self.autoint_layers = nn.ModuleList([
            AutoIntLayer(embed_dim, num_heads, dropout, use_residual)
            for _ in range(num_layers)
        ])

        # ========== Linear 部分 ==========
        if use_linear:
            # 计算输入维度
            self._compute_input_dim(feature_configs, embed_dim)
            self.linear = nn.Linear(self.input_dim, 1)

        # ========== 输出层 ==========
        # 输出维度 = field_num * embed_dim
        self.output_layer = nn.Linear(self.field_num * embed_dim, 1)

        # 保存注意力权重（用于可解释性）
        self.attention_weights = None

    def _compute_input_dim(self, feature_configs, embed_dim):
        """计算输入维度"""
        dim = 0
        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                dim += config.get('embed_dim', embed_dim)
            elif config['type'] == 'numerical':
                dim += 1
        self.input_dim = dim

    def forward(self, features):
        """
        前向传播

        参数:
            features: dict, 特征字典

        返回:
            output: (batch, 1) CTR 预测
        """
        # ========== Embedding ==========
        embeddings_list = []
        linear_features = []

        for name, config in self.feature_configs.items():
            if config['type'] == 'categorical':
                emb = self.embeddings[name](features[name])
                if len(emb.shape) == 3:
                    emb = emb.squeeze(1)
                embeddings_list.append(emb)
                linear_features.append(emb)
            elif config['type'] == 'numerical':
                val = features[name]
                if len(val.shape) == 1:
                    val = val.unsqueeze(-1)
                # 投影到 embed_dim
                proj = self.numerical_proj[name](val)
                embeddings_list.append(proj)
                linear_features.append(val)

        # 拼接为嵌入矩阵
        embed_matrix = torch.stack(embeddings_list, dim=1)  # (batch, field_num, embed_dim)

        # ========== AutoInt 层 ==========
        x = embed_matrix
        all_attention_weights = []

        for layer in self.autoint_layers:
            x, attention_weights = layer(x)
            all_attention_weights.append(attention_weights)

        # 保存注意力权重
        self.attention_weights = all_attention_weights

        # ========== 输出 ==========
        logits = []

        # Linear 部分
        if self.use_linear:
            linear_input = torch.cat(linear_features, dim=-1)
            linear_logit = self.linear(linear_input)
            logits.append(linear_logit)

        # AutoInt 输出
        # 展平并输出
        x_flat = x.view(x.size(0), -1)  # (batch, field_num * embed_dim)
        autoint_logit = self.output_layer(x_flat)
        logits.append(autoint_logit)

        # 求和
        logit = sum(logits)

        # Sigmoid
        output = torch.sigmoid(logit)

        return output

    def get_attention_weights(self):
        """获取注意力权重（用于可解释性）"""
        return self.attention_weights

    def get_feature_importance(self, features, feature_names=None):
        """
        获取特征重要性

        参数:
            features: 输入特征
            feature_names: 特征名称列表

        返回:
            importance: 特征重要性字典
        """
        with torch.no_grad():
            # 前向传播
            _ = self.forward(features)

            # 获取最后一层的注意力权重
            last_attention = self.attention_weights[-1]  # (batch, num_heads, field_num, field_num)

            # 平均所有头
            avg_attention = last_attention.mean(dim=1)  # (batch, field_num, field_num)

            # 平均每个特征被关注的程度
            importance = avg_attention.mean(dim=1).mean(dim=0)  # (field_num,)

            # 归一化
            importance = importance / importance.sum()

            if feature_names is None:
                feature_names = list(self.feature_configs.keys())

            return {name: float(imp) for name, imp in zip(feature_names, importance)}


class AutoIntPlus(nn.Module):
    """
    AutoInt+: 结合 DNN 的版本
    """

    def __init__(self, feature_configs, embed_dim=16,
                 num_heads=8, num_layers=3, dropout=0.1,
                 dnn_hidden_units=[256, 128]):
        super().__init__()

        self.feature_configs = feature_configs
        self.embed_dim = embed_dim

        # Embedding 层
        self.embeddings = nn.ModuleDict()
        self.field_num = 0

        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                self.embeddings[name] = nn.Embedding(config['vocab_size'], embed_dim)
                self.field_num += 1

        self.numerical_proj = nn.ModuleDict()
        for name, config in feature_configs.items():
            if config['type'] == 'numerical':
                self.numerical_proj[name] = nn.Linear(1, embed_dim)
                self.field_num += 1

        # AutoInt 层
        self.autoint_layers = nn.ModuleList([
            AutoIntLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # DNN 层
        self._compute_input_dim(feature_configs, embed_dim)
        dnn_layers = []
        input_dim = self.input_dim
        for hidden in dnn_hidden_units:
            dnn_layers.append(nn.Linear(input_dim, hidden))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout))
            input_dim = hidden
        self.dnn = nn.Sequential(*dnn_layers)

        # 输出层
        self.autoint_output = nn.Linear(self.field_num * embed_dim, 1)
        self.dnn_output = nn.Linear(dnn_hidden_units[-1], 1)

    def _compute_input_dim(self, feature_configs, embed_dim):
        dim = 0
        for name, config in feature_configs.items():
            if config['type'] == 'categorical':
                dim += embed_dim
            elif config['type'] == 'numerical':
                dim += 1
        self.input_dim = dim

    def forward(self, features):
        # Embedding
        embeddings_list = []
        all_features = []

        for name, config in self.feature_configs.items():
            if config['type'] == 'categorical':
                emb = self.embeddings[name](features[name])
                if len(emb.shape) == 3:
                    emb = emb.squeeze(1)
                embeddings_list.append(emb)
                all_features.append(emb)
            elif config['type'] == 'numerical':
                val = features[name]
                if len(val.shape) == 1:
                    val = val.unsqueeze(-1)
                proj = self.numerical_proj[name](val)
                embeddings_list.append(proj)
                all_features.append(val)

        embed_matrix = torch.stack(embeddings_list, dim=1)

        # AutoInt
        x = embed_matrix
        for layer in self.autoint_layers:
            x, _ = layer(x)

        x_flat = x.view(x.size(0), -1)
        autoint_logit = self.autoint_output(x_flat)

        # DNN
        dnn_input = torch.cat(all_features, dim=-1)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_output(dnn_output)

        # 组合
        logit = autoint_logit + dnn_logit
        output = torch.sigmoid(logit)

        return output


# 使用示例
if __name__ == "__main__":
    # 特征配置
    feature_configs = {
        'cat1': {'type': 'categorical', 'vocab_size': 100},
        'cat2': {'type': 'categorical', 'vocab_size': 200},
        'cat3': {'type': 'categorical', 'vocab_size': 150},
        'num1': {'type': 'numerical'},
    }

    # 创建模型
    model = AutoInt(
        feature_configs=feature_configs,
        embed_dim=16,
        num_heads=4,
        num_layers=3,
        dropout=0.1
    )

    # 模拟输入
    batch_size = 32
    features = {
        'cat1': torch.randint(0, 100, (batch_size,)),
        'cat2': torch.randint(0, 200, (batch_size,)),
        'cat3': torch.randint(0, 150, (batch_size,)),
        'num1': torch.randn(batch_size),
    }

    # 前向传播
    output = model(features)
    print(f"输出形状: {output.shape}")

    # 获取特征重要性
    importance = model.get_feature_importance(features, list(feature_configs.keys()))
    print(f"特征重要性: {importance}")

    # 查看注意力权重
    attn_weights = model.get_attention_weights()
    print(f"注意力权重层数: {len(attn_weights)}")
    print(f"每层形状: {attn_weights[0].shape}")
```

## 4. 可解释性

### 4.1 注意力可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_attention(model, features, feature_names, layer_idx=0, head_idx=0):
    """
    可视化注意力权重

    参数:
        model: AutoInt 模型
        features: 输入特征
        feature_names: 特征名称
        layer_idx: 要可视化的层索引
        head_idx: 要可视化的头索引
    """
    # 前向传播
    with torch.no_grad():
        _ = model.forward(features)

    # 获取注意力权重
    attention = model.attention_weights[layer_idx][0, head_idx].cpu().numpy()

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention,
        xticklabels=feature_names,
        yticklabels=feature_names,
        annot=True,
        fmt='.2f',
        cmap='Blues'
    )
    plt.title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
    plt.xlabel('Key Features')
    plt.ylabel('Query Features')
    plt.tight_layout()
    plt.show()


def explain_prediction(model, features, feature_names):
    """
    解释预测结果

    参数:
        model: AutoInt 模型
        features: 输入特征
        feature_names: 特征名称

    返回:
        explanation: 解释字典
    """
    with torch.no_grad():
        # 获取预测
        prediction = model(features)

        # 获取特征重要性
        importance = model.get_feature_importance(features, feature_names)

        # 获取每层的注意力摘要
        layer_summaries = []
        for layer_idx, attn in enumerate(model.attention_weights):
            # 平均所有头和样本
            avg_attn = attn.mean(dim=(0, 1)).cpu().numpy()
            layer_summaries.append({
                'layer': layer_idx,
                'attention_matrix': avg_attn,
                'top_interactions': get_top_interactions(avg_attn, feature_names, top_k=3)
            })

    return {
        'prediction': float(prediction.mean()),
        'feature_importance': importance,
        'layer_summaries': layer_summaries
    }


def get_top_interactions(attention_matrix, feature_names, top_k=5):
    """获取 top-k 重要的特征交互"""
    interactions = []
    n = len(feature_names)

    for i in range(n):
        for j in range(n):
            if i != j:
                interactions.append({
                    'from': feature_names[i],
                    'to': feature_names[j],
                    'weight': attention_matrix[i, j]
                })

    # 排序
    interactions = sorted(interactions, key=lambda x: x['weight'], reverse=True)
    return interactions[:top_k]
```

## 5. 模型对比

### 5.1 与其他注意力模型对比

| 模型 | 注意力类型 | 可解释性 | 计算复杂度 |
|------|-----------|----------|-----------|
| DIN | Target Attention | 中 | O(n) |
| AutoInt | Self-Attention | 高 | O(n²) |
| Transformer | Self-Attention | 高 | O(n²) |

### 5.2 适用场景

**适合 AutoInt：**
- 特征数量适中（< 100）
- 需要可解释性
- 特征交叉复杂

**不太适合：**
- 特征数量很大
- 对实时性要求极高

## 6. 调参建议

### 6.1 模型参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| embed_dim | 16-64 | 嵌入维度 |
| num_heads | 2-8 | 注意力头数 |
| num_layers | 2-4 | 层数不宜太深 |
| dropout | 0.1-0.3 | 防止过拟合 |

### 6.2 训练参数

| 参数 | 推荐值 |
|------|--------|
| learning_rate | 1e-4 |
| batch_size | 256-1024 |
| weight_decay | 1e-4 |

## 7. 学习总结

### 7.1 核心要点

1. **自注意力学习交叉**：自动学习特征交互
2. **可解释性**：注意力权重可解释
3. **多层堆叠**：学习高阶交叉

### 7.2 关键优势

- 自动学习特征交叉，无需人工设计
- 注意力权重提供可解释性
- Transformer 结构成熟稳定

## 8. 练习题

1. 实现 AutoInt 的多头注意力可视化。

2. 比较不同层数对模型效果的影响。

3. 设计一个结合 AutoInt 和 DCN 的混合模型。
