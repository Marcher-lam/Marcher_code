# CTR 预估基础 学习文档

## 1. 基础认知

### 1.1 什么是 CTR 预估？

CTR（Click-Through Rate，点击率）预估是预测用户在看到某个广告/推荐内容后点击的概率。

$$CTR = \frac{点击次数}{展示次数}$$

**CTR 预估的目标**：给定用户、物品、上下文等特征，预测用户点击该物品的概率：

$$P(click = 1 | user, item, context)$$

### 1.2 为什么 CTR 预估重要？

在推荐系统和广告系统中，CTR 预估是核心环节：

```
用户请求 → 召回 → 排序（CTR预估） → 重排 → 展示
```

**应用场景：**

| 场景 | 说明 |
|------|------|
| 广告投放 | 预估点击率决定广告排序和出价 |
| 信息流推荐 | 预估用户对内容的兴趣 |
| 搜索排序 | 结合相关性和点击概率 |
| 电商推荐 | 预估购买/点击概率 |

**商业价值：**

$$eCPM = CTR \times CPC \times 1000$$

- eCPM（千次展示期望收益）= CTR × 单次点击价格 × 1000
- 准确的 CTR 预估直接影响平台收益

### 1.3 CTR 预估 vs 传统推荐

| 维度 | 传统推荐 | CTR 预估 |
|------|----------|----------|
| 目标 | 排序、推荐列表 | 预测点击概率 |
| 输出 | 分数（可排序即可） | 概率（需要校准） |
| 评估 | NDCG、MAP、Recall | AUC、LogLoss、Calibration |
| 特征 | 主要是 ID 特征 | 大量稠密特征 |
| 模型 | 协同过滤、矩阵分解 | LR、FM、DeepFM |

### 1.4 CTR 预估的发展历程

```
时间线：
2000s ────────┬────────────────┬────────────────┬────────────────┬──────►
              │                │                │                │
           LR + 特征工程     FM (2010)      Wide&Deep       DeepFM
           (逻辑回归)       (因子分解机)    (2016)          (2017)
                              │
                          FFM (2015)
                              │
                          Deep Learning Era
                          DIN/DIEN/DCN/...
```

## 2. 核心概念

### 2.1 问题定义

**输入：**
- 用户特征：ID、年龄、性别、历史行为等
- 物品特征：ID、类别、价格、热度等
- 上下文特征：时间、位置、设备等
- 交叉特征：用户-物品的交互特征

**输出：**
- 点击概率 $\hat{y} \in [0, 1]$

**损失函数：**
$$L = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)$$

这是二分类交叉熵损失（Binary Cross Entropy）。

### 2.2 特征工程

CTR 预估中的特征通常分为几类：

#### 2.2.1 类别特征（Categorical Features）

```python
# 类别特征示例
user_id = "user_12345"
item_category = "electronics"
device_type = "mobile"
time_period = "morning"
```

**处理方法：**
- One-Hot 编码
- Embedding 编码
- 特征哈希

#### 2.2.2 数值特征（Numerical Features）

```python
# 数值特征示例
user_age = 25
item_price = 99.99
user_history_clicks = 150
item_popularity = 0.85
```

**处理方法：**
- 归一化（Normalization）
- 分桶离散化（Binning）
- 对数变换

#### 2.2.3 交叉特征（Cross Features）

```python
# 交叉特征示例
user_item_category = "user_12345 AND electronics"
age_device = "25 AND mobile"
```

**为什么需要交叉特征？**
- 单独的用户特征和物品特征不足以表达用户对物品的偏好
- 例如：年轻人可能更喜欢游戏类内容

### 2.3 评估指标

#### 2.3.1 AUC（Area Under ROC Curve）

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_true, y_pred)
```

**AUC 的含义：**
- 随机抽取一个正样本和一个负样本，正样本预测分数高于负样本的概率
- AUC = 0.5：随机猜测
- AUC = 1.0：完美预测
- AUC > 0.7：可用
- AUC > 0.8：良好

#### 2.3.2 LogLoss（交叉熵损失）

```python
from sklearn.metrics import log_loss

loss = log_loss(y_true, y_pred)
```

**LogLoss 的含义：**
- 衡量预测概率与真实标签的差异
- 越小越好
- 对概率校准敏感

#### 2.3.3 Calibration（校准度）

$$Calibration = \frac{实际点击数}{预测点击数}$$

- Calibration ≈ 1：预测概率准确
- Calibration > 1：低估
- Calibration < 1：高估

#### 2.3.4 GAUC（Group AUC）

按用户分组计算 AUC，然后加权平均：

$$GAUC = \frac{\sum_{u} w_u \times AUC_u}{\sum_{u} w_u}$$

### 2.4 正负样本

**正样本（Positive）：** 用户点击的样本
**负样本（Negative）：** 用户未点击的样本

**样本不平衡问题：**
- 实际 CTR 通常很低（1%-5%）
- 正负样本比例严重不平衡
- 需要采样策略或加权处理

**负采样策略：**
```python
# 随机负采样
def random_negative_sampling(positive_items, all_items, ratio=4):
    """
    为每个正样本采样 ratio 个负样本
    """
    negatives = []
    for pos_item in positive_items:
        # 随机采样不在正样本中的物品
        for _ in range(ratio):
            neg = random.choice(all_items)
            while neg in positive_items:
                neg = random.choice(all_items)
            negatives.append(neg)
    return negatives
```

## 3. 基线模型：逻辑回归

### 3.1 逻辑回归模型

逻辑回归是 CTR 预估最经典的基线模型：

$$\hat{y} = \sigma(w_0 + \sum_{i=1}^{n} w_i x_i)$$

其中 $\sigma(x) = \frac{1}{1+e^{-x}}$ 是 Sigmoid 函数。

### 3.2 逻辑回归的实现

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 示例数据
data = {
    'user_id': ['u1', 'u2', 'u3', 'u1', 'u2'],
    'item_id': ['i1', 'i1', 'i2', 'i2', 'i3'],
    'user_age': [25, 30, 35, 25, 30],
    'item_price': [100, 100, 200, 200, 150],
    'click': [1, 0, 1, 0, 1]
}

# 特征处理
categorical_features = ['user_id', 'item_id']
numerical_features = ['user_age', 'item_price']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# 构建 Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 训练
X = data[['user_id', 'item_id', 'user_age', 'item_price']]
y = data['click']
model.fit(X, y)

# 预测
predictions = model.predict_proba(X)[:, 1]
print("CTR 预测:", predictions)
```

### 3.3 逻辑回归的局限

1. **无法学习特征交叉**：需要手工构造交叉特征
2. **特征工程重**：需要大量人工设计特征
3. **泛化能力弱**：对未见过的特征组合表现差

## 4. 特征交叉的重要性

### 4.1 为什么需要特征交叉？

**示例：**
- 用户 A：喜欢"科技"类别
- 物品 B：属于"科技"类别，价格"高"

单独看：
- 用户喜欢科技：权重 +0.3
- 物品价格高：权重 -0.2

组合看：
- 用户 A 对"科技+高价"可能有特殊偏好
- 这个组合效应无法由单独特征表达

### 4.2 手工特征交叉

```python
def create_cross_features(df, feature_pairs):
    """
    创建交叉特征

    参数:
        df: 数据框
        feature_pairs: 要交叉的特征对列表
    """
    for f1, f2 in feature_pairs:
        cross_name = f"{f1}_{f2}"
        df[cross_name] = df[f1].astype(str) + "_" + df[f2].astype(str)
    return df

# 示例
df = pd.DataFrame({
    'user_age_bucket': ['young', 'middle', 'young'],
    'item_category': ['tech', 'fashion', 'fashion'],
    'click': [1, 0, 1]
})

# 创建交叉特征
df = create_cross_features(df, [('user_age_bucket', 'item_category')])
print(df)
# 输出包含新列: user_age_bucket_item_category
# 'young_tech', 'middle_fashion', 'young_fashion'
```

**问题：**
- 组合爆炸：n 个特征有 $C(n,2)$ 种两两组合
- 稀疏性：很多组合在训练数据中出现很少
- 泛化差：未见过的组合无法预测

## 5. 从 LR 到 FM

### 5.1 LR 的特征组合问题

LR 模型：
$$\hat{y} = \sigma(w_0 + \sum_{i=1}^{n} w_i x_i)$$

添加二阶交叉：
$$\hat{y} = \sigma(w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} w_{ij} x_i x_j)$$

**问题：** $w_{ij}$ 参数量 $O(n^2)$，且很多 $x_i x_j$ 在训练数据中从未出现。

### 5.2 FM 的解决方案

FM（Factorization Machines）用隐向量内积代替独立参数：

$$\hat{y} = \sigma(w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j)$$

其中 $v_i$ 是特征 i 的 k 维隐向量。

**优势：**
- 参数量从 $O(n^2)$ 降到 $O(nk)$
- 可以泛化到未见过的特征组合

### 5.3 对比

| 模型 | 参数量 | 特征交叉 | 泛化能力 |
|------|--------|----------|----------|
| LR | O(n) | 无 | 弱 |
| LR+Poly2 | O(n²) | 有 | 强 |
| FM | O(nk) | 有 | 强 |

## 6. 深度学习时代

### 6.1 Embedding + MLP

深度学习方法的核心结构：

```
输入特征 → Embedding层 → 拼接/交互 → MLP → 输出
```

```python
import torch
import torch.nn as nn

class DeepCTR(nn.Module):
    def __init__(self, field_dims, embed_dim, hidden_units):
        super().__init__()

        # Embedding 层
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])

        # MLP 层
        input_dim = len(field_dims) * embed_dim
        layers = []
        for hidden in hidden_units:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.Dropout(0.2))
            input_dim = hidden

        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, num_fields)
        embeddings = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        concat = torch.cat(embeddings, dim=1)  # (batch, num_fields * embed_dim)
        output = self.mlp(concat)
        return output.squeeze()
```

### 6.2 主要模型架构

```
模型演进：

Wide&Deep (2016)
├── Wide: 线性模型（记忆）
└── Deep: MLP（泛化）

DeepFM (2017)
├── FM: 二阶交叉
└── Deep: MLP

DCN (2017)
├── Cross Network: 显式交叉
└── Deep Network: MLP

DIN (2018)
├── Attention: 用户行为加权
└── Deep: MLP

DIEN (2019)
├── GRU: 兴趣演化
└── Attention: 兴趣提取
```

## 7. 工程实践要点

### 7.1 数据处理

```python
import pandas as pd
import numpy as np

class CTRDataProcessor:
    """CTR 数据处理器"""

    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}

    def process_categorical(self, df, column, min_freq=10):
        """处理类别特征"""
        # 低频类别归为 'others'
        value_counts = df[column].value_counts()
        rare_values = value_counts[value_counts < min_freq].index
        df[column] = df[column].replace(rare_values, 'others')

        # Label Encoding
        from sklearn.preprocessing import LabelEncoder
        if column not in self.label_encoders:
            self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column])
        else:
            df[column] = self.label_encoders[column].transform(df[column])

        return df

    def process_numerical(self, df, column, method='normalize'):
        """处理数值特征"""
        if method == 'normalize':
            from sklearn.preprocessing import StandardScaler
            if column not in self.scalers:
                self.scalers[column] = StandardScaler()
                df[column] = self.scalers[column].fit_transform(
                    df[[column]]
                ).flatten()
            else:
                df[column] = self.scalers[column].transform(
                    df[[column]]
                ).flatten()
        elif method == 'bucket':
            # 分桶
            df[column] = pd.qcut(df[column], q=10, labels=False, duplicates='drop')

        return df

    def process(self, df, cat_cols, num_cols):
        """完整处理流程"""
        for col in cat_cols:
            df = self.process_categorical(df, col)

        for col in num_cols:
            df = self.process_numerical(df, col)

        return df
```

### 7.2 在线服务

```python
class CTRModelServer:
    """CTR 模型在线服务"""

    def __init__(self, model, feature_processor):
        self.model = model
        self.feature_processor = feature_processor

    def predict(self, user_features, item_features, context_features):
        """
        在线预测 CTR

        参数:
            user_features: 用户特征字典
            item_features: 物品特征字典
            context_features: 上下文特征字典
        """
        # 特征拼接
        features = {**user_features, **item_features, **context_features}

        # 特征处理
        processed = self.feature_processor.process_single(features)

        # 模型预测
        with torch.no_grad():
            ctr = self.model(processed)

        return ctr.item()

    def batch_predict(self, user_features, item_list, context_features):
        """
        批量预测（用于排序）
        """
        predictions = []
        for item_features in item_list:
            ctr = self.predict(user_features, item_features, context_features)
            predictions.append(ctr)
        return predictions
```

### 7.3 A/B 测试

```python
def ab_test_analysis(control_data, treatment_data):
    """
    A/B 测试分析

    参数:
        control_data: 对照组数据 {'impressions': n, 'clicks': m}
        treatment_data: 实验组数据
    """
    # 计算 CTR
    control_ctr = control_data['clicks'] / control_data['impressions']
    treatment_ctr = treatment_data['clicks'] / treatment_data['impressions']

    # 计算提升
    lift = (treatment_ctr - control_ctr) / control_ctr * 100

    # 统计显著性检验
    from scipy import stats
    contingency = [
        [control_data['clicks'], control_data['impressions'] - control_data['clicks']],
        [treatment_data['clicks'], treatment_data['impressions'] - treatment_data['clicks']]
    ]
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    return {
        'control_ctr': control_ctr,
        'treatment_ctr': treatment_ctr,
        'lift': lift,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

## 8. 模型选择指南

### 8.1 模型对比

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| LR | 简单快速 | 基线、低延迟 |
| FM | 二阶交叉 | 中等规模、稀疏特征 |
| DeepFM | FM+Deep | 大规模、工业场景 |
| Wide&Deep | 记忆+泛化 | Google 风格 |
| DIN | 行为序列 | 用户行为丰富 |
| DCN | 显式交叉 | 高阶特征交叉 |

### 8.2 选择建议

```
数据规模小（<100万样本）
└── LR / FM

数据规模中（100万-1亿样本）
├── 特征稀疏 → FM
└── 特征稠密 → LR

数据规模大（>1亿样本）
├── 用户行为丰富 → DIN / DIEN
├── 需要高阶交叉 → DCN / DCNv2
└── 通用场景 → DeepFM / Wide&Deep

延迟要求极高（<10ms）
└── LR / 小规模 FM
```

## 9. 评估与监控

### 9.1 离线评估

```python
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
import matplotlib.pyplot as plt

def evaluate_ctr_model(y_true, y_pred):
    """全面评估 CTR 模型"""
    metrics = {}

    # AUC
    metrics['auc'] = roc_auc_score(y_true, y_pred)

    # LogLoss
    metrics['logloss'] = log_loss(y_true, y_pred)

    # Calibration
    metrics['calibration'] = sum(y_pred) / sum(y_true)

    # 分桶校准
    metrics['bucket_calibration'] = bucket_calibration(y_true, y_pred)

    return metrics

def bucket_calibration(y_true, y_pred, n_buckets=10):
    """分桶校准分析"""
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df['bucket'] = pd.qcut(df['pred'], q=n_buckets, labels=False)

    results = []
    for bucket in range(n_buckets):
        bucket_df = df[df['bucket'] == bucket]
        actual_ctr = bucket_df['true'].mean()
        predicted_ctr = bucket_df['pred'].mean()
        results.append({
            'bucket': bucket,
            'actual_ctr': actual_ctr,
            'predicted_ctr': predicted_ctr,
            'calibration': actual_ctr / predicted_ctr if predicted_ctr > 0 else 0
        })

    return pd.DataFrame(results)
```

### 9.2 在线监控

```python
class CTRMonitor:
    """CTR 在线监控"""

    def __init__(self, window_size=10000):
        self.window_size = window_size
        self.predictions = []
        self.actuals = []

    def log(self, prediction, actual):
        """记录预测和实际结果"""
        self.predictions.append(prediction)
        self.actuals.append(actual)

        # 保持窗口大小
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.actuals.pop(0)

    def get_metrics(self):
        """获取当前指标"""
        if not self.predictions:
            return None

        return {
            'auc': roc_auc_score(self.actuals, self.predictions),
            'actual_ctr': np.mean(self.actuals),
            'predicted_ctr': np.mean(self.predictions),
            'calibration': np.mean(self.actuals) / np.mean(self.predictions)
        }

    def check_anomaly(self):
        """检查异常"""
        metrics = self.get_metrics()
        if not metrics:
            return False, {}

        anomalies = []

        # AUC 下降
        if metrics['auc'] < 0.65:
            anomalies.append(f"AUC 异常低: {metrics['auc']:.4f}")

        # 校准偏差
        if abs(metrics['calibration'] - 1) > 0.2:
            anomalies.append(f"校准偏差大: {metrics['calibration']:.4f}")

        return len(anomalies) > 0, anomalies
```

## 10. 常见问题与易错点

### 10.1 常见问题

**Q1：CTR 预估输出必须是真实概率吗？**

A：是的。CTR 预估的输出应该是校准后的概率，用于：
- 广告出价计算
- 收益预估
- 流量分配

**Q2：如何处理样本不平衡？**

A：
- 下采样负样本
- 调整损失函数权重
- 使用 Focal Loss

**Q3：如何评估模型效果？**

A：
- 离线：AUC、LogLoss、Calibration
- 在线：实际 CTR 提升业务指标

### 10.2 易错点

1. **预测值未校准**：模型输出的概率与实际 CTR 不一致
2. **特征穿越**：使用了未来信息
3. **数据泄露**：训练集和测试集重叠
4. **评估不当**：用训练集评估或 AUC 计算错误

## 11. 学习总结

### 11.1 核心要点

1. **CTR 预估是二分类问题**：预测点击概率
2. **特征工程是关键**：类别特征、数值特征、交叉特征
3. **评估指标特殊**：AUC、LogLoss、Calibration
4. **从 LR 到深度学习**：特征交叉的自动学习

### 11.2 知识图谱

```
CTR 预估
├── 基础概念
│   ├── 点击率
│   ├── 特征工程
│   └── 评估指标
├── 模型演进
│   ├── LR
│   ├── FM
│   ├── DeepFM
│   └── DIN/DIEN/...
├── 工程实践
│   ├── 数据处理
│   ├── 在线服务
│   └── A/B 测试
└── 监控运维
    ├── 离线评估
    └── 在线监控
```

## 12. 练习题

### 12.1 基础题

1. CTR 的计算公式是什么？

2. AUC = 0.8 表示什么含义？

3. 为什么 CTR 预估需要校准？

### 12.2 进阶题

4. 实现一个简单的 LR CTR 预估模型。

5. 比较不同负采样比例对模型效果的影响。

### 12.3 思考题

6. 为什么 FM 比手工特征交叉更有效？

7. 如何设计一个实时 CTR 预估系统？

## 13. 学习路径建议

### 13.1 学习顺序

1. **理解问题** → CTR 预估的定义和重要性
2. **掌握评估** → AUC、LogLoss、Calibration
3. **学习基线** → 逻辑回归模型
4. **理解交叉** → 特征交叉的重要性
5. **学习 FM** → 因子分解机
6. **深度学习** → DeepFM、Wide&Deep 等

### 13.2 推荐资源

- **论文**：Deep Neural Networks for YouTube Recommendations
- **论文**：Wide & Deep Learning for Recommender Systems
- **书籍**：《深度学习推荐系统》- 王喆
- **竞赛**：Kaggle CTR 预估竞赛

### 13.3 下一步学习

- **FM**：因子分解机详解
- **DeepFM**：FM 与深度学习的结合
- **Wide&Deep**：Google 的经典架构
- **DIN**：注意力机制在推荐中的应用
