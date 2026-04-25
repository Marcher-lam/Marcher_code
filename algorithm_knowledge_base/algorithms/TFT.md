# TFT (Temporal Fusion Transformer) 学习文档

> Google提出的可解释时间序列预测模型，融合RNN与Transformer的优势

---

## 1. 算法基础认知

### 1.1 一句话定义

TFT（Temporal Fusion Transformer）是由Google于2020年提出的时间序列预测模型，结合了RNN（处理序列）和Transformer（注意力）的优势，同时具有**可解释性**，是多步时间序列预测的SOTA模型之一。

### 1.2 直觉类比

TFT就像一个"时间序列全能分析师"！它有三大法宝：
1. **记忆能力**：像LSTM一样记住过去发生了什么
2. **全局视野**：像Transformer一样理解不同时刻的关联
3. **解释能力**：还能告诉你"为什么"这么预测！

想象你在预测明天的销量：
- 普通RNN：只看最近几天的数据
- Transformer：能关注到一个月前的促销活动
- TFT：不仅能预测，还能告诉你"上周的促销对今天影响最大"

### 1.3 发展背景

- 2020年，Google的Lim等人在论文"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"中提出
- 在多个基准数据集上刷新SOTA
- 成为pytorch-forecasting库的标配模型

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 时间序列 → 多步预测 |
| 输出 | 多步预测+注意力权重+特征重要性 |
| 模型类型 | Transformer + RNN混合 |
| 核心特点 | 可解释的多步预测 |

### 1.5 前置知识

- [必备]：时间序列基础
- [必备]：LSTM/GRU
- [推荐]：Transformer
- [可选]：注意力机制

---

## 2. 核心原理

### 2.1 整体架构

```
┌─────────────────────────────────────────┐
│           Embedding Layers              │
│  (线性嵌入 + 时间位置 + 已知 + 未知 + 静态)│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      Gated Residual Network (GRN)        │
│        (特征提取 + 门控残差连接)          │
└──────────────────┬──────────────────────┘
                   │
    ┌───────────────┴───────────────┐
    │                             │
    ▼                             ▼
┌───────────────────┐    ┌───────────────────┐
│  Variable Selection│    │  Interpretable   │
│     Network       │    │    Modules        │
│  (特征选择网络)   │    │  (可解释模块)     │
└────────┬──────────┘    └────────┬──────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         Temporal Decoder               │
│    (LSTM + Multi-Head Self-Attention) │
│        (时序解码器)                   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌──────────────���──────────────────────────┐
│           Output Layer                 │
│   (多步预测 + 分位数输出 + 可解释性)    │
└─────────────────────────────────────────┘
```

### 2.2 核心创新

TFT相比普通Transformer的三大创新：

| 模块 | 作用 | 创新点 |
|------|------|--------|
| **Variable Selection Network** | 特征选择 | 自动学习哪些协变量重要 |
| **Static Covariate Encoding** | 静态特征 | 处理不随时间变化的特征 |
| **Temporal Decoder** | 时序建模 | LSTM + Transformer融合 |
| **Interpretability** | 可解释性 | 注意力权重+特征重要性 |

### 2.3 与其他方法对比

| 方法 | 序列建模 | 可解释性 | 多步预测 | 灵活输入 |
|------|---------|---------|---------|----------|
| LSTM | ✓ | ✗ | 需改进 | 一般 |
| Transformer | ✓ | ✗ | ✓ | ✓ |
| **TFT** | ✓ | ✓ | ✓ | ✓ |
| DeepAR | ✓ | ✗ | ✓ | ✓ |

### 2.4 工作流程

```
1. 输入处理
   ├── 已知未来输入 (未来已知特征)
   ├── 已知历史输入 (历史观测)
   └── 静态特征 (不随时间变)

2. 特征选择
   ├── Variable Selection Network
   └── 软选择 + 门控

3. 序列建模
   ├── LSTM Encoder (短期模式)
   ├── Multi-Head Attention (长期依赖)
   └── Temporal Fusion (融合)

4. 输出
   ├── 多步预测
   └── 注意力权重
```

---

## 3. 数学公式与推导

### 3.1 输入处理

#### 3.1.1 嵌入层

对于每个输入变量x，进行线性嵌入：
$$E = W_x \cdot x + b_x$$

其中$W_x$是可学习的权重矩阵。

#### 3.1.2 时间位置编码

使用正弦位置编码：
$$PE(pos, 2i) = sin(pos / 10000^{2i/d})$$
$$PE(pos, 2i+1) = cos(pos / 10000^{2i/d})$$

### 3.2 Gated Residual Network (GRN)

门控残差网络是TFT的核心模块：
$$GRN(x) = \alpha \odot W_3(ReLU(W_1 x + b_1)) + (1-\alpha) \odot x$$

其中门控权重：
$$\alpha = \sigma(W_2 x + b_2)$$

作用：
- ReLU引入非线性
- 门控允许信息直通
- 残差连接稳定训练

### 3.3 Variable Selection Network

#### 3.3.1 特征重要性

对于每个特征j，计算重要性分数：
$$v_j = \sum_{i} |W_{sel,ij}|$$

#### 3.3.2 软选择

$$E_{sel} = \sum_j \gamma_j \odot E_j$$

$$\gamma_j = \frac{exp(v_j)}{sum_k exp(v_k)}$$

### 3.4 Temporal Decoder

#### 3.4.1 LSTM编码

$$h_t = LSTM(h_{t-1}, E_t)$$

#### 3.4.2 多头注意力

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d}}\right)V$$

其中：
- Q = LSTM输出
- K, V = 历史编码

### 3.5 分位数输出

为了预测不确定性，TFT输出多个分位数：

分位数损失：
$$L_q = \frac{1}{q-0.5} (\hat{y}_q - y) \cdot (\mathbb{I}_{y<\hat{y}_q} - q)$$

常用分位数：q ∈ {0.1, 0.5, 0.9}

---

## 4. 训练过程讲解

### 4.1 数据准备

TFT需要特殊的数据格式：

```python
from pytorch_forecasting import TimeSeriesDataSet

# 定义数据集
train_dataset = TimeSeriesDataSet(
    data,
    time_idx="time_idx",        # 时间索引
    target="target",            # 预测目标
    group_ids=["group"],       # 时间序列分组
    encoder_length=24,        # 编码长度（历史）
    decoder_length=48,          # 解码长度（未来）
    static_categoricals=["categorical"],   # 静态类别
    static_reals=["numeric"],              # 静态数值
    time_varying=["value"],                # 时变特征
    known_future=["future_known"],         # 已知未来特征
)
```

### 4.2 模型定义

```python
from pytorch_forecasting.models.tft import TemporalFusionTransformer

# 创建模型
model = TemporalFusionTransformer.from_dataset(
    train_dataset,
    hidden_size=64,           # 隐藏层大小
    attention_head_size=4,    # 注意力头数
    dropout=0.1,             # Dropout
    hidden_continuous_size=64, # 连续特征维度
)
```

### 4.3 训练配置

```python
from pytorch_forecasting import Trainer

trainer = Trainer(
    max_epochs=30,
    gpus=1,
    gradient_clip_val=1.0,
)

trainer.fit(model, train_dataloader)
```

### 4.4 超参数选择

| 参数 | 典型值 | 说明 |
|------|--------|------|
| hidden_size | 64-256 | 隐藏层维度 |
| attention_head_size | 4-8 | 注意力头数 |
| dropout | 0.1-0.3 | Dropout率 |
| encoder_length | 24-168 | 历史窗口 |
| decoder_length | 24-168 | 预测窗口 |
| batch_size | 64-256 | 批量大小 |

---

## 5. 应用场景

### 5.1 需求预测

| 场景 | 输入特征 | 预测目标 |
|------|---------|----------|
| 零售销量 | 历史销量、价格、促销 | 未来7天销量 |
| 电商订单 | 历史订单、流量 | 未来订单量 |

### 5.2 金融预测

| 场景 | 输入特征 | 预测目标 |
|------|---------|----------|
| 股价预测 | 历史价格、成交量 | 未来价格区间 |
| 风险评估 | 历史波动、公司指标 | 风险分位数 |

### 5.3 物联网

| 场景 | 输入特征 | 预测目标 |
|------|---------|----------|
| 传感器预测 | 历史读数、设备状态 | 未来读数 |
| 异常检测 | 时序数据 | 异常概率 |

### 5.4 能源预测

| 场景 | 输入特征 | 预测目标 |
|------|---------|----------|
| 电力负荷 | 历史负荷、天气 | 未来负荷 |
| 风电预测 | 风机数据、气象 | 发电量 |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **多步预测** | 同时输出多个未来时刻 |
| **可解释性** | 注意力+特征重要性 |
| **灵活输入** | 支持静态+时变+已知未来 |
| **不确定性** | 分位数输出 |
| **处理长序列** | Transformer处理长期依赖 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **计算量大** | RNN+Transformer双重计算 |
| **调参难** | 超参数多 |
| **数据要求** | 需要足够的历史数据 |
| **对小数据** | 容易过拟合 |

### 6.3 注意事项

- 数据量：建议至少1000个时间点
- 特征质量：特征工程很重要
- 归一化：需要正确的归一化
- 过拟合：小数据集需要正则化

---

## 7. 调库实现（Python）

### 7.1 安装

```bash
pip install pytorch-forecasting torch
```

### 7.2 完整代码

```python
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, Trainer
from pytorch_forecasting.models.tft import TemporalFusionTransformer
import matplotlib.pyplot as plt


def prepare_data(num_series=10, num_timesteps=500):
    """准备示例数据"""
    np.random.seed(42)
    
    data = []
    for group in range(num_series):
        base = np.random.rand() * 100
        trend = np.linspace(0, 50, num_timesteps)
        seasonal = 20 * np.sin(np.arange(num_timesteps) * 2 * np.pi / 30)
        
        for t in range(num_timesteps):
            data.append({
                'group': str(group),
                'time_idx': t,
                'target': base + trend[t] + seasonal[t] + np.random.randn(),
                'value': np.random.randn(),
            })
    
    return pd.DataFrame(data)


def train_tft(train_data, encoder_length=24, decoder_length=48):
    """训练TFT模型"""
    
    # 创建数据集
    train_dataset = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        static_categoricals=[],
        time_varying=["value"],
    )
    
    # 创建DataLoader
    train_dataloader = train_dataset.to_dataloader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    
    # 创建模型
    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
    )
    
    # 训练
    trainer = Trainer(
        max_epochs=10,
        accelerator="cpu",
    )
    trainer.fit(model, train_dataloader)
    
    return model, train_dataset


def predict_with_explainability(model, test_data, train_dataset):
    """预测并解释"""
    
    # 预测
    predictions = model.predict(test_data)
    
    # 获取注意力权重
    attention = model.predict_learned_attention()
    
    # 获取特征重要性
    feature_importance = model.get_feature_importance()
    
    return predictions, attention, feature_importance


def visualize_results(model, train_dataset):
    """可视化结果"""
    
    # 1. 特征重要性
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 特征重要性
    importance = model.get_feature_importance()
    axes[0, 0].barh(range(len(importance)), importance)
    axes[0, 0].set_title('Feature Importance')
    
    # 2. 预测示例
    raw_predictions = model.predict_validation()
    axes[0, 1].plot(raw_predictions)
    axes[0, 1].set_title('Validation Predictions')
    
    plt.tight_layout()
    plt.savefig('tft_results.png', dpi=100)
    plt.show()


if __name__ == "__main__":
    print("准备数据...")
    train_data = prepare_data(num_series=5, num_timesteps=200)
    
    print("训练模型...")
    model, train_dataset = train_tft(train_data)
    
    print("预测并解释...")
    visualize_results(model, train_dataset)
    
    print("训练完成!")
```

---

## 8. 手工代码实现（理解���理）

### 8.1 简化版TFT

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGRN(nn.Module):
    """简化的门控残差网络"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        gated = torch.sigmoid(self.w2(x))
        hidden = F.relu(self.w1(x))
        output = self.w3(hidden)
        return gated * output + (1 - gated) * x


class VariableSelection(nn.Module):
    """简化的特征选择网络"""
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.grn = SimpleGRN(num_features, hidden_dim)
        self.weights = nn.Linear(num_features, 1)
    
    def forward(self, features):
        importance = torch.softmax(self.weights(features), dim=-1)
        selected = self.grn(features)
        return importance * selected


class SimpleLSTMDecoder(nn.Module):
    """简化的LSTM解码器"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
    
    def forward(self, x, key_values):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, key_values, key_values)
        return attn_out


class SimpleTFT(nn.Module):
    """简化版TFT - 理解原理"""
    def __init__(self, num_features, hidden_dim=64, output_dim=1):
        super().__init__()
        self.feature_selection = VariableSelection(num_features, hidden_dim)
        self.decoder = SimpleLSTMDecoder(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x_encoder, x_decoder):
        # 特征选择
        selected = self.feature_selection(x_encoder)
        
        # 编码
        encoder_out = selected.unsqueeze(1)
        
        # 解码+注意力
        decoder_out = self.decoder(x_decoder.unsqueeze(1), encoder_out)
        
        # 输出
        output = self.output(decoder_out.squeeze(1))
        
        return output


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 测试
    num_features = 5
    model = SimpleTFT(num_features, hidden_dim=32)
    
    x_encoder = torch.randn(10, num_features)
    x_decoder = torch.randn(5, num_features)
    
    output = model(x_encoder, x_decoder)
    
    print(f"输入: {x_encoder.shape}")
    print(f"输出: {output.shape}")
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
    timesteps = 100
    attention_weights = np.random.rand(timesteps, timesteps)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 注意力热力图
    im = axes[0].imshow(attention_weights, cmap='viridis', aspect='auto')
    axes[0].set_title('Attention Weights')
    plt.colorbar(im, ax=axes[0])
    
    # 单时刻注意力
    axes[1].plot(attention_weights[50])
    axes[1].set_title('Attention at t=50')
    axes[1].set_xlabel('Historical Timestep')
    
    plt.tight_layout()
    plt.savefig('tft_attention.png', dpi=100)
    plt.show()


def visualize_predictions():
    """可视化预测结果"""
    
    np.random.seed(42)
    n = 100
    
    # 真实值
    y_true = np.cumsum(np.random.randn(n))
    
    # 预测（分位数）
    y_pred = y_true + np.random.randn(n) * 0.1
    y_lower = y_pred - np.random.rand(n) * 2
    y_upper = y_pred + np.random.rand(n) * 2
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, 'b-', label='True')
    plt.plot(y_pred, 'r-', label='Prediction')
    plt.fill_between(range(n), y_lower, y_upper, alpha=0.3, label='95% CI')
    plt.legend()
    plt.title('TFT Multi-step Prediction')
    plt.savefig('tft_prediction.png', dpi=100)
    plt.show()


if __name__ == "__main__":
    visualize_attention()
    visualize_predictions()
```

### 9.2 特征重要性图

```python
def plot_feature_importance():
    """特征重要性图"""
    
    features = ['price', 'promotion', 'weather', 'trend', 'seasonality']
    importance = np.random.rand(len(features))
    importance = importance / importance.sum()
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance)
    plt.xlabel('Importance')
    plt.title('TFT Feature Importance')
    plt.savefig('tft_feature_importance.png', dpi=100)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| MAE | 平均绝对误差 |
| RMSE | 均方根误差 |
| Quantile Loss | 分位数损失 |
| Coverage | 覆盖率 |

### 10.2 评估代码

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_tft(y_true, y_pred, y_lower=None, y_upper=None):
    """评估TFT模型"""
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    metrics = {'MAE': mae, 'RMSE': rmse}
    
    if y_lower is not None and y_upper is not None:
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        metrics['Coverage'] = coverage
    
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    return metrics


def quantile_loss(y_true, y_pred, q):
    """分位数损失"""
    err = y_true - y_pred
    return np.mean(np.maximum(q * err, (q - 1) * err))
```

---

## 11. 常见问题与易错点

### Q1: TFT需要多少数据？

**答案**：建议至少1000个时间点，数据太少容易过拟合。

### Q2: 如何设置encoder/decoder长度？

**答案**：encoder_length通常是预测周期的2-3倍，如预测7天则encoder=21。

### Q3: 为什么注意力可视化是空的？

**answers**：需要设置`mode="raw"`调用`predict_learned_attention()`。

### Q4: 如何提高预测精度？

**答案**：1）添加更多相关特征 2）调优hidden_size 3）增加encoder_length。

### Q5: TFT vs Transformer区别？

**答案**：TFT多了特征选择、静态编码、分位数输出和可解释性。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心创新 | 特征选择+静态编码+可解释 |
| 序列建模 | LSTM + Transformer |
| 输出 | 多步预测+分位数 |
| 评估 | 注意力+特征重要性 |

### 12.2 公式汇总

GRN:
$$GRN(x) = \alpha \odot W_3(ReLU(W_1 x)) + (1-\alpha) \odot x$$

注意力:
$$Attention(Q,K,V) = softmax(QK^T/√d)V$$

分位数损失:
$$L_q = \frac{1}{q-0.5}(\hat{y}_q - y)(\mathbb{I}_{y<\hat{y}_q} - q)$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. TFT的核心创新是：
   - A) 使用Transformer
   - B) 可解释性
   - C) 特征选择+LSTM+Transformer

2. TFT的输出包括：
   - A) 单点预测
   - B) 多步+分位数+注意力

3. Variable Selection的作用是：
   - A) 选择重要特征
   - B) 特征嵌入
   - C) 特征归一化

### 13.2 简答题

1. 解释GRN的工作原理？
2. 比较TFT和Transformer的区别？
3. 为什么TFT需要分位数输出？

### 13.3 编程题

1. 用pytorch-forecasting训练TFT模型。
2. 实现��意��可视化。
3. 比较不同hidden_size的效果。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
时间序列基础
    ↓
LSTM/GRU
    ↓
Transformer
    ↓
TFT原理
    ↓
实战项目
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| LSTM | 序列建模基础 |
| Transformer | 注意力机制 |
| DeepAR | 时间序列预测 |
| N-BEATS | 时间序列SOTA |

### 14.3 扩展阅读

1. Lim et al. (2020). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
2. pytorch-forecasting文档

---

## 附录

### A. 快速开始

```python
# 最小示例
from pytorch_forecasting.models.tft import TemporalFusionTransformer

model = TemporalFusionTransformer.from_dataset(
    dataset,
    hidden_size=32,
)
trainer.fit(model, dataloader)
```

### B. 参考

1. Lim et al. (2020). Temporal Fusion Transformers. arXiv:1912.09363
2. https://pytorch-forecasting.readthedocs.io/

---

**文档结束**