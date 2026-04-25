# DeepAR 学习文档

> Amazon提出的概率自回归时间序列预测模型，通过RNN输出概率分布实现不确定性量化

---

## 1. 算法基础认知

### 1.1 一句话定义

DeepAR是Amazon于2019年提出的基于RNN的概率自回归时间序列预测模型，通过在循环神经网络中输出概率分布来实现预测和不确定性量化，特别擅长处理多维时间序列和具有复杂季节性模式的数据。

### 1.2 直觉类比

DeepAR就像一个"会预测未来的统计学家"。传统的时间序列预测只告诉我们"明天销售额是100"，DeepAR会说"明天销售额有70%的概率在90-110之间，95%的概率在80-120之间"——它不只给出一个点估计，而是给出一个完整的概率分布！

想象你是一个零售店经理：
- 传统预测：下周进500件货（点估计）
- DeepAR：根据历史数据，不仅预测销量，还能告诉你"有80%把握进450-550件，有95%把握进400-600件"——这就是概率分布预测！

### 1.3 发展背景

- 2019年，Amazon的Flunkina等人在论文"DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Neural Networks"中提出
- 源自Amazon的实际业务需求（库存管理、需求预测）
- 在Amazon内部广泛应用，效果显著

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 时间序列 → 概率预测 |
| 输出 | 概率分布（均值+方差） |
| 模型 | 自回归RNN |
| 特点 | 端到端不确定性量化 |

---

## 2. 核心原理

### 2.1 为什么需要概率预测？

传统点估计方法存在局限性：

| 问题 | 影响 |
|------|------|
| 无法量化不确定性 | 决策风险不可知 |
| 无法估计置信区间 | 决策缺乏依据 |
| 无法处理多步预测误差累积 | 长序列预测失效 |

DeepAR通过输出完整的概率分布 $P(y_{t+1:T}|y_{1:t})$ 来解决这些问题。

### 2.2 核心思想

DeepAR使用自回归RNN作为条件概率模型：

$$P(y_{t+1:T}|y_{1:t}) = \prod_{t+1}^{T} P(y_\tau|y_{1:\tau-1})$$

其中每个 $P(y_\tau|y_{1:\tau-1})$ 由RNN的参数化分布（如高斯分布或负二项分布）给出。

### 2.3 架构流程

```
时间序列输入 y_{1:t}
    │
    ▼
Embedding/特征变换
    │
    ▼
LSTM/GRU 自回归编码
    │
    ▼
概率分布输出层
    │
    ▼
预测分布 P(y_{t+1}|y_{1:t})
```

### 2.4 关键创新

1. **概率分布输出**：使用参数化分布而非点估计
2. **自回归建模**：利用时间依赖性
3. **局部+全局特征**：同时学习局部模式和全局协变量

---

## 3. 数学公式与推导

### 3.1 条件概率建模

对于给定历史观测 $y_{1:t}$，未来值 $y_{t+1}$ 的条件概率为：

$$P(y_{t+1}|y_{1:t}) = \mathcal{N}(\mu_{t+1}, \sigma_{t+1}^2)$$

其中：
- $\mu_{t+1} = h_\mu(h_t, x_{t+1})$ 是均值网络
- $\sigma_{t+1} = \exp(h_\sigma(h_t, x_{t+1}))$ 是方差网络（取指数确保方差为正）
- $h_t$ 是RNN的隐藏状态

### 3.2 目标函数

负对数似然（Negative Log-Likelihood）：

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t|y_{1:t-1})$$

对于高斯分布：
$$\log P(y_t|y_{1:t-1}) = -\frac{1}{2}\left(\log(2\pi\sigma_t^2) + \frac{(y_t - \mu_t)^2}{\sigma_t^2}\right)$$

### 3.3 RNN隐藏状态更新

LSTM或GRU的隐藏状态更新：

$$h_t = \text{RNN}(y_{t-1}, h_{t-1}, x_t)$$

其中 $x_t$ 是时间 $t$ 的外生变量（协变量）。

### 3.4 训练过程

**前向传播**：
$$h_t \leftarrow h_t(h_{t-1}, y_{t-1})$$
$$\mu_t, \sigma_t \leftarrow \text{DistributionParams}(h_t)$$

**计算损失**：
$$\mathcal{L}_t \leftarrow -\log P(y_t|\mu_t, \sigma_t)$$

**反向传播**：$\nabla_\theta \mathcal{L}_t$

### 3.5 多步预测

对于 $h$ 步预测，递归使用：

$$\hat{y}_{t+1} \sim \mathcal{N}(\mu_{t+1}, \sigma_{t+1}^2)$$
$$\hat{y}_{t+2} \sim \mathcal{N}(\mu_{t+2}, \sigma_{t+2}^2)$$

注意：预测时采样 $\hat{y}$ 作为下一个时间步的输入，形成递归。

---

## 4. 训练过程讲解

### 4.1 数据准备

```python
# 时间序列数据格式
# 每条序列: [timestamp, value, covariates...]
# 训练数据: list of series

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, series_list, context_length, prediction_length):
        self.series_list = series_list
        self.context_length = context_length
        self.prediction_length = prediction_length
    
    def __getitem__(self, idx):
        series = self.series_list[idx]
        # 随机选择起始点
        start_idx = np.random.randint(0, len(series) - self.context_length - self.prediction_length)
        
        # 输入和目标
        context = series[start_idx:start_idx+self.context_length]
        target = series[start_idx+self.context_length:start_idx+self.context_length+self.prediction_length]
        
        return torch.FloatTensor(context), torch.FloatTensor(target)
```

### 4.2 模型配置

```python
# DeepAR配置
config = {
    'input_dim': 10,           # 特征维度（含协变量）
    'rnn_type': 'lstm',         # LSTM或GRU
    'rnn_layers': 2,           # RNN层数
    'rnn_hidden': 40,            # 隐藏维度
    'distribution': 'normal',   # 分布类型（normal/negative-binomial）
    'context_length': 24,       # 输入序列长度
    'prediction_length': 6,     # 预测长度
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}
```

### 4.3 训练循环

```python
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        context, target = batch
        optimizer.zero_grad()
        
        # 前向传播
        distribution_params = model(context)
        loss = model.loss(distribution_params, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### 4.4 评估与调参

```python
# 评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(model, test_data):
    predictions = []
    actuals = []
    
    model.eval()
    with torch.no_grad():
        for context, target in test_data:
            pred = model.predict(context)
            predictions.append(pred)
            actuals.append(target)
    
    # 计算指标
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    
    # 计算P50和P90分位数误差
    # ...
    
    return {'MSE': mse, 'MAE': mae}
```

---

## 5. 应用场景

### 5.1 需求预测

零售场景中的商品需求预测：

```python
# 电商需求预测示例
# 输入：历史销售数据 + 协变量（促销、价格、节假日）
# 输出：未来N天的需求分布

model = DeepAR(input_dim=10)
model.fit(sales_history)  # 训练

# 预测
future_demand = model.predict(weeks=4)
# future_demand.mean = [100, 120, 90, 110]
# future_demand.quantile(0.1) = [80, 95, 70, 85]
# future_demand.quantile(0.9) = [125, 150, 115, 140]
```

### 5.2 金融时间序列

股票价格和风险预测：

```python
# 金融预测
# 输出：价格分布，可计算VaR等风险指标

price_predictions = model.predict(asset_prices)
# 计算95% VaR
var_95 = price_predictions.quantile(0.05)
```

### 5.3 能源负荷预测

电力负荷预测：

```python
# 电网负荷预测
# 输入：历史负荷 + 温度 + 时间特征
# 输出：负荷分布

load_predictions = model.predict(load_series)
```

### 5.4 对比传统方法

| 方法 | 输出 | 不确定性 | 多维支持 |
|------|------|----------|----------|
| ARIMA | 点估计 | 无 | 弱 |
| Prophet | 点估计 | 部分 | 弱 |
| VAR | 点估计 | 无 | 强 |
| **DeepAR** | **分布** | **完整** | **强** |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 不确定性量化 | 输出完整概率分布 |
| 多维支持 | 同时预测多相关序列 |
| 复杂模式 | 学习非线性季节性 |
| 端到端 | 统一训练框架 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 计算复杂度 | O(T×hidden_dim) |
| 数据需求 | 需要较长序列 |
| 超参数敏感 | RNN深度分布 |
| 可解释性弱 | 神经网络黑箱 |

### 6.3 注意事项

- 需要足够的训练数据（建议每个序列100+时间点）
- 协变量需要提前准备好
- 长期预测误差会累积

---

## 7. 调库实现（Python + GluonTS）

### 7.1 GluonTS实现

```python
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.mx.distribution import NormalOutput

# 创建数据集
from gluonts.dataset.repository import get_dataset

dataset = get_dataset("m4_hourly")

# 创建模型
estimator = DeepAREstimator(
    freq="H",                    # 数据频率
    prediction_length=48,         # 预测长度
    context_length=168,           # 输入长度
    num_layers=2,                 # RNN层数
    hidden_size=40,               # 隐藏维度
    distr_output=NormalOutput(),    # 输出分布
    trainer=Trainer(
        epochs=30,
        learning_rate=0.001,
        num_batches_per_epoch=100
    )
)

# 训练
predictor = estimator.train(dataset.train)

# 预测
from gluonts.evaluation import Evaluator
from gluonts.evaluation_quantileEvaluator import QuantileEvaluator

forecasts = list(predictor.predict(dataset.test))
evaluator = QuantileEvaluator(quantiles=[0.5, 0.9])
results = evaluator(forecasts, dataset.test)

print(f"P50 Error: {results['P50']}")
print(f"P90 Error: {results['P90']}")
```

### 7.2 PyTorch实现

```python
import torch
import torch.nn as nn

class DeepARModel(nn.Module):
    def __init__(self, input_dim, rnn_type='lstm', rnn_layers=2, 
                 rnn_hidden=40, distribution='normal'):
        super().__init__()
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.rnn_hidden = rnn_hidden
        
        # RNN
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, rnn_hidden, rnn_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, rnn_hidden, rnn_layers, batch_first=True)
        
        # 分布输出层
        self.distr_mean = nn.Linear(rnn_hidden, 1)
        self.distr_log_std = nn.Linear(rnn_hidden, 1)
        
        self.distribution = distribution
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        rnn_out, (h_n, c_n) = self.rnn(x)
        
        # 取最后一个隐藏状态
        last_hidden = rnn_out[:, -1, :]
        
        # 输出分布参数
        mean = self.distr_mean(last_hidden)
        log_std = self.distr_log_std(last_hidden)
        std = torch.exp(log_std)  # 确保 std > 0
        
        return mean, std
    
    def loss(self, mean, std, target):
        # 负对数似然
        nll = 0.5 * torch.log(2 * torch.pi * std**2) + (target - mean)**2 / (2 * std**2)
        return nll.mean()
    
    def predict(self, x):
        mean, std = self.forward(x)
        return mean, std


# 使用示例
model = DeepARModel(input_dim=5)

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    for context, target in dataloader:
        optimizer.zero_grad()
        mean, std = model(context)
        loss = model.loss(mean, std, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss={loss.item():.4f}")

# 预测
mean, std = model.predict(test_context)
print(f"预测均值: {mean.item():.2f}, 标准差: {std.item():.2f}")
```

### 7.3 评估与可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_forecasts(test_data, forecasts, quantiles=[0.1, 0.5, 0.9]):
    """绘制预测结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (test, forecast) in enumerate(zip(test_data[:4], forecasts[:4])):
        if i >= 4:
            break
        
        ax = axes[i]
        t = np.arange(len(test))
        
        # 绘制真实值
        ax.plot(t, test, 'b-', label='Actual')
        
        # 绘制预测分位数
        for q in quantiles:
            f = np.array([f[q] for f in forecast])
            ax.plot(t, f, '--', label=f'P{int(q*100)}')
        
        ax.legend()
        ax.set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig('deepar_forecasts.png', dpi=100)
    plt.show()
```

---

## 8. 手工代码实现（理解原理）

```python
import numpy as np
from scipy.stats import norm

class DeepARManual:
    """简化版DeepAR - 理解原理"""
    
    def __init__(self, rnn_hidden=20, rnn_layers=1, lr=0.01):
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.lr = lr
        
        # 简化的RNN权重（单层）
        self.Wxh = np.random.randn(1, rnn_hidden) * 0.1  # 输入到隐藏
        self.Whh = np.random.randn(rnn_hidden, rnn_hidden) * 0.1  # 隐藏到隐藏
        self.bh = np.zeros(rnn_hidden)
        
        # 输出层
        self Why = np.random.randn(rnn_hidden, 1) * 0.1
        self.Wsy = np.random.randn(rnn_hidden, 1) * 0.1
        self.by = np.zeros(1)
        self.bs = np.zeros(1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def rnn_step(self, x_t, h_prev):
        """单步RNN"""
        h = self.tanh(x_t @ self.Wxh + h_prev @ self.Whh + self.bh)
        return h
    
    def forward(self, X):
        """前向传播
        
        X: [seq_len, input_dim]
        """
        seq_len = len(X)
        h = np.zeros(self.rnn_hidden)
        
        for t in range(seq_len):
            x_t = X[t:t+1] if X.ndim > 1 else np.array([X[t]])
            h = self.rnn_step(x_t, h)
        
        # 输出分布参数
        mean = h @ self.Why + self.by
        log_std = h @ self.Wsy + self.bs
        std = np.exp(np.clip(log_std, -10, 10))
        
        return mean.flatten(), std.flatten()
    
    def loss(self, X, y):
        """负对数似然"""
        mean, std = self.forward(X)
        nll = 0.5 * np.log(2*np.pi*std**2) + (y - mean)**2 / (2*std**2)
        return nll
    
    def predict(self, X):
        """预测"""
        mean, std = self.forward(X)
        return mean, std
    
    def predict_future(self, X, steps, samples=100):
        """多步预测，采样"""
        predictions = []
        
        for _ in range(samples):
            current_input = X.copy()
            sample_preds = []
            
            for _ in range(steps):
                mean, std = self.forward(current_input)
                # 采样
                y_sample = np.random.normal(mean, std)
                sample_preds.append(y_sample)
                
                # 更新输入（简化：只用值）
                current_input = np.array([y_sample])
            
            predictions.append(sample_preds)
        
        predictions = np.array(predictions)
        return predictions.mean(axis=0), predictions.std(axis=0)


# 测试
if __name__ == "__main__":
    np.random.seed(42)
    
    # 生成模拟数据（正弦波 + 噪声）
    t = np.linspace(0, 20, 200)
    y = 10 * np.sin(t) + 2 + np.random.randn(200) * 0.5
    
    # 训练
    model = DeepARManual(rnn_hidden=10)
    
    # 简单训练几步
    for i in range(100):
        idx = np.random.randint(10, 190)
        context = y[idx-10:idx]
        target = y[idx]
        
        loss = model.loss(context, target)
        
        # 简化更新（梯度下降）
        mean, std = model.forward(context)
        error = (target - mean) / (std**2 + 1e-8)
        # 这里略去真正的梯度更新...
    
    # 预测未来
    context = y[-10:]
    future_mean, future_std = model.predict(context)
    
    print(f"预测: {future_mean:.2f} ± {future_std:.2f}")
    print(f"实际下一步: {y[-1]:.2f}")
```

---

## 9. 可视化与结果理解

### 9.1 预测可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(y_history, y_actual, forecasts, quantiles=[0.1, 0.5, 0.9]):
    """可视化预测结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 历史数据
    t_hist = np.arange(len(y_history))
    axes[0, 0].plot(t_hist, y_history, 'b-', label='Historical')
    axes[0, 0].set_title('Historical Data')
    axes[0, 0].legend()
    
    # 预测结果
    t_future = np.arange(len(y_actual))
    axes[0, 1].plot(t_future, y_actual, 'b-', label='Actual')
    
    for q in quantiles:
        pred_q = [f[q] for f in forecasts]
        axes[0, 1].plot(t_future, pred_q, '--', label=f'P{int(q*100)}')
    
    axes[0, 1].set_title('Forecast with Quantiles')
    axes[0, 1].legend()
    
    # 误差分布
    errors = [np.abs(f[0.5] - a) for f, a in zip(forecasts, y_actual)]
    axes[1, 0].hist(errors, bins=20)
    axes[1, 0].set_title('Absolute Error Distribution')
    
    # 分位数覆盖
    coverage = []
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        covered = sum(
            f[q] <= a <= f[1-q] 
            for f, a in zip(forecasts, y_actual)
        ) / len(y_actual)
        coverage.append((q, covered))
    
    axes[1, 1].bar([f'{int(q*100)}' for q, _ in coverage], 
                   [c for _, c in coverage])
    axes[1, 1].axhline(y=0.8, color='r', linestyle='--', 
                        label='Target 80%')
    axes[1, 1].set_title('Quantile Coverage')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('deepar_visualization.png', dpi=100)
    plt.show()
```

### 9.2 区间预测解释

```python
# 预测结果的解释
forecast_example = {
    'mean': 100,
    'std': 10,
    'P10': 88,
    'P50': 100,
    'P90': 112
}

print("95%预测区间: [88, 112]")
print("有95%的概率真实值在这个范围内")
print(f"的90%预测区间: [{forecast_example['P10']}, {forecast_example['P90']}]")
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| MSE/MAE | 点估计误差 |
| P50 | P50分位数误差 |
| P90 | P90分位数误差 |
| Coverage | 区间覆盖率 |
| CRPS | 连续排名概率分数 |

### 10.2 计算P50/P90误差

```python
def compute_quantile_errors(y_true, forecasts):
    """计算分位数误差"""
    errors = {}
    
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        pred_q = np.array([f[q] for f in forecasts])
        mae = np.mean(np.abs(pred_q - y_true))
        errors[f'P{int(q*100)}'] = mae
    
    return errors

# 评估
errors = compute_quantile_errors(test_data, forecasts)
for q, mae in errors.items():
    print(f"{q} MAE: {mae:.2f}")
```

---

## 11. 常见问题与易错点

### Q1: 如何处理多个相关序列？

**答案**：DeepAR支持多维输入，将多个序列作为特征维度即可。

### Q2: 长期预测误差累积？

**答案**：采样递归预测会导致误差累积，这是正常现象。可以用蒙特卡洛采样来估计不确定性。

### Q3: 训练数据不足？

**答案**：建议每个时间序列至少100个时间点。可以使用迁移学习。

### Q4: 如何选择分布类型？

**答案**：count数据用负二项分布，持续值用高斯分布。

### Q5: 协变量如何处理？

**答案**：协变量和目标值一起作为输入特征。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心思想 | 概率自回归建模 |
| 输出 | 完整概率分布 |
| 优势 | 不确定性量化 |
| 模型 | RNN + 分布参数化 |

### 12.2 公式汇总

条件概率：
$$P(y_t|y_{1:t-1}) = \mathcal{N}(\mu_t, \sigma_t^2)$$

负对数似然：
$$\mathcal{L} = -\log P(y_t|\mu_t, \sigma_t)$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. DeepAR的核心优势是：
   - A) 预测精度高
   - B) 不确定性量化
   - C) 训练快

2. DeepAR使用什么输出分布：
   - A) 分类分布
   - B) 高斯分布
   - C) 泊松分布

### 13.2 简答题

1. 解释为什么DeepAR能输出概率分布？
2. 比较多步预测中点估计和采样的区别。

### 13.3 编程题

1. 实现简化版DeepAR并预测正弦波数据。
2. 用GluonTS在M4数据集上测试DeepAR。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
时间序列基础
    ↓
ARIMA/LSTM
    ↓
概率预测
    ↓
DeepAR
    ↓
多维序列
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| LSTM | 基础RNN |
| Transformer | 时序Transformer |
| N-BEATS | 神经网络时间序列 |
| TimesFM | Google新模型 |

### 14.3 扩展阅读

- Flunkina et al. (2019). DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Neural Networks. Amazon

---

## 附录

### 参考

1. Flunkina, A., et al. (2019). DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Neural Networks. Amazon Research.
2. GluonTS: Probabilistic Time Series Modeling in Python
3. https://gluon-ts.readthedocs.io/

---

**文档结束**