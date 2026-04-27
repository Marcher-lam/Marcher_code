# N-BEATS 学习文档

> 神经基扩展自回归时间序列预测模型，无需任何时间序列假设的端到端预测

---

## 1. 算法基础认知

**一句话定义**：N-BEATS（Neural Basis Expansion AutoRegressive Time Series）是一种无需任何时间序列假设（如平稳性、季节性）的深度学习预测模型，通过双栈架构自动学习数据中的趋势和周期模式，在M4竞赛中取得领先成绩。

**直觉类比**：N-BEATS就像一个"智能音乐作曲家"。想象你在听一首复杂的交响乐，N-BEATS不需要知道音乐是什么风格的，它自己能从中分离出"主旋律"（趋势部分）和"装饰音"（周期部分）。它把复杂的时间序列分解为"基础乐章"，分别预测后再叠加起来得到最终预测。这比传统的"假设数据符合某种模式"的方法灵活得多。

**历史背景**：
- 2020年，Oreshkin等人提出N-BEATS
- 在M4竞赛中获得领先成绩（MAPE显著优于传统方法）
- 后续发展出N-HiTS、N-BEATSx等变体

**核心定位**：N-BEATS代表了"端到端深度学习"在时间序列预测领域的成功实践，打破了传统统计方法的假设限制。

**前置知识**：
- [必备]：时间序列基础（趋势、周期、平稳性）
- [必备]：深度学习基础（全连接层、激活函数）
- [推荐]：LSTM/GRU序列模型

---

## 2. 核心原理

### 2.1 传统方法的局限

传统时间序列预测方法面临诸多局限：

**（1）ARIMA的假设约束**
- 需要数据平稳（差分处理后平稳）
- 需要手动识别p, q参数
- 对异常值敏感

**（2）指数平滑的线性假设**
- 假设趋势和周期是线性叠加
- 无法捕捉复杂非线性模式

**（3）Prophet的先验假设**
- 假设趋势是分段线性
- 假设周期是傅里叶级数
- 需要人工指定周期

**N-BEATS的创新**：用神经网络自动学习这些模式，无须先验假设！

### 2.2 双栈分解架构

N-BEATS的核心创新是**双栈分解**架构：

```
输入：历史序列 [y_1, y_2, ..., y_t]
  │
  ▼
┌──────────────────────────────┐
│      共享嵌入层 (FC)          │ ← 将输入投射到隐藏空间
└──────────────┬───────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌─────────────┐      ┌─────────────┐
│ 趋势栈     │      │ 周期栈      │
│ (Trend)   │      │ (Seasonal) │
└─────┬─────┘      └─────┬─────┘
      │                   │
      ▼                   ▼
┌─────────────┐      ┌─────────────┐
│ 趋势预测    │      │ 周期预测    │
│ forecast_t │      │ forecast_s │
└─────┬─────┘      └─────┬─────┘
      │                   │
      └─────────┬─────────┘
                ▼
         叠加输出
      ┌─────────────┐
      │ forecast  │
      │ (最终预测) │
      └───────────┘
```

**为什么需要两个栈？**

| 栈 | 负责学习的模式 | 典型特征 |
|----|----------------|----------|
| 趋势栈 | 长期趋势 | 线性增长/下降 |
| 周期栈 | 季节性波动 | 周期性变化 |

### 2.3 核心思想：基函数展开

N-BEATS的核心思想是**基函数��开**：

$$\hat{y}_{t+1:t+H} = \sum_{b=1}^{B} \theta_b \cdot \text{basis}_b$$

其中：
- $H$：预测步长（horizon）
- $B$：blocks数量
- $\theta_b$：第b个block学习的系数
- $\text{basis}_b$：基函数（如多项式、三角函数）

**关键洞察**：这就像把时间序列分解为"基础乐章"的叠加。每个block学习一个"基础乐章"，然后叠加起来。

### 2.4 Block机制

每个N-BEATS block包含两个输出：
- **backcast**：对输入历史的拟合（用于学习）
- **forecast**：对未来步的预测（用于输出）

```
         输入 x
           │
     ┌─────┴─────┐
     ▼            ▼
  backcast    forecast
  (回顾)      (预测)
```

**残差链接**：当前block的输入 = 上一个block的输入 - 当前block的backcast

$$x_{new} = x_{old} - \text{backcast}$$

这实现了分块分解！

---

## 3. 数学公式与推导

### 3.1 Block计算

**通用Block公式**：

$$\text{forecast}, \text{backcast} = \text{Block}(x; \theta)$$

其中：
- $x$：输入序列 $[y_{t-L+1}, ..., y_t]$
- $\theta$：可学习参数
- $\text{forecast}$：预测 $[y_{t+1}, ..., y_{t+H}]$
- $\text{backcast}$：回顾 $[y_{t-L+1}, ..., y_t]$

### 3.2 趋势栈（Trend Stack）

**趋势Block**使用多项式基函数：

$$\text{basis}_b^{\text{trend}}(h) = [1, h, h^2, ..., h^D]$$

其中 $h = 1, 2, ..., H$ 是预测步。

**趋势Block计算**：

```python
# 伪代码
def trend_block(x, degree=3):
    # x: [batch, lookback]
    
    # FC1: 提取隐藏表示
    theta = FC1(x)  # [batch, basis_dim]
    
    # 基函数: [1, h, h^2, ..., h^D] for h=1..H
    basis = polynomial_basis(H, degree)  # [H, degree+1]
    
    # 预测 = θ × 基函数
    forecast = theta @ basis.t()  # [batch, H]
    backcast = theta @ basis.t()  # 对于lookback
    
    # FC2: 将基系数转回隐藏空间
    backcast = FC2(forecast)
    
    return forecast, backcast
```

### 3.3 周期栈（Seasonal Stack）

**周期Block**使用傅里叶基函数：

$$\text{basis}_b^{\text{season}}(h) = [\sin(2\pi k h / T), \cos(2\pi k h / T)]_{k=1}^K$$

其中：
- $T$：周期长度（如一年=12个月）
- $K$：傅里叶项数

### 3.4 损失函数

**损失 = 预测误差 + 正则化**：

$$\mathcal{L} = \frac{1}{H} \sum_{h=1}^{H} w_h \cdot |y_{t+h} - \hat{y}_{t+h}| + \lambda \|\theta\|^2$$

其中 $w_h$ 是分层权重（对远期预测更严格）。

### 3.5 解释性

N-BEATS的一个重要优点是**可解释性**：

**趋势分解可视化**：
- 趋势栈的输出 ≈ 长期趋势
- 周期栈的输出 ≈ 季节性波动

**实际应用**：
- 可以单独分析趋势和周期
- 对于业务理解很有帮助

---

## 4. 训练过程讲解

### 4.1 训练流程

```
       准备数据
           │
           ▼
    ┌───────────────┐
    │  加载批次     │ ← 滑动窗口采样
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │   前向传播    │ ← 双栈分解
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  计算损失    │ ← MAE/MSE
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  反向传播    │ ← BPTT
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  更新参数    │ ← Adam
    └───────────────┘
```

### 4.2 数据准备：滑动窗口

```python
def create_sliding_windows(series, lookback=100, horizon=24):
    """创建滑动窗口数据集"""
    
    X, Y = [], []
    
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i:i+lookback])
        Y.append(series[i+lookback:i+lookback+horizon])
    
    return np.array(X), np.array(Y)
```

### 4.3 多尺度预测

N-BEATS支持多尺度预测：

| 任务 | Lookback | Horizon |
|------|----------|---------|
| 短期 | 24 | 1-8 |
| 中期 | 48 | 8-24 |
| 长期 | 100 | 24+ |

### 4.4 超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| lookback | 100 | 回看窗口 |
| horizon | 24 | 预测步长 |
| hidden_dim | 128-256 | 隐藏维度 |
| n_blocks | 1-4 | 每栈blocks数 |
| degree | 2-3 | 多项式阶数 |
| lr | 1e-3 | 学习率 |
| batch_size | 512 | 批次 |

### 4.5 训练技巧

| 技巧 | 说明 |
|------|------|
| 残差链接 | 每block减去backcast |
| 权重平均 | 防止过拟合 |
| 分层损失 | 对不同horizon加权 |

---

## 5. 应用场景

### 5.1 时间序列预测

**核心应用**：任意时间序列的多步预测

```python
# 典型应用
series = load_sales_data()  # 销售序列
lookback = 100
horizon = 24

model = NBEATS(input_dim=lookback, output_dim=horizon)
model.fit(series)  # 训练

forecast = model.predict(series[-lookback:])  # 预测未来24步
```

### 5.2 业务预测

**场景**：
- **零售销量预测**：预测未来销量
- **电力负荷预测**：预测用电量
- **交通流量预测**：预测车流量

### 5.3 M4/M5竞赛

**背景**：
- M4竞赛：10万个时间序列
- M5竞赛：4万个时间序列
- N-BEATS在M4中表现优异

### 5.4 金融预测

**场景**：
- 股票价格预测
- 汇率预测

### 5.5 异常检测

**方法**：
- 预测值与实际值差异大 → 异常
- 用于工业设备监控

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **无假设** | 不需要平稳性、季节性假设 |
| **可解释** | 分解为趋势+周期两部分 |
| **高性能** | M4竞赛领先 |
| **灵活** | 支持任意lookback/horizon |
| **端到端** | 不需要特征工程 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **计算重** | 需要GPU训练 |
| **数据要求** | 需要大量训练数据 |
| **长周期差** | 远期预测精度下降 |
| **黑盒** | 可解释性有限 |

### 6.3 改进方向

| 方向 | 方法 |
|------|------|
| N-HiTS | 加入时间注意力 |
| 时间卷积 | 使用TCN代替FC |
| 多尺度 | 级联预测不同horizon |

---

## 7. 调库实现

### 7.1 使用Darts（推荐）

```python
# 安装
# pip install darts

import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler

# 加载数据
df = pd.read_csv('sales.csv', parse_dates=['date'])
series = TimeSeries.from_dataframe(df, 'date', 'sales')

# 训练/测试分割
train, val = series[:-24], series[-24:]

# 归一化
scaler = Scaler()
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)

# 模型
model = NBEATSModel(
    input_chunk_length=100,
    output_chunk_length=24,
    n_epochs=100,
    random_state=42
)

# 训练
model.fit(train_scaled)

# 预测
pred = model.predict(n=24)

# 反归一化
pred_unscaled = scaler.inverse_transform(pred)
```

### 7.2 使用PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NBEATSBlock(nn.Module):
    """N-BEATS Block"""
    
    def __init__(self, units, basis_dim, degree=3, is_trend=True):
        super().__init__()
        self.is_trend = is_trend
        
        # FC层
        self.fc1 = nn.Linear(units, units)
        self.fc2 = nn.Linear(units, basis_dim)
        self.fc3 = nn.Linear(basis_dim, units)
        
        # 多项式基函数
        if is_trend:
            t = torch.arange(degree + 1, dtype=torch.float32)
            basis = torch.stack([t ** i for i in range(degree + 1)], dim=0)
            self.register_buffer('basis', basis)
            
    def forward(self, x):
        """
        Args:
            x: [batch, units]
        Returns:
            backcast: [batch, units]
            forecast: [batch, horizon]
        """
        batch_size = x.size(0)
        
        # FC1: 提取
        hidden = F.relu(self.fc1(x))
        
        # FC2: 基系数
        theta = self.fc2(hidden)  # [batch, basis_dim]
        
        if self.is_trend:
            # 趋势：多项式基函数
            H = theta.size(1)
            basis = self.basis[:H]  # [H, degree+1]
            forecast = torch.matmul(theta, basis.t())  # [batch, H]
        else:
            # 周期：待实现
            forecast = theta
            
        # FC3: backcast
        backcast = self.fc3(theta)
        
        return backcast, forecast


class NBEATSModel(nn.Module):
    """N-BEATS模型"""
    
    def __init__(self, input_dim=100, output_dim=24, hidden=128, n_blocks=3):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Linear(input_dim, hidden)
        
        # 趋势栈
        self.trend_blocks = nn.ModuleList([
            NBEATSBlock(hidden, 32, degree=3, is_trend=True)
            for _ in range(n_blocks)
        ])
        
        # 周期栈
        self.seasonal_blocks = nn.ModuleList([
            NBEATSBlock(hidden, 32, degree=3, is_trend=False)
            for _ in range(n_blocks)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(hidden, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]
        Returns:
            forecast: [batch, output_dim]
        """
        batch_size = x.size(0)
        
        # 嵌入
        x = self.embedding(x)
        
        # 趋势栈
        trend_forecast = 0
        for block in self.trend_blocks:
            backcast, forecast = block(x)
            x = x - backcast  # 残差
            trend_forecast = trend_forecast + forecast
            
        # 周期栈
        season_forecast = 0
        for block in self.seasonal_blocks:
            backcast, forecast = block(x)
            x = x - backcast
            season_forecast = season_forecast + forecast
            
        # 叠加
        forecast = trend_forecast + season_forecast
        
        return forecast


# 训练
def train_nbeats():
    """训练示例"""
    
    model = NBEATSModel(input_dim=100, output_dim=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 数据
    lookback = 100
    horizon = 24
    X = np.random.randn(1000, lookback)
    Y = np.random.randn(1000, horizon)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # 训练循环
    for epoch in range(10):
        total_loss = 0
        for x, y in loader:
            out = model(x)
            loss = F.mse_loss(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")


if __name__ == "__main__":
    # 测试
    x = torch.randn(4, 100)
    model = NBEATSModel(input_dim=100, output_dim=24)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
```

---

## 8. 手工代码实现

### 8.1 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


class NBEATSFull(nn.Module):
    """完整N-BEATS模型"""
    
    def __init__(self, 
                 lookback=100, 
                 horizon=24,
                 hidden=256, 
                 trend_blocks=2, 
                 seasonal_blocks=2,
                 degree=3):
        super().__init__()
        
        self.lookback = lookback
        self.horizon = horizon
        
        # 共享嵌入
        self.shared_fc = nn.Sequential(
            nn.Linear(lookback, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # 趋势栈
        self.trend_stack = nn.ModuleList([
            TrendBlock(hidden, horizon, degree)
            for _ in range(trend_blocks)
        ])
        
        # 周期栈
        self.seasonal_stack = nn.ModuleList([
            SeasonalBlock(hidden, horizon)
            for _ in range(seasonal_blocks)
        ])
        
    def forward(self, x):
        """前向传播"""
        
        # 嵌入
        h = self.shared_fc(x)
        
        # 趋势栈
        trend_fc = 0
        for block in self.trend_stack:
            backcast, fc = block(h)
            h = h - backcast
            trend_fc = trend_fc + fc
        
        # 周期栈
        season_fc = 0
        for block in self.seasonal_stack:
            backcast, fc = block(h)
            h = h - backcast
            season_fc = season_fc + fc
        
        # 叠加
        forecast = trend_fc + season_fc
        
        return forecast
    
    def predict(self, history):
        """预测未来"""
        
        self.eval()
        with torch.no_grad():
            x = torch.tensor(history[-self.lookback:], dtype=torch.float32)
            x = x.unsqueeze(0)  # batch维度
            pred = self.forward(x)
            
        return pred.squeeze().numpy()


class TrendBlock(nn.Module):
    """趋势Block（多项式基函数）"""
    
    def __init__(self, units, horizon, degree=3):
        super().__init__()
        
        self.units = units
        self.horizon = horizon
        self.degree = degree
        
        # FC层
        self.fc1 = nn.Linear(units, units)
        self.fc2 = nn.Linear(units, horizon)
        self.fc3 = nn.Linear(units, units)
        
    def forward(self, x):
        """前向传播"""
        
        # 提取
        h = F.relu(self.fc1(x))
        
        # 系数
        theta = self.fc2(h)  # [batch, horizon]
        
        # 多项式基函数
        t = torch.arange(self.horizon, dtype=torch.float32, device=x.device)
        basis = torch.stack([t ** i for i in range(self.degree + 1)], dim=1)  # [horizon, degree+1]
        
        # 预测
        forecast = theta @ basis.t()  # [batch, horizon]
        
        # backcast
        backcast = self.fc3(theta)
        
        return backcast, forecast


class SeasonalBlock(nn.Module):
    """周期Block（傅里叶基函数）"""
    
    def __init__(self, units, horizon, n_fourier=5):
        super().__init__()
        
        self.units = units
        self.horizon = horizon
        self.n_fourier = n_fourier
        
        # FC层
        self.fc1 = nn.Linear(units, units)
        self.fc_fourier = nn.Linear(units, n_fourier * 2)
        self.fc3 = nn.Linear(units, units)
        
    def forward(self, x):
        """前向传播"""
        
        # 提取
        h = F.relu(self.fc1(x))
        
        # 傅里叶系数
        theta = self.fc_fourier(h)  # [batch, n_fourier*2]
        
        # 傅里叶基函数
        t = torch.arange(self.horizon, dtype=torch.float32, device=x.device)
        
        forecasts = []
        for k in range(self.n_fourier):
            sin = torch.sin(2 * np.pi * (k + 1) * t / 12)
            cos = torch.cos(2 * np.pi * (k + 1) * t / 12)
            forecasts.extend([sin, cos])
        
        basis = torch.stack(forecasts, dim=1)  # [horizon, n_fourier*2]
        
        # 预测
        forecast = theta @ basis.t()  # [batch, horizon]
        
        # backcast
        backcast = self.fc3(theta)
        
        return backcast, forecast


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, series, lookback, horizon):
        self.series = series
        self.lookback = lookback
        self.horizon = horizon
        
    def __len__(self):
        return len(self.series) - self.lookback - self.horizon + 1
    
    def __getitem__(self, idx):
        x = self.series[idx:idx+self.lookback]
        y = self.series[idx+self.lookback:idx+self.lookback+self.horizon]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_nbeats_model(series, epochs=100, lookback=100, horizon=24):
    """训练N-BEATS模型"""
    
    # 数据归一化
    mean = np.mean(series)
    std = np.std(series)
    series_norm = (series - mean) / std
    
    # 数据集
    dataset = TimeSeriesDataset(series_norm, lookback, horizon)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 模型
    model = NBEATSFull(lookback, horizon, hidden=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            pred = model(x)
            loss = F.mse_loss(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    return model, mean, std
```

---

## 9. 可视化与结果理解

### 9.1 预测可视化

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_forecast(historical, forecast, truth=None, title="N-BEATS Forecast"):
    """可视化预测结果"""
    
    plt.figure(figsize=(12, 6))
    
    # 历史
    plt.plot(range(len(historical)), historical, 
            label='Historical', color='blue')
    
    # 预测
    future_idx = range(len(historical), len(historical) + len(forecast))
    plt.plot(future_idx, forecast, 
            label='Forecast', color='red', linestyle='--')
    
    # 真实值
    if truth is not None:
        plt.plot(future_idx, truth, 
                label='Ground Truth', color='green')
    
    plt.axvline(x=len(historical), color='gray', linestyle=':')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# 示例
if __name__ == "__main__":
    # 生成数据
    t = np.arange(200)
    y = np.sin(t / 10) + np.random.randn(200) * 0.1
    
    # 预测
    model, mean, std = train_nbeats_model(y[:150])
    
    # 可视化
    history = y[:150]
    forecast_raw = model.predict(history[-100:])
    forecast = forecast_raw * std + mean
    
    visualize_forecast(history, forecast)
```

### 9.2 趋势/周期分解

```python
def visualize_components(model, history):
    """可视化趋势和周期分量"""
    
    model.eval()
    
    with torch.no_grad():
        x = torch.tensor(history[-100:], dtype=torch.float32).unsqueeze(0)
        
        # 嵌入
        h = model.shared_fc(x)
        
        # 趋势
        trend = 0
        for block in model.trend_stack:
            _, fc = block(h)
            trend = trend + fc.squeeze().numpy()
        
        # 周期
        season = 0
        for block in model.seasonal_stack:
            _, fc = block(h)
            season = season + fc.squeeze().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(trend, label='Trend')
    plt.plot(season, label='Seasonal')
    plt.legend()
    plt.title("Decomposition")
    plt.grid(True)
    plt.show()
```

### 9.3 误差分析

```python
def error_analysis(predictions, truths):
    """误差分析"""
    
    errors = np.abs(truths - predictions)
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(errors)), errors)
    plt.xlabel('Horizon Step')
    plt.ylabel('Absolute Error')
    plt.title('Prediction Error by Horizon')
    plt.grid(True)
    plt.show()
    
    print(f"MAE: {np.mean(errors):.4f}")
    print(f"MSE: {np.mean(errors**2):.4f}")
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| MAE | 平均绝对误差 |
| MSE | 均方误差 |
| MAPE | 百分比误差 |
| SMAPE | 对称百分比误差 |

### 10.2 M4竞赛基准

| 方法 | MAPE |
|------|------|
| N-BEATS | ~3.2% |
| Theta | ~3.5% |
| ARIMA | ~4.1% |
| ETS | ~4.0% |

### 10.3 评估代码

```python
def evaluate_model(predictions, truths):
    """评估模型"""
    
    mae = np.mean(np.abs(truths - predictions))
    mse = np.mean((truths - predictions) ** 2)
    mape = np.mean(np.abs((truths - predictions) / truths)) * 100
    
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return {'mae': mae, 'mse': mse, 'mape': mape}
```

---

## 11. 常见问题与易错点

### 11.1 数据不足

**问题**：训练数据太少

**解决**：
- 使用数据归一化
- 使用预训练+微调
- 减小模型复杂度

### 11.2 远期预测不准

**问题**：预测步长太远时误差大

**解决**：
- 级联预测（短期→中期→长期）
- 增大lookback
- 分层损失函数

### 11.3 过拟合

**问题**：训练损失低但测试损失高

**解决**：
- Dropout
- 早停
- 权重衰减

### 11.4 周期设置

**问题**：不知道数据的周期

**解决**：
- 探索性分析（ACF/PACF）
- 使用傅里叶项自动学习

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 本质 | 神经网络自动学习趋势+周期 |
| 核心 | 双栈分解 + 基函数展开 |
| 创新 | 无假设、端到端 |
| 优势 | 可解释、M4竞赛领先 |

### 12.2 公式记忆

$$\hat{y} = \text{Trend Stack}(x) + \text{Seasonal Stack}(x)$$

### 12.3 扩展阅读

| 论文 | 年份 | 贡献 |
|------|------|------|
| N-BEATS | 2020 | 原始论文 |
| N-HiTS | 2023 | 时间注意力 |
| TFT | 2020 | 时序Transformer |

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：N-BEATS为什么不需要假设数据平稳？

**答案**：因为它用神经网络自动学习数据中的趋势和周期，不需要像ARIMA那样手工差分。

**练习2**：趋势栈和周期栈的区别？

**答案**：趋势栈学习低频变化（趋势），使用多项式基；周期栈学习高频变化（季节），使用傅里叶基。

**练习3**：backcast的作用？

**答案**：backcast用于学习，让模型拟合输入历史，然后通过残差链接让下一个block专注于学习残余模式。

### 13.2 进阶思考

**思考1**：如何选择lookback和horizon？

**提示**：根据预测任务（短期/中期/长期）、数据量、计算资源。

**思考2**：N-BEATS和LSTM的区别？

**提示**：N-BEATS是直接多步预测，LSTM是迭代预测。

**思考3**：为什么远期预测更难？

**提示**：误差累积、信息衰减。

### 13.3 编程练习

**练习**：实现一个销量预测系统

```python
# 要求：
# 1. 准备销量数据
# 2. 训练N-BEATS
# 3. 预测未来24步
# 4. 评估误差
```

---

## 14. 学习路径建议

### 14.1 入门（1周）

| 天 | 内容 | 目标 |
|----|------|------|
| 1-2 | 时间序列基础 | 趋势/周期 |
| 3-4 | 深度学习基础 | FC/激活 |
| 5-6 | 基函数展开 | 多项式/傅里叶 |
| 7 | N-BEATS原理 | 双栈架构 |

### 14.2 进阶（2周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | N-BEATS实现 | 代码复现 |
| 2 | 训练优化 | 调参 |

### 14.3 实战（3周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | 数据准备 | 业务数据 |
| 2 | 模型训练 | 实际训练 |
| 3 | 部署 | API服务 |

---

## 附录

### A. 重要参考

| 参考 | 链接 |
|------|------|
| N-BEATS论文 | https://arxiv.org/abs/1905.10437 |
| Darts文档 | https://unit8co.github.io/darts/ |
| M4竞赛 | https://www.m4competition.com/ |

### B. 参数参考

| 参数 | 推荐值 |
|------|--------|
| lookback | 100 |
| horizon | 24 |
| hidden | 256 |
| epoch | 100 |

### C. 代码资源

```python
# 推荐项目
# 1. Darts: NBEATSModel
# 2. pytorch-forecasting: NBEATS
# 3. GluonTS: NBEATS
```

---

**文档结束**