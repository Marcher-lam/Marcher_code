# Prophet 学习文档

> Facebook开源的时间序列预测框架，处理带有周期性的业务时间序列

---

## 1. 算法基础认知

### 1.1 一句话定义

Prophet是Facebook开发的时间序列预测框架，专门处理带有趋势和周期性的业务时间序列数据，能够自动检测周期变化点并适应holidays。

### 1.2 直觉类比

Prophet就像一位经验丰富的业务分析师——它会分析数据的整体趋势（是增长还是下降）、周期性（每周、每月、每年）、以及特殊事件（节假日）的影响，最后给出一个可靠的预测区间！

想象你是零售店经理：
- 看数据：每天的销售额
- Prophet分析：整体在增长（趋势）、周末生意好（周期）、过年生意特别好（节假日）
- 输出：不仅预测明天卖多少，还能告诉你"下周一是情人节，可能销量会下降5%"

### 1.3 发展背景

- 2017年，Facebook的Sean Taylor和Ben Letham在论文"Forecasting at Scale"中提出Prophet
- 发布后在Kaggle比赛中广泛应用，成为商业时间序列预测的流行工具
- 2020年成为Netflix等公司的核心技术

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 时间序列 → 预测 |
| 输出 | 未来趋势+置信区间 |
| 模型 | 加法分解模型 |
| 特点 | 自动周期检测 |

---

## 2. 核心原理

### 2.1 核心思想

Prophet的核心思想是**将时间序列分解为趋势+周期+节假日三个部分的加法模型**：

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

其中：
- $g(t)$：整体趋势（growth）
- $s(t)$：周期性成分（seasonality）
- $h(t)$：假期影响（holidays）
- $\epsilon_t$：噪声

核心思想可以概括为：**分而治之，分别建模后叠加**

### 2.2 工作流程

1. **数据输入**：带有时间戳的数值序列
2. **趋势检测**：自动检测趋势变化点
3. **周期建模**：傅里叶级数表示周期
4. **假期效应**：假期列表
5. **后验分布**：使用Stan进行贝叶斯推断

### 2.3 关键概念

- **Changepoint**：趋势变化点
- **Seasonality**：周期性（周，月、年）
- **Holiday Effects**：假期效应
- **Uncertainty**：不确定性区间

---

## 3. 数学公式与推导

### 3.1 趋势模型

$$g(t) = (a + b \cdot t) \prod_i \left(1 + \frac{t - s_i}{D_i}\right)^{+} $$

其中$a$是截距，$b$是增长率，$s_i$是变化点。

### 3.2 季节性

$$s(t) = \sum_{n=1}^{N} \left(a_n \cos(2\pi n t / P) + b_n \sin(2\pi n t / P)\right)$$

傅里叶级数表示，$P$是周期（周=7，月=30，年=365）。

### 3.3 假期效应

$$h(t) = \sum_{j} h_j(t)$$

每个假期j有对应的参数。

### 3.4 损失函数

使用最大后验估计（MAP）：
$$\log p(y|g,s,h) = \sum_t \log p(y_t|g(t),s(t),h(t)) + \log p(g) + \log p(s) + \log p(h)$$

---

## 4. 训练过程讲解

### 4.1 数据准备

```python
# 数据格式要求
df = pd.DataFrame({
    'ds': ['2023-01-01', '2023-01-02', ...],  # 日期
    'y': [100, 120, ...]  # 数值
})
```

### 4.2 趋势参数

| 参数 | 说明 | 建议值 |
|------|------|---------|
| changepoint_prior_scale | 趋势灵活度 | 0.05-0.5 |
| n_changepoints | 变化点数量 | 25 |
| changepoint_range | 变化点范围 | 0.8 |

### 4.3 周期参数

| 参数 | 说明 | 建议值 |
|------|------|---------|
| seasonality_mode | 加法/乘法 | 加法 |
| seasonality_prior_scale | 周期灵活度 | 10 |
| yearly_seasonality | 年周期 | True |
| weekly_seasonality | 周周期 | True |

### 4.4 假期参数

```python
# 假期数据
holidays = pd.DataFrame({
    'holiday': ['christmas', 'thanksgiving'],
    'ds': ['2023-12-25', '2023-11-23'],
    'lower_window': [0, 0],
    'upper_window': [1, 1],
})
```

---

## 5. 应用场景

### 5.1 典型应用

- **业务预测**：销量、营收
- **用户增长**：DAU/MAU预测
- **流量预测**：网站访问量
- **需求预测**：供应链

### 5.2 业务案例

```python
# 零售销量预测
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    holidays=holidays,
    changepoint_prior_scale=0.1
)

# 加入额外回归变量
model.add_regressor('temperature')
model.add_regressor('promotion')
```

### 5.3 对比其他方法

| 方法 | 适用场景 | 优点 | 缺点 |
|------|-----------|------|------|
| ARIMA | 平稳序列 | 理论完善 | 需要平稳 |
| LSTM | 复杂非线性 | 表达强 | 需要大量数据 |
| Prophet | 业务时序 | 简单+可解释 | 不适合高频 |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **易用** | 自动检测周期，无需特征工程 |
| **可解释** | 分解可视化 |
| **鲁棒** | 处理缺失和异常 |
| **快速** | 秒级训练 |
| **贝叶斯** | 提供不确定性区间 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| **不适合高频** | 日以上频率 |
| **不适合短序列** | 至少几个月数据 |
| **对突变反应慢** | 变化点检测有延迟 |

### 6.3 注意事项

- 数据至少包含一个完整周期（建议1年以上）
- 节假日需要手动指定
- 异常值需要预处理

---

## 7. 调库实现（Python）

### 7.1 安装

```bash
pip install prophet plotly
```

### 7.2 完整代码

```python
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt


def prepare_prophet_data(df, date_col, value_col):
    """准备Prophet格式数据"""
    return pd.DataFrame({
        'ds': df[date_col],
        'y': df[value_col]
    })


def train_prophet(df_train, periods=30, yearly_seasonality=True, 
                 weekly_seasonality=True, holidays=None):
    """训练并预测"""
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )
    
    model.fit(df_train)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return model, forecast


def plot_components(model, forecast):
    """绘制分解图"""
    fig = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig('prophet_components.png')
    plt.show()


def evaluate_forecast(df_test, forecast, value_col):
    """评估预测"""
    y_pred = forecast.tail(len(df_test))['yhat'].values
    y_true = df_test[value_col].values
    
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return mae, rmse


if __name__ == "__main__":
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    trend = np.linspace(10, 50, 365)
    seasonal = 5 * np.sin(np.arange(365) * 2 * np.pi / 365)
    noise = np.random.randn(365) * 2
    
    df = pd.DataFrame({
        'date': dates,
        'value': trend + seasonal + noise
    })
    
    df_prophet = prepare_prophet_data(df, 'date', 'value')
    model, forecast = train_prophet(df_prophet, periods=30)
    
    print("预测结果:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
```

---

## 8. 手工代码实现（理解原理）

```python
import numpy as np

class SimpleProphet:
    """简化版Prophet - 理解原理"""
    
    def __init__(self, yearly_period=365, weekly_period=7):
        self.yearly_period = yearly_period
        self.weekly_period = weekly_period
    
    def fit(self, dates, values):
        self.dates = dates
        self.values = values
        
        # 简化：线性趋势
        self.trend_slope = (values[-1] - values[0]) / len(values)
        self.trend_intercept = values[0]
        
        # 简化：傅里叶周期
        self.yearly_coef = self._fit_fourier(values, self.yearly_period)
        self.weekly_coef = self._fit_fourier(values, self.weekly_period)
        
        return self
    
    def _fit_fourier(self, values, period):
        """简化的傅里叶系数估计"""
        n = len(values)
        t = np.arange(n)
        coefs = []
        for k in [1, 2]:
            a = 2/n * np.sum(values * np.cos(2*np.pi*k*t/period))
            b = 2/n * np.sum(values * np.sin(2*np.pi*k*t/period))
            coefs.append((a, b))
        return coefs
    
    def predict(self, future_dates):
        """预测"""
        predictions = []
        
        for i, d in enumerate(future_dates):
            t = len(self.dates) + i
            
            # 趋势
            trend = self.trend_intercept + self.trend_slope * t
            
            # 年周期
            yearly = sum(a * np.cos(2*np.pi*(k+1)*t/self.yearly_period) + 
                        b * np.sin(2*np.pi*(k+1)*t/self.yearly_period)
                        for k, (a, b) in enumerate(self.yearly_coef))
            
            # 周周期
            weekly = sum(a * np.cos(2*np.pi*(k+1)*t/self.weekly_period) + 
                       b * np.sin(2*np.pi*(k+1)*t/self.weekly_period)
                       for k, (a, b) in enumerate(self.weekly_coef))
            
            predictions.append(trend + yearly + weekly)
        
        return np.array(predictions)


if __name__ == "__main__":
    import pandas as pd
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=180, freq='D')
    trend = np.linspace(10, 30, 180)
    seasonal = 3 * np.sin(np.arange(180) * 2 * np.pi / 365)
    noise = np.random.randn(180) * 1
    values = trend + seasonal + noise
    
    model = SimpleProphet()
    model.fit(dates, values)
    
    future_dates = pd.date_range('2023-07-01', periods=30, freq='D')
    predictions = model.predict(future_dates)
    
    print("预测30天:")
    print(predictions[:5])
```

---

## 9. 可视化与结果理解

### 9.1 趋势分解可视化

```python
import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd
import numpy as np

# 数据
np.random.seed(42)
df = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', periods=365),
    'y': np.linspace(10, 50, 365) + 5*np.sin(np.arange(365)*2*np.pi/365) + np.random.randn(365)*2
})

# 训练
model = Prophet()
model.fit(df)

# 预测
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 原始数据+预测
ax = axes[0, 0]
ax.plot(df['ds'], df['y'], 'b-', label='历史数据')
ax.plot(forecast['ds'], forecast['yhat'], 'r-', label='预测')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
               alpha=0.3, color='red', label='置信区间')
ax.legend()
ax.set_title('预测结果')

# 2. 趋势
ax = axes[0, 1]
ax.plot(forecast['ds'], forecast['trend'], 'g-')
ax.set_title('趋势成分')

# 3. 年周期
ax = axes[1, 0]
ax.plot(forecast['ds'], forecast['yearly'], 'orange')
ax.set_title('年周期')

# 4. 周周期
ax = axes[1, 1]
ax.plot(forecast['ds'], forecast['weekly'], 'purple')
ax.set_title('周周期')

plt.tight_layout()
plt.savefig('prophet_analysis.png', dpi=100)
plt.show()
```

### 9.2 变化点检测

```python
# 可视化变化点
fig = model.plot(forecast)
for cp in model.changepoints:
    plt.axvline(cp, color='red', linestyle='--', alpha=0.5)
plt.title('趋势变化点')
plt.savefig('changepoints.png', dpi=100)
plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| MAE | 平均绝对误差 |
| RMSE | 均方根误差 |
| MAPE | 平均绝对百分比误差 |
| 覆盖率 | 真实值在区间内比例 |

### 10.2 评估代码

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_prophet(df_test, forecast, value_col):
    """评估Prophet"""
    y_pred = forecast.tail(len(df_test))['yhat'].values
    y_true = df_test[value_col].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 覆盖率
    lower = forecast.tail(len(df_test))['yhat_lower'].values
    upper = forecast.tail(len(df_test))['yhat_upper'].values
    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"覆盖率: {coverage:.1f}%")
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'Coverage': coverage}
```

---

## 11. 常见问题与易错点

### Q1: 数据需要什么格式？

**答案**：Prophet要求两列：ds（日期）和y（数值）。

### Q2: 预测精度不高？

**答案**：检查是否有周期性未被捕捉，可能需要添加自定义周期或假期。

### Q3: 置信区间太宽/窄？

**答案**：调整interval_width参数，或修改changepoint_prior_scale。

### Q4: 趋势变化点太多/少？

**答案**：调整n_changepoints或changepoint_prior_scale。

### Q5: 为什么周期不明显？

**答案**：数据可能太短，至少需要一年的数据。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心公式 | y(t) = g(t) + s(t) + h(t) + ε |
| 趋势 | 分段线性+变化点 |
| 周期 | 傅里叶级数 |
| 假期 | 手动指定 |

### 12.2 公式汇总

加法模型：
$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

趋势：
$$g(t) = (a + bt) \prod_i (1 + \frac{t-s_i}{D_i})^+$$

周期：
$$s(t) = \sum_n (a_n \cos + b_n \sin)$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. Prophet的核心假设是：
   - A) 时间序列是加法模型
   - B) 时间序列是乘法模型
   - C) 时间序列是指数模型

2. 年周期用傅里叶级数表示时，P等于：
   - A) 7
   - B) 30
   - C) 365

### 13.2 简答题

1. 解释Prophet如何检测趋势变化点？
2. 比较Prophet和ARIMA的适用场景。

### 13.3 编程题

1. 使用Prophet预测你所在城市的温度。
2. 添加中国节假日并预测零售销量。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
时间序列基础
    ↓
趋势分解
    ↓
ARIMA
    ↓
Prophet原理
    ↓
LSTM时序
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| ARIMA | 经典时序 |
| LSTM | 深度学习时序 |
| ExponentialSmoothing | 指数平滑 |

### 14.3 扩展阅读

- Taylor & Letham (2017). Forecasting at Scale. Facebook Research.

---

## 附录

### 参考

1. Taylor & Letham (2017). Forecasting at Scale. arXiv:1708.05897
2. https://facebook.github.io/prophet/
3. https://github.com/facebook/prophet

---

**文档结束**