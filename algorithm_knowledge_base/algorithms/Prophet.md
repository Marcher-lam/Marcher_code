# Prophet 学习文档

> Facebook开源的时间序列预测框架，处理带有周期性的业务时间序列

---

## 1. 算法基础认知

**一句话定义**：Prophet是Facebook开发的时间序列预测框架，专门处理带有趋势和周期性的业务时间序列数据，能够自动检测周期变化点并适应 holidays。

**直觉类比**：Prophet就像一位经验丰富的业务分析师——它会分析数据的整体趋势（是增长还是下降）、周期性（每周、每月、每年）、以及特殊事件（节假日）的影响，最后给出一个可靠的预测区间。

**历史背景**：2017年，Facebook的Sean Taylor和Ben Letham在论文"Forecasting at Scale"中提出Prophet。发布后在Kaggle比赛中广泛应用，也成为商业时间序列预测的流行工具。

**算法定位**：
- 类型：时间序列 → 预测
- 输出：未来趋势和置信区间
- 模型类型：加法模型（分解模型）

**前置知识**：
- [必备]：时间序列基础
- [必备]：时间序列分解
- [扩展]：ARIMA、指数平滑

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

核心思想可以概括为：**分而治之，分别建模后叠加**。

### 2.2 工作流程

1. **数据输入**：带有时间戳的数值序列
2. **趋势检测**：自动检测趋势变化点
3. **周期建模**：傅里叶级数表示周期
4. **假期效应**：假期列表
5. **后验分布**：使用Stan进行贝叶斯推断

### 2.3 关键概念

- **Changepoint**：趋势变化点
- **Seasonality**：周期性（周、月、年）
- **Holiday Effects**：假期效应
- **Uncertainty**：不确定性区间

---

## 3. 数学公式

### 3.1 趋势模型

$$g(t) = (a + b \cdot t) \prod_i \left(1 + \frac{t - s_i}{D_i}\right)^{+} $$

其中$a$是截距，$b$是增长率。

### 3.2 季节性

$$s(t) = \sum_{n=1}^{N} \left(a_n \cos(2\pi n t / P) + b_n \sin(2\pi n t / P)\right)$$

傅里叶级数表示，$P$是周期（周=7，月=30，年=365）。

### 3.3 假期效应

$$h(t) = \sum_{j} h_j(t)$$

每个假期j有对应的参数。

---

## 4. 调库实现

### 4.1 环境

```bash
pip install prophet plotly
```

### 4.2 完整代码

```python
"""
Prophet 时间序列预测
"""

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


# 数据准备
def prepare_data(df, date_col, value_col, freq='D'):
    """准备Prophet格式数据"""
    df_prophet = pd.DataFrame({
        'ds': df[date_col],
        'y': df[value_col]
    })
    return df_prophet


# 训练预测
def train_predict(df_train, periods=30, yearly_seasonality=True, 
                 weekly_seasonality=True, holidays=None):
    """训练并预测"""
    
    # 创建模型
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=0.05,  # 趋势灵活度
        seasonality_prior_scale=10,      # 周期灵活度
    )
    
    # 训练
    model.fit(df_train)
    
    # 预测
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return model, forecast


# 可视化
def plot_components(model, forecast):
    """绘制分解图"""
    fig = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig('prophet_components.png')
    plt.show()


# 评估
def evaluate(df_test, forecast, value_col):
    """评估预测"""
    # 取预测值
    y_pred = forecast.tail(len(df_test))['yhat'].values
    y_true = df_test[value_col].values
    
    # 指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return mae, rmse


# 示例
if __name__ == "__main__":
    # 示例数据
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    trend = np.linspace(10, 50, 365)
    seasonal = 5 * np.sin(np.arange(365) * 2 * np.pi / 365)
    noise = np.random.randn(365) * 2
    
    df = pd.DataFrame({
        'date': dates,
        'value': trend + seasonal + noise
    })
    
    print("训练Prophet模型...")
    model, forecast = train_predict(df, periods=30)
    
    print("\n预测结果:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    print("\n✓ 程序执行完毕")
```

---

## 5. 应用场景

### 5.1 典型应用

- **业务预测**：销量、营收
- **用户增长**：DAU/MAU预测
- **流量预测**：网站访问量
- **需求预测**：供应链

### 5.2 参数

| 参数 | 说明 | 默认值 |
|------|------|----------|
| yearly | 年周期 | True |
| weekly | 周周期 | True |
| changepoint_prior | 趋势灵活度 | 0.05 |
| seasonality_prior | 周期灵活度 | 10 |
| holidays | 假期列表 | None |

---

## 6. 优缺点

### 6.1 优点

- 易用：自动检测周期
- 可解释：分解可视化
- 鲁棒：处理缺失和异常
- 快速：秒级训练

### 6.2 缺点

- 不适合高频数据
- 不适合短序列
- 对突变不如ARIMA

---

## 7. 练习

**问题**：什么时候使用Prophet？

答案：业务时间序列、有明显周期、有节假日效应。

---

## 8. 学习路径

### 8.1 前置

- [ ] 时间序列基础

### 8.2 进阶

- [ ] ARIMA
- [ ] LSTM时序

---

## 附录

### A. 完整代码

见第4节。

### B. 参考

1. Taylor & Letham, "Forecasting at Scale", 2017

---

**文档结束**