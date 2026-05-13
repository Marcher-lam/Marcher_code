# ARIMA 学习文档

> 时间序列预测的标准方法，自回归差分移动平均模型。

---

## 1. 算法基础认知

### 1.1 发展背景

ARIMA（AutoRegressive Integrated Moving Average，自回归差分移动平均模型）由 Box 和 Jenkins 于 1970 年在《Time Series Analysis: Forecasting and Control》一书中正式提出，是时间序列分析领域最具影响力的统计方法。ARIMA 模型及其变体（如 SARIMA、ARIMAX）在经济学、金融学、气象学、工业等领域有着广泛应用。

### 1.2 核心定位

ARIMA 模型是经典的时间序列预测方法，通过以下三个组件建模时间序列的动态特性：

| 组件 | 全称 | 含义 |
|------|------|------|
| AR | AutoRegressive | 自回归，利用自身历史值预测当前值 |
| I | Integrated | 差分，使序列平稳化 |
| MA | Moving Average | 移动平均，平滑随机波动 |

### 1.3 模型表示

ARIMA 模型通常记为 $\text{ARIMA}(p, d, q)$，其中：

- $p$：自回归阶数
- $d$：差分阶数
- $q$：移动平均阶数

例如，$\text{ARIMA}(2, 1, 1)$ 表示二阶自回归、一阶差分、一阶移动平均。

---

## 2. 核心原理

### 2.1 平稳性概念

**严平稳**：对所有 $t_1, t_2, ..., t_n$ 和 $\tau$，$(X_{t_1}, ..., X_{t_n})$ 与 $(X_{t_1+\tau}, ..., X_{t_n+\tau})$ 同分布。

**宽平稳**（二阶平稳）：
$$E[X_t] = \mu, \quad \text{Var}[X_t] = \sigma^2, \quad \text{Cov}(X_t, X_{t-k}) = \gamma_k$$

平稳序列的统计特性不随时间变化，是 ARIMA 建模的基础假设。

### 2.2 非平稳的类型

**趋势非平稳**：存在线性或非线性趋势
$$X_t = \mu + \alpha t + \epsilon_t$$

**季节性非平稳**：存在周期性模式
$$X_t = S_{t mod s} + \epsilon_t$$

**单位根非平稳**：特征根等于 1

### 2.3 差分运算

**一阶差分**：
$$\Delta X_t = X_t - X_{t-1}$$

**d 阶差分**：
$$\Delta^d X_t = \Delta(\Delta^{d-1} X_t)$$

差分可以将非平稳序列转化为平稳序列，是 ARIMA 中"I"的含义。

### 2.4 自回归（AR）

$p$ 阶自回归模型：
$$X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + ... + \phi_p X_{t-p} + \epsilon_t$$

或写成：
$$X_t = \sum_{i=1}^p \phi_i X_{t-i} + \epsilon_t$$

其中 $\epsilon_t \sim WN(0, \sigma^2)$ 为白噪声。

### 2.5 移动平均（MA）

$q$ 阶移动平均模型：
$$X_t = \epsilon_t - \theta_1 \epsilon_{t-1} - \theta_2 \epsilon_{t-2} - ... - \theta_q \epsilon_{t-q}$$

或写成：
$$X_t = \epsilon_t - \sum_{j=1}^q \theta_j \epsilon_{t-j}$$

---

## 3. 数学公式与推导

### 3.1 ARIMA 模型定义

$\text{ARIMA}(p, d, q)$ 模型定义为：
$$\Delta^d X_t = \sum_{i=1}^p \phi_i \Delta^d X_{t-i} + \epsilon_t - \sum_{j=1}^q \theta_j \epsilon_{t-j}$$

其中 $\Delta^d X_t$ 是 $d$ 阶差分后的平稳序列。

### 3.2 滞后算子

引入滞后算子 $B$：
$$B X_t = X_{t-1}$$

则差分可以表示为：
$$\Delta = 1 - B$$
$$\Delta^d = (1 - B)^d$$

AR 模型：
$$\phi(B) X_t = \epsilon_t$$

其中 $\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - ... - \phi_p B^p$ 为自回归多项式。

MA 模型：
$$X_t = \theta(B) \epsilon_t$$

其中 $\theta(B) = 1 - \theta_1 B - \theta_2 B^2 - ... - \theta_q B^q$ 为移动平均多项式。

### 3.3 ARIMA 的一般形式

$$\phi(B) \Delta^d X_t = \theta(B) \epsilon_t$$

展开后：
$$\Delta^d X_t = \sum_{i=1}^p \phi_i \Delta^d X_{t-i} + \epsilon_t - \sum_{j=1}^q \theta_j \epsilon_{t-j}$$

### 3.4 ��征方程与平稳性

**AR 特征方程**：
$$1 - \phi_1 z - \phi_2 z^2 - ... - \phi_p z^p = 0$$

**平稳条件**：所有特征根的模小于 1（在单位圆内）。

**MA 特征方程**：
$$1 - \theta_1 z - \theta_2 z^2 - ... - \theta_q z^q = 0$$

**可逆条件**：所有特征根的模小于 1。

### 3.5 参数估计

最大似然估计（MLE）是最常用的参数估计方法。给定观测 $(X_1, ..., X_T)$，对数似然函数：
$$\log L(\phi, \theta, \sigma^2) = -\frac{T}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{t=1}^T \epsilon_t^2$$

其中 $\epsilon_t$ 由模型递推计算。

### 3.6 置信区间

大样本下，参数估计近似正态分布：
$$\hat{\phi} \approx N(\phi, \sigma_{\hat{\phi}}^2)$$

$95\%$ 置信区间：
$$\hat{\phi} \pm 1.96 \times \text{SE}(\hat{\phi})$$

---

## 4. 训练过程讲解

### 4.1 Box-Jenkins 方法

Box-Jenkins 方法包括以下步骤：

```
1. 可视化数据，识别非平稳类型
2. 对非平稳序列进行差分
3. 识别差分阶数 d
4. 分析 ACF 和 PACF，确定 p, q
5. 估计模型参数
6. 检验残差是否为白噪声
7. 如果不满足，返回步骤 2 或 3 调整
8. 预测和 forecast
```

### 4.2 差分阶数识别

**单位根检验（ADF 检验）**：

原假设 $H_0$：存在单位根（序列非平稳）

检验统计量：
$$\tau = \frac{\hat{\rho}}{\text{SE}(\hat{\rho})}$$

其中 $\hat{\rho}$ 为 $\Delta X_t = \rho X_{t-1} + \epsilon_t$ 的最小二乘估计。

**KPSS 检验**：

原假设 $H_0$：序列平稳（与 ADF 互补）。

### 4.3 模型阶数识别

**ACF（自相关函数）**：
$$\rho_k = \frac{\gamma_k}{\gamma_0} = \frac{\text{Cov}(X_t, X_{t-k})}{\text{Var}(X_t)}$$

**PACF（偏自相关函数）**：

在去除中间滞后影响后的自相关，衡量 $X_t$ 与 $X_{t-k}$ 的直接关系。

**阶数识别规则**：

| ACF | PACF | 模型 |
|-----|-----|------|
| 截尾 | p 阶截尾 | AR(p) |
| q 阶截尾 | 截尾 | MA(q) |
| 拖尾 | 拖尾 | ARIMA(p,d,q) |

### 4.4 AIC/BIC 准则

**AIC（Akaike Information Criterion）**：
$$\text{AIC} = -2 \log L + 2k$$

**BIC（Bayesian Information Criterion）**：
$$\text{BIC} = -2 \log L + k \log T$$

其中 $k = p + q + 1$ 为参数个数，$T$ 为样本量。

选择使 AIC/BIC 最小的模型阶数。

### 4.5 残差检验

**Ljung-Box 检验**：
$$Q = T(T+2) \sum_{k=1}^K \frac{\rho_k^2}{T-k}$$

原假设 $H_0$：残差为白噪声。

---

## 5. 应用场景

### 5.1 经济预测

- 股票价格预测
- GDP 增长率预测
- 通货膨胀率预测

### 5.2 销售预测

- 商品销量预测
- 季度销售额预测

### 5.3 气象预测

- 温度预测
- 降水量预测

### 5.4 代码示例

```python
import numpy as np
import pandas as pd

def arima_forecast(data, p=1, d=0, q=1, h=10):
    """ARIMA 预测示例
    
    参数:
        data: 时间序列数据
        p: 自回归阶数
        d: 差分阶数
        q: 移动平均阶数
        h: 预测步数
    """
    from statsmodels.tsa.arima.model import ARIMA
    
    # 差分
    if d > 0:
        data_diff = np.diff(data, n=d)
    else:
        data_diff = data
    
    # 拟合 ARIMA
    model = ARIMA(data, order=(p, d, q))
    results = model.fit()
    
    # 预测
    forecast = results.forecast(steps=h)
    
    return forecast
```

---

## 6. 优缺点分析

### 6.1 优点

1. **理论基础扎实**：基于统计理论，方法成熟
2. **可解释性强**：参数有明确含义
3. **计算简单**：无需迭代优化
4. **预测区间**：可生成置信区间

### 6.2 ��点

1. **线性假设**：只能捕捉线性关系
2. **参数敏感**：对 $p, d, q$ 选择敏感
3. **非自适应性**：需要人工确定阶数
4. **难处理复杂季节性**

### 6.3 改进方向

- **SARIMA**：加入季节性组件
- **ARIMAX**：加入外生变量
- **Auto-ARIMA**：自动选择阶数
- **Prophet**：Facebook 的分解方法

---

## 7. 调库实现

### 7.1 statsmodels 实现

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt

class ARIMAModel:
    """ARIMA 时间序列预测模型"""
    
    def __init__(self, order=(p, d, q)):
        self.order = order
        self.model = None
        self.results = None
        
    def fit(self, data, verbose=True):
        """
        拟合 ARIMA 模型
        
        参数:
            data: 一维时间序列
            order: (p, d, q) 元组
        """
        if isinstance(data, pd.Series):
            data = data.values
            
        self.model = ARIMA(data, order=self.order)
        self.results = self.model.fit()
        
        if verbose:
            print(self.results.summary())
            
        return self
    
    def forecast(self, steps=10, alpha=0.05):
        """
        预测未来值
        
        参数:
            steps: 预测步数
            alpha: 置信水平
        """
        forecast = self.results.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        conf = forecast.conf_int(alpha=alpha)
        
        return mean.values, conf.values
    
    def diagnose(self):
        """诊断检验"""
        # 残差检验
        residuals = self.results.resid
        
        # Ljung-Box 检验
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
        print("Ljung-Box 检验:")
        print(lb)
        
        # ADF 检验（残差平稳性）
        adf = adfuller(residuals)
        print(f"\nADF 统计量: {adf[0]:.4f}")
        print(f"p-value: {adf[1]:.4f}")
        
        return residuals


def demo_arima():
    """ARIMA 演示"""
    print("=== ARIMA 时间序列预测演示 ===\n")
    
    # 生成示例数据
    np.random.seed(42)
    n = 200
    
    # 模拟 ARIMA(1,1,1) 数据
    # X_t = 0.5*X_{t-1} + epsilon_t - 0.3*epsilon_{t-1}
    phi = 0.5
    theta = 0.3
    sigma = 1.0
    
    errors = np.random.randn(n) * sigma
    x = np.zeros(n)
    
    for t in range(1, n):
        x[t] = phi * x[t-1] + errors[t] - theta * errors[t-1]
    
    # 差分使序列平稳
    x_diff = np.diff(x)
    
    # ADF 检验
    adf_result = adfuller(x_diff)
    print(f"ADF 统计量: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print(f"结论: {'平稳' if adf_result[1] < 0.05 else '非平稳'}\n")
    
    # 拟合 ARIMA
    model = ARIMAModel(order=(1, 1, 1))
    model.fit(x[:180])
    
    # 预测
    pred, conf = model.forecast(steps=20)
    
    print(f"\n未来 20 步预测:")
    print(f"  第1步: {pred[0]:.4f}")
    print(f"  第10步: {pred[9]:.4f}")
    print(f"  第20步: {pred[19]:.4f}")
    
    # 诊断
    residuals = model.diagnose()
    
    return model


if __name__ == "__main__":
    demo_arima()
```

### 7.2 pmdarima 自动阶数选择

```python
# pip install pmdarima
from pmdarima import auto_arima

# 自动选择 ARIMA 阶数
model = auto_arima(
    data, 
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None,  # 自动差分
    trace=True,
    information_criterion='aic'
)

# 查看最优阶数
print(f"最优阶数: {model.order}")

# 预测
forecast = model.predict(n_periods=10)
```

---

## 8. 手工代码实现

### 8.1 完整 ARIMA 实现

```python
import numpy as np
from scipy import linalg

class ARIMAManual:
    """手动实现 ARIMA
    
    参数:
        p: 自回归阶数
        d: 差分阶数  
        q: 移动平均阶数
    """
    
    def __init__(self, p=1, d=0, q=1):
        self.p = p
        self.d = d
        self.q = q
        self.params = None
        self.residuals = None
        
    def difference(self, x, d):
        """d 阶差分"""
        if d == 0:
            return x
        return self.difference(np.diff(x), d-1)
    
    def fit(self, x):
        """
        使用 Yule-Walker 估计 AR 参数
        使用条件最小二乘估计 MA 参数
        """
        x = np.array(x)
        
        # 差分
        x_diff = self.difference(x, self.d)
        T = len(x_diff)
        
        # 估计 AR 系数（Yule-Walker）
        n = self.p + self.q
        
        # 构建自协方差矩阵
        gamma = np.zeros(n + 1)
        for k in range(n + 1):
            if k == 0:
                gamma[k] = np.var(x_diff)
            else:
                gamma[k] = np.cov(x_diff[:-k], x_diff[k:])[0, 1]
        
        # Yule-Walker 方程
        Gamma = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Gamma[i, j] = gamma[abs(i-j)]
        
        # 右端向量
        gamma_r = gamma[1:n+1]
        
        # 求解
        try:
            phi = linalg.solve(Gamma, gamma_r)
        except:
            phi = np.zeros(n)
        
        self.ar = phi[:self.p]
        self.ma = -phi[self.p:]
        self.residuals = np.zeros(T)
        
        # 计算残差
        for t in range(max(self.p, self.q), T):
            ar_term = np.sum(self.ar * x_diff[t-self.p:t][::-1]) if self.p > 0 else 0
            ma_term = np.sum(self.ma * self.residuals[t-self.q:t][::-1]) if self.q > 0 else 0
            self.residuals[t] = x_diff[t] - ar_term + ma_term
        
        # 估计方差
        self.sigma2 = np.var(self.residuals[max(self.p, self.q):])
        
        self.params = np.concatenate([self.ar, self.ma])
        
        return self
    
    def predict(self, x, h=1):
        """预测未来 h 步"""
        x = np.array(x)
        x_diff = self.difference(x, self.d)
        
        # 滚动预测
        forecast = np.zeros(h)
        x_extend = list(x_diff)
        
        for step in range(h):
            pred = 0
            
            # AR 部分
            if self.p > 0:
                for i in range(self.p):
                    if step - i - 1 >= 0:
                        pass
                    elif len(x_extend) - 1 - i >= 0:
                        pred += self.ar[i] * x_extend[-(1+i)]
            
            # MA 部分
            if self.q > 0:
                for j in range(self.q):
                    if len(x_extend) - 1 - j >= 0:
                        pred += self.ma[j] * self.residuals[-(1+j)]
            
            x_extend.append(pred)
            forecast[step] = pred
        
        return forecast
    
    def forecast_ci(self, x, h=1, alpha=0.05):
        """带置信区间的预测"""
        pred = self.predict(x, h)
        
        # 置信区间
        z = 1.96  # 95%
        half_width = z * np.sqrt(np.arange(h) * self.sigma2 + self.sigma2)
        
        lower = pred - half_width
        upper = pred + half_width
        
        return pred, lower, upper


def demo_manual():
    """手工实现演示"""
    print("=== ARIMA 手工���现���示 ===\n")
    
    # 生成模拟数据
    np.random.seed(42)
    n = 200
    
    # ARIMA(2,1,1) 数据
    x = np.zeros(n)
    errors = np.random.randn(n)
    
    phi = [0.5, -0.25]
    theta = 0.3
    
    for t in range(2, n):
        x[t] = phi[0]*x[t-1] + phi[1]*x[t-2] + errors[t] - theta*errors[t-1]
    
    # 拟合
    model = ARIMAManual(p=2, d=1, q=1)
    model.fit(x[:150])
    
    print(f"AR 系数: {model.ar}")
    print(f"MA 系数: {model.ma}")
    print(f"残差方差: {model.sigma2:.4f}")
    
    # 预测
    pred, lower, upper = model.forecast_ci(x[:150], h=20)
    
    print(f"\n预测结果:")
    print(f"  第1步: {pred[0]:.4f} [{lower[0]:.4f}, {upper[0]:.4f}]")
    print(f"  第10步: {pred[9]:.4f} [{lower[9]:.4f}, {upper[9]:.4f}]")
    print(f"  第20步: {pred[19]:.4f} [{lower[19]:.4f}, {upper[19]:.4f}]")


if __name__ == "__main__":
    demo_manual()
```

---

## 9. 可视化与结果理解

### 9.1 时间序列可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_timeseries():
    """时间序列可视化"""
    np.random.seed(42)
    t = np.arange(200)
    
    # 模拟数据
    x = np.cumsum(np.random.randn(200)) + 100
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 原始序列
    axes[0].plot(t, x)
    axes[0].set_title('原始时间序列')
    axes[0].grid(True, alpha=0.3)
    
    # 一阶差分
    x_diff = np.diff(x)
    axes[1].plot(t[1:], x_diff)
    axes[1].set_title('一阶差分序列')
    axes[1].grid(True, alpha=0.3)
    
    # 预测
    x_train = x[:180]
    x_test = x[180:]
    
    axes[2].plot(t[:180], x_train, label='训练')
    axes[2].plot(t[180:], x_test, label='真实')
    axes[2].plot(t[180:], np.full(20, x_train[-1]), label='预测', linestyle='--')
    axes[2].legend()
    axes[2].set_title('预测结果')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('arima_timeseries.png', dpi=150)
    plt.show()


def plot_acf_pacf():
    """ACF 和 PACF 可视化"""
    from statsmodels.tsa.stattools import acf, pacf
    
    np.random.seed(42)
    data = np.cumsum(np.random.randn(200))
    
    # 计算 ACF 和 PACF
    acf_values = acf(data, nlags=20)
    pacf_values = pacf(data, nlags=20)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # ACF
    axes[0].bar(range(21), acf_values)
    axes[0].set_title('自相关函数 (ACF)')
    axes[0].set_xlabel('滞后')
    axes[0].grid(True, alpha=0.3)
    
    # PACF
    axes[1].bar(range(21), pacf_values)
    axes[1].set_title('偏自相关函数 (PACF)')
    axes[1].set_xlabel('滞后')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('arima_acf_pacf.png', dpi=150)
    plt.show()
```

---

## 10. 模型评估

### 10.1 评估指标

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_arima(y_true, y_pred):
    """评估 ARIMA 预测结果"""
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
```

### 10.2 交叉验证

时间序列交叉验证需要注意数据的时序性：

```python
def timeseries_cv(data, model_class, n_splits=5):
    """时间序列交叉验证"""
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(data):
        model = model_class()
        model.fit(data[train_idx])
        pred = model.predict(data[train_idx], h=len(test_idx))
        
        score = np.sqrt(np.mean((data[test_idx] - pred) ** 2))
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

---

## 11. 常见问题与易错点

### 11.1 差分阶数选择

**问题**：如何确定差分阶数 $d$？

**解答**：
1. ADF 检验确定
2. 观察差分后的 ACF 是否快速衰减
3. $d$ 通常为 0, 1 或 2

### 11.2 季节性处理

**问题**：如何处理季节性数据？

**解答**：
1. 使用 SARIMA（Seasonal ARIMA）
2. 先做季节差分
3. 使用 STL 分解

### 11.3 模型不收敛

**问题**：模型拟合失败？

**解答**：
1. 增加差分阶数
2. 减少 $p, q$
3. 检查数据是否有异常值

---

## 12. 学习总结

**核心要点**：

1. **Box-Jenkins 方法**：识别 $\to$ 估计 $\to$ 检验 $\to$ 预测
2. **平稳性检验**：ADF 检验、KPSS 检验
3. **ACF/PACF 识别**：确定 $p, q$
4. **差分运算**：使非平稳序列平稳化

**学习建议**：

1. 掌握时间序列基本概念
2. 理解平稳性定义和检验
3. 实践 Box-Jenkins 全流程

---

## 13. 练习题与思考题

### 13.1 基础练习

1. 推导 AR(1) 模型的方差表达式
2. 证明差分运算可以使随机游走序列平稳化
3. 使用 ACF/PACF 识别模型阶数

### 13.2 进阶练习

1. 手动实现 ARIMA(1,1,1)
2. 进行时间序列交叉验证
3. 对比 ARIMA 与指数平滑

### 13.3 思考题

1. ARIMA 与机器学习方法的优势对比？
2. 如何处理高维时间序列？

---

### 13.4 详细答案与解析

#### 练习1：AR(1) 方差推导

**问题**：推导 AR(1) 模型 $X_t = \phi X_{t-1} + \epsilon_t$ 的方差。

**解答**：

两边取方差：
$$\text{Var}(X_t) = \text{Var}(\phi X_{t-1} + \epsilon_t)$$

由于 $X_{t-1}$ 与 $\epsilon_t$ 独立：
$$\text{Var}(X_t) = \phi^2 \text{Var}(X_{t-1}) + \text{Var}(\epsilon_t)$$

设 $\text{Var}(X_t) = \gamma_0$ 为常数：
$$\gamma_0 = \phi^2 \gamma_0 + \sigma_\epsilon^2$$

$$\gamma_0(1 - \phi^2) = \sigma_\epsilon^2$$

$$\gamma_0 = \frac{\sigma_\epsilon^2}{1 - \phi^2}$$

当 $|\phi| < 1$（平稳条件）时成立。

#### 练习2：差分与平稳性

**问题**：证明随机游走 $X_t = X_{t-1} + \epsilon_t$ 的一阶差分是平稳的。

**解答**：

随机游走增量：
$$\Delta X_t = X_t - X_{t-1} = \epsilon_t$$

由于 $\epsilon_t$ 是独立同分布的白噪声：
$$E[\Delta X_t] = E[\epsilon_t] = 0$$
$$\text{Var}(\Delta X_t) = \text{Var}(\epsilon_t) = \sigma^2$$
$$\text{Cov}(\Delta X_t, \Delta X_{t-k}) = 0 \quad (k > 0)$$

满足宽平稳条件，因此差分后是平稳序列。

#### 练习3：ACF/PACF 阶数识别

**问题**：对于 AR(p) 模型，PACF 有什么特征？

**解答**：

- AR(p) 模型的 PACF 在滞后 $p$ 之后截尾（等于 0）
- ACF 呈指数衰减或震荡拖尾

例如 AR(2)：
- PACF: $\phi_{22} \neq 0$，$\phi_{kk} = 0$ for $k > 2$
- ACF: $\rho_k = c_1 \phi_1^k + c_2 \phi_2^k$

---

## 14. 学习路径建议

### 入门阶段（1-2周）

1. 掌握时间序列基本概念
2. 学习平稳性定义
3. 理解 ACF/PACF
4. 实践 ADF 检验

### 进阶阶段（2-3周）

1. 掌握 Box-Jenkins 方法
2. 学习 ARIMA 建模全流程
3. 进行模型诊断
4. 实践自动阶数选择

### 高级阶段

1. SARIMA 季节性模型
2. ARIMAX 外生变量
3. 时间序列与深度学习结合
4. 多元时间序列

**推荐学习路线**：

```
描述统计 → 平稳性 → ACF/PACF → ADF检验 → 
AR模型 → MA模型 → ARIMA → SARIMA → Auto-ARIMA
```