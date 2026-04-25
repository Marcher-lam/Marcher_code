# Isotonic Regression 保序回归 学习文档

> 单调约束的回归方法，确保预测单调递增或递减

---

## 1. 算法基础认知

### 1.1 一句话定义

Isotonic Regression（保序回归）是一种约束回归方法，确保预测值随输入单调变化。当数据存在天然的单调关系时（如年龄与收入、剂量与疗效），它能提供更可靠的预测。

### 1.2 直觉类比

Isotonic Regression就像"整理书架"——如果你把书按从低到高排列，发现某本书比左边的书还矮，就把它和左边的书交换位置，反复直到所有书都按从小到大的顺序排列！这就是保序的核心思想：**让预测值保持单调性**！

想象你预测"年龄越大，收入越高"：
- 普通回归可能预测：20岁收入5万，30岁收入8万，35岁收入7万（违反单调性！）
- 保序回归确保：20岁<30岁<35岁的收入 monotonic（单调）

### 1.3 发展背景

- 1970年代，Barlow等人提出Isotonic Regression
- 1978年，Best和Chakravarti将PAVA算法应用于保序回归
- 广泛用于：剂量反应、排名学习、序数回归

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 回归 → 保序约束 |
| 输出 | 单调预测值 |
| 方法 | PAVA算法 |
| 特点 | 非参数、灵活 |

---

## 2. 核心原理

### 2.1 问题定义

给定训练数据 $(x_i, y_i)$，学习一个单调函数 $f$：
$$f(x_i) \leq f(x_j) \quad \text{when} \quad x_i \leq x_j$$

### 2.2 vs 普通回归

| 方面 | 普通回归 | 保序回归 |
|------|----------|----------|
| 单调约束 | 无 | 有 |
| 适用场景 | 通用 | 有单调关系 |
| 稳定性 | 可能违反约束 | 保证单调 |

### 2.3 核心思想

**PAVA算法**（Pool Adjacent Violators Algorithm）：
1. 按输入排序
2. 从左到右扫描
3. 遇到违反单调的点，与前一个合并
4. 重复直到满足单调

---

## 3. 数学公式与推导

### 3.1 优化目标

最小化均方误差，同时满足单调约束：
$$\min_f \sum_i (y_i - f(x_i))^2$$
$$s.t. \quad f(x_1) \leq f(x_2) \leq ... \leq f(x_n)$$

### 3.2 PAVA算法

```python
# PAVA伪代码
def pava(y):
    n = len(y)
    fits = [y[0]]
    weights = [1]
    
    for i in range(1, n):
        fits.append(y[i])
        weights.append(1)
        
        # 合并违反点
        while len(fits) >= 2:
            if fits[-2] <= fits[-1]:
                break
            # 合并最后两个
            w1, w2 = weights[-2], weights[-1]
            new_w = w1 + w2
            new_f = (fits[-2]*w1 + fits[-1]*w2) / new_w
            fits[-2] = new_f
            weights[-2] = new_w
            fits.pop()
            weights.pop()
    
    return fits
```

### 3.3 公式推导

设两块区域 $A$ 和 $B$ 需要合并：
$$f_A = \frac{\sum_{i \in A} y_i}{|A|}, \quad f_B = \frac{\sum_{i \in B} y_i}{|B|}$$

如果 $f_A > f_B$，合并后：
$$f_{AB} = \frac{|A|f_A + |B|f_B}{|A| + |B|}$$

---

## 4. 训练过程讲解

### 4.1 数据准备

```python
# 数据按x排序
X_sorted, indices = np.sort(X, return_inverse=True)
y_sorted = y[indices]
```

### 4.2 PAVA实现

```python
from sklearn.isotonic import IsotonicRegression

# 训练
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(X, y)

# 预测
y_pred = ir.predict(X_new)
```

### 4.3 参数

| 参数 | 说明 |
|------|------|
| out_of_bounds | 超出范围处理 |
| increasing | 递增/递减 |

---

## 5. 应用场景

### 5.1 剂量反应

```python
# 药物剂量 vs 疗效
doses = [0.1, 0.5, 1.0, 2.0, 5.0]
responses = [5, 15, 25, 40, 45]

# 保序回归确保疗效随剂量单调
ir = IsotonicRegression()
ir.fit(doses, responses)
```

### 5.2 排名学习

```python
# 学习评分函数
scores_train = model.predict_scores(features_train)
ir = IsotonicRegression()
scores_adjusted = ir.fit_transform(scores_train, labels_train)
```

### 5.3 序数回归

```python
# 预测有序类别
ir = IsotonicRegression()
thresholds = ir.fit_transform(continuous_scores, ordinal_labels)
```

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 单调保证 | 满足约束 |
| 非参数 | 无分布假设 |
| 计算快 | PAVA O(n) |
| 稳定 | 鲁棒 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 只能单调 | 不适合一般数据 |
| 维度灾难 | 高维需降维 |
| 不连续 | 预测可能跳跃 |

---

## 7. 调库实现（Python）

### 7.1 sklearn

```python
import numpy as np
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

# 数据
np.random.seed(42)
X = np.sort(np.random.rand(100) * 10)
y = X**2 + np.random.randn(100) * 2

# 保序回归
ir = IsotonicRegression(out_of_bounds='clip')
y_pred = ir.fit_transform(X, y)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Isotonic')
plt.legend()
plt.title('Isotonic Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.savefig('isotonic_demo.png', dpi=100)
plt.show()
```

### 7.2 多维扩展

```python
# 使用RBF核+保序
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler

# 保序在最后
pipeline = Pipeline([
    ('rbf', RBFSampler(gamma=0.1)),
    ('iso', IsotonicRegression())
])
pipeline.fit(X_multi, y)
```

---

## 8. 手工代码实现（理解原理）

```python
import numpy as np

def pava_isotonic(y, weights=None):
    """
    Pool Adjacent Violators Algorithm (PAVA)
    保序回归核心算法
    """
    if weights is None:
        weights = np.ones(len(y))
    
    n = len(y)
    if n == 0:
        return []
    
    # 初始化
    fitted = [y[0]]
    weights_fitted = [weights[0]]
    
    for i in range(1, n):
        fitted.append(y[i])
        weights_fitted.append(weights[i])
        
        # 合并违反点
        while len(fitted) >= 2:
            # 检查最后两个是否违反单调
            if fitted[-2] <= fitted[-1]:
                break
            
            # 违反，合并
            w1, w2 = weights_fitted[-2], weights_fitted[-1]
            new_w = w1 + w2
            new_f = (fitted[-2]*w1 + fitted[-1]*w2) / new_w
            
            fitted[-2] = new_f
            weights_fitted[-2] = new_w
            fitted.pop()
            weights_fitted.pop()
    
    return fitted, weights_fitted


def isotonic_transform(x_train, y_train, x_test):
    """保序回归变换"""
    # 按x排序
    order = np.argsort(x_train)
    x_sorted = x_train[order]
    y_sorted = y_train[order]
    
    # PAVA
    fitted, weights_fitted = pava_isotonic(y_sorted)
    
    # 对测试集预测
    predictions = np.zeros_like(x_test)
    x_test_order = np.argsort(x_test)
    
    # 简化：找到最近训练点
    for i, xi in enumerate(x_test):
        idx = np.searchsorted(x_sorted, xi)
        idx = min(max(idx, 0), len(fitted)-1)
        predictions[x_test_order[i]] = fitted[idx]
    
    return predictions


class IsotonicRegressionManual:
    """手工实现保序回归"""
    def __init__(self):
        self.x_train_ = None
        self.y_fitted_ = None
        
    def fit(self, X, y):
        self.x_train_ = np.sort(X)
        y_sorted = y[np.argsort(X)]
        self.y_fitted_, _ = pava_isotonic(y_sorted)
        return self
    
    def predict(self, X):
        predictions = np.zeros_like(X)
        for i, xi in enumerate(X):
            idx = np.searchsorted(self.x_train_, xi)
            idx = min(max(idx, 0), len(self.y_fitted_)-1)
            predictions[i] = self.y_fitted_[idx]
        return predictions
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.predict(X)


if __name__ == "__main__":
    np.random.seed(42)
    
    # 测试数据
    X = np.sort(np.random.rand(50) * 10)
    y = X**2 + np.random.randn(50) * 5
    
    # 手工实现
    ir_manual = IsotonicRegressionManual()
    y_pred_manual = ir_manual.fit_transform(X, y)
    
    # sklearn
    ir_sklearn = IsotonicRegression()
    y_pred_sklearn = ir_sklearn.fit_transform(X, y)
    
    print("手工实现前5个:", y_pred_manual[:5])
    print("sklearn前5个:", y_pred_sklearn[:5])
    print("差异:", np.abs(y_pred_manual - y_pred_sklearn).max())
```

---

## 9. 可视化与结果理解

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_isotonic():
    """可视化保序回归"""
    np.random.seed(42)
    
    # 数据：单调递增+噪声
    X = np.sort(np.random.rand(100) * 10)
    y = X + np.random.randn(100) * 3
    
    # 添加违反点
    y[10] = -5
    y[80] = 50
    
    # sklearn
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression()
    y_pred = ir.fit_transform(X, y)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 原始数据
    axes[0].scatter(X, y, alpha=0.6)
    axes[0].plot(X, y, 'b--', alpha=0.5, label='True')
    axes[0].set_title('原始数据（含违反点）')
    axes[0].legend()
    
    # 保序回归
    axes[1].scatter(X, y, alpha=0.6, label='Data')
    axes[1].plot(X, y_pred, 'r-', linewidth=2, label='Isotonic')
    axes[1].set_title('保序回归后')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('isotonic_effect.png', dpi=100)
    plt.show()


visualize_isotonic()
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| MSE | 均方误差 |
| MAE | 平均绝对误差 |
| 单调性检验 | 是否满足约束 |

### 10.2 评估代码

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_isotonic(y_true, y_pred):
    """评估保序回归"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 检查单调性
    monotonic_violations = np.sum(np.diff(y_pred) < 0)
    
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"单调违反次数: {monotonic_violations}")
    
    return {'MSE': mse, 'MAE': mae, 'Violations': monotonic_violations}
```

---

## 11. 常见问题与易错点

### Q1: 何时使用保序回归？

**答案**：当特征与目标存在单调关系时（如年龄-收入、剂量-疗效）。

### Q2: 如何处理递减约束？

**答案**：对y取负，或设置increasing=False。

### Q3: 多维数据怎么办？

**答案**：先降维，或用核方法。

### Q4: 预测不连续？

**答案**：这是PAVA特性，可用样条插值平滑。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 核心 | 单调约束回归 |
| 算法 | PAVA |
| 优点 | 保证单调性 |
| 应用 | 剂量反应、排名 |

### 12.2 公式汇总

优化目标：
$$\min \sum (y_i - f(x_i))^2 \quad s.t. f(x_1) \leq ... \leq f(x_n)$$

PAVA合并：
$$f_{AB} = \frac{|A|f_A + |B|f_B}{|A| + |B|}$$

---

## 13. 练习题与思考题

### 13.1 选择题

1. PAVA算法的时间复杂度是：
   - A) O(n²)
   - B) O(n)
   - C) O(n log n)

2. 保序回归适用于：
   - A) 任意数据
   - B) 单调关系数据
   - C) 线性数据

### 13.2 简答题

1. 解释PAVA算法如何确保单调性？
2. 比较保序回归和普通线性回归。

### 13.3 编程题

1. 实现带权重的PAVA算法。
2. 用保序回归预测剂量-反应曲线。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
回归基础
    ↓
约束回归
    ↓
PAVA算法
    ↓
保序回归
    ↓
剂量反应建模
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| 线性回归 | 基础 |
| 样条回归 | 连续版 |
| 排名学习 | 应用 |

### 14.3 扩展阅读

- Barlow et al. (1972). Isotonic Regression. Statistical Decision Rules.

---

## 附录

### 参考

1. Best & Chakravarti (1978). Active Set Algorithms for Isotonic Regression.
2. sklearn.isotonic documentation

---

**文档结束**