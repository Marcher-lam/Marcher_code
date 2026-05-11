# 支持向量机(SVM) 学习文档

> 寻找最大间隔分类超平面，通过核技巧处理非线性问题。

> 来源线索：本节内容根据原书中关于"Support Vector Machines"的相关章节(Ch 3.10.5)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：SVM寻找使两类样本间隔最大的分类超平面，只依赖边界上的"支持向量"。

**直觉类比**：在操场上，穿红衣服和蓝衣服的学生自然分成两群。你需要在两群之间画一条线分开他们。SVM选择的线使得离最近的红蓝学生（支持向量）到线的距离最大——留最大的安全距离。

**历史背景**：SVM由Vapnik & Chervonenkis (1963)发展线性版本，Boser, Guyon & Vapnik (1992)引入核技巧。Cortes & Vapnik (1995)提出软间隔。原书Ch 3.10.5中作为参数化分类方法引入。

**算法定位**：监督学习/分类（也可回归）。在原书中用于策略分类和状态识别。

**前置知识**：线性分类、拉格朗日对偶、核函数。

## 2. 核心原理

**核心思想**：在特征空间中找最大间隔超平面$w^Tx + b = 0$。间隔定义为$\frac{2}{\|w\|}$。最大化间隔等价于最小化$\|w\|^2$。

**工作流程**：
1. 将数据映射到高维特征空间（可选，核技巧）
2. 求解凸二次规划找最优$w, b$
3. 用支持向量定义决策边界
4. 新样本通过$w^Tx + b$的符号分类

**核技巧**：不显式计算高维映射$\phi(x)$，而用核函数$K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$。常用核：RBF $K(x,y) = e^{-\gamma\|x-y\|^2}$。

## 3. 数学公式与推导

### 硬间隔SVM

$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^Tx_i + b) \geq 1, \forall i$$

### 软间隔SVM（允许误分类）

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(w^Tx_i+b) \geq 1-\xi_i, \xi_i \geq 0$$

### 对偶问题

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i\alpha_j y_iy_j K(x_i,x_j) \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C$$

决策函数：$f(x) = \text{sign}(\sum_i \alpha_i y_i K(x_i, x) + b)$

只有$\alpha_i > 0$的样本是支持向量。

## 4-8. 核心实现

```python
"""SVM：简化SMO实现"""
import numpy as np

class SimpleSVM:
    """简化SVM（线性核，梯度下降）"""
    def __init__(self, C=1.0, lr=0.01, n_iters=1000):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1).astype(float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.n_iters):
            for i in range(n):
                margin = y[i] * (X[i] @ self.w + self.b)
                if margin < 1:
                    self.w -= self.lr * (self.w - self.C * y[i] * X[i])
                    self.b += self.lr * self.C * y[i]
                else:
                    self.w -= self.lr * self.w
        return self

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    svm = SimpleSVM(C=1.0).fit(X, y)
    y_pred = svm.predict(X)
    acc = np.mean(y_pred == np.where(y == 0, -1, 1))
    print(f"SVM准确率: {acc:.2f}")
    print(f"权重: {svm.w.round(3)}, 偏置: {svm.b:.3f}")
```

## 9-14. 简要

### 12. 学习总结
SVM：$\min \frac{1}{2}\|w\|^2 + C\sum\xi_i$，最大化分类间隔。核技巧$K(x_i,x_j)$处理非线性。只依赖支持向量。

### 13. 练习题
**Q1**：为什么SVM只依赖支持向量？
**A1**：对偶问题中，只有$\alpha_i > 0$（违反间隔约束的样本）参与决策函数。删除非支持向量不影响模型。

### 14. 学习路径
**前置**：线性分类、拉格朗日对偶 | **进阶**：核方法、SMO算法
**资源**：原书Ch 3.10.5、Cortes & Vapnik (1995)
