# 序贯Kriging(Sequential Kriging) 学习文档

> 来自地质统计学的连续空间优化方法，用高斯过程建模+期望改进准则指导黑箱函数优化。

> 来源线索：本节内容根据原书中关于"Sequential Kriging"的相关章节(Ch 7.4)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：序贯Kriging使用高斯过程(GP)作为元模型，通过期望改进(EI)准则自适应地选择下一个评估点，适用于昂贵的连续黑箱函数优化。

**直觉类比**：想象你在一片未知地形中找最高峰，但每次测量海拔要花1小时。Kriging让你建立地形的大致模型，然后告诉你"哪里最可能有更高的峰"——这就是EI准则。

**历史背景**：Kriging由南非矿业工程师Krige (1951)提出用于金矿储量估计。Matheron (1963)发展了理论基础。Jones et al. (1998)将EI准则与GP结合提出高效全局优化(EGO)。

**算法定位**：连续空间中的贝叶斯优化方法。适用于评估成本高的黑箱函数（如工程仿真、药物试验）。

**前置知识**：
- 高斯过程回归
- 贝叶斯更新
- 优化基础

## 2. 核心原理

**核心思想**：
1. 用GP拟合已有的观测数据 $(x^n, y^n)_{n=1}^{N}$
2. GP给出每个点的预测均值 $\mu(x)$ 和不确定性 $\sigma(x)$
3. 计算期望改进 $EI(x) = E[\max(f^* - f(x), 0)]$
4. 选择EI最大的点进行下一次评估

**GP元模型**：
$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

预测分布：
$$f(x) | D \sim \mathcal{N}(\mu(x), \sigma^2(x))$$

**期望改进(EI)准则**：
$$EI(x) = (f^* - \mu(x)) \Phi\left(\frac{f^* - \mu(x)}{\sigma(x)}\right) + \sigma(x) \phi\left(\frac{f^* - \mu(x)}{\sigma(x)}\right)$$

其中 $f^*$ 是当前观测到的最优值，$\Phi$ 和 $\phi$ 分别是标准正态的CDF和PDF。

**EI的直觉**：
- **第一项**：已知的改进潜力（均值超过当前最优的部分）
- **第二项**：探索的潜力（不确定性大的区域）
- EI自动平衡探索(exploration)和利用(exploitation)

## 3. Python 实现

```python
import numpy as np
from scipy.stats import norm

def expected_improvement(x, mu_func, sigma_func, f_best):
    """计算期望改进"""
    mu = mu_func(x)
    sigma = sigma_func(x)
    if sigma < 1e-10:
        return 0
    z = (f_best - mu) / sigma  # 注意：假设最小化
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return max(ei, 0)

def simple_kriging_optimization(f, bounds, n_init=5, n_iter=20):
    """简化的Kriging优化"""
    dim = len(bounds)
    # 初始随机采样
    X = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_init, dim))
    y = np.array([f(x) for x in X])

    for i in range(n_iter):
        f_best = y.min()
        # 简单GP预测（实际应用中用sklearn GP）
        # 这里用简化的近邻方法示意
        candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], (100, dim))
        eis = []
        for xc in candidates:
            dists = np.linalg.norm(X - xc, axis=1)
            w = 1.0 / (dists + 0.01)
            mu = w @ y / w.sum()
            sigma = np.sqrt(np.sum(w * (y - mu)**2) / w.sum() + 0.1)
            eis.append(expected_improvement(xc, lambda x: mu, lambda x: sigma, f_best))

        x_next = candidates[np.argmax(eis)]
        y_next = f(x_next)
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

    best_idx = np.argmin(y)
    return X[best_idx], y[best_idx]

# 示例
f = lambda x: (x[0] - 2)**2 + np.sin(4 * x[0])
bounds = np.array([[-5, 5]])
x_opt, y_opt = simple_kriging_optimization(f, bounds)
print(f"最优解: x={x_opt:.3f}, f(x)={y_opt:.3f}")
```

## 4. 与其他方法的关系

- **高斯过程回归**：Kriging的底层模型
- **知识梯度**：离散空间的类似方法
- **贝叶斯优化**：Kriging + EI的统称
- **响应曲面法**：更简单但更局限的连续优化方法

## 5. 参考文献

- Krige, D.G. (1951). A statistical approach to some mine valuation problems. *J. South African Inst. Mining Metallurgy*
- Jones, D.R. et al. (1998). Efficient global optimization of expensive black-box functions. *J. Global Optimization*
- Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization*, Ch 7.4
