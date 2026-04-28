# CAVE算法 学习文档

> 用凸分段线性函数近似边际值函数，是资源分配问题的核心近似方法。

> 来源线索：本节内容根据原书中关于"CAVE Algorithm"的相关章节(Ch 18.3.2)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：CAVE(Concave Approximation of Value Estimation)用分段线性凹函数近似资源边际值，保证策略的单调性和凸结构。

**核心思想**：资源的边际价值$\partial V/\partial R$通常递减（凹性）。CAVE用线性段近似这个边际价值函数，通过蒙特卡洛采样更新断点。

**更新公式**：对资源水平$R$，采样得到新估计$\hat{v}$：

$$\bar{v}^{n+1}(R) = (1-\alpha_n)\bar{v}^n(R) + \alpha_n \hat{v}^{n+1}(R)$$

然后投影到凹分段线性函数空间，保证凹性约束。

**应用场景**：运输车队管理、能源存储、血液库存等资源分配问题。

## 4-8. 核心实现

```python
"""CAVE算法：凹分段线性值函数近似"""
import numpy as np

class CAVE:
    def __init__(self, breakpoints, gamma=0.95):
        self.bp = np.array(breakpoints)
        self.n_bp = len(breakpoints)
        self.marginal_v = np.zeros(self.n_bp - 1)  # 边际价值
        self.gamma = gamma

    def value(self, R):
        """评估资源水平R的总价值"""
        v = 0
        for i in range(self.n_bp - 1):
            if R > self.bp[i+1]:
                v += (self.bp[i+1] - self.bp[i]) * self.marginal_v[i]
            elif R > self.bp[i]:
                v += (R - self.bp[i]) * self.marginal_v[i]
            else:
                break
        return v

    def update(self, R, sample_value, alpha=0.1):
        """更新边际价值估计"""
        for i in range(self.n_bp - 1):
            if self.bp[i] < R <= self.bp[i+1]:
                self.marginal_v[i] = (1-alpha)*self.marginal_v[i] + alpha*sample_value
        self._project_concave()

    def _project_concave(self):
        """投影到凹函数（边际价值递减）"""
        for i in range(len(self.marginal_v)-1):
            if self.marginal_v[i] < self.marginal_v[i+1]:
                avg = (self.marginal_v[i] + self.marginal_v[i+1]) / 2
                self.marginal_v[i] = avg + 0.01
                self.marginal_v[i+1] = avg - 0.01

if __name__ == "__main__":
    np.random.seed(42)
    cave = CAVE(breakpoints=[0, 10, 20, 30, 40, 50])
    for n in range(500):
        R = np.random.randint(0, 50)
        true_v = max(0, 5 - 0.1*R)  # 递减边际价值
        cave.update(R, true_v + np.random.randn()*0.5, alpha=0.05)
    print("CAVE边际价值:", cave.marginal_v.round(3))
    print("资源价值 V(R):", {r: f"{cave.value(r):.2f}" for r in [0, 10, 20, 30, 40]})
```

## 9-14. 简要

### 12. 学习总结
CAVE：分段线性凹函数近似边际价值。保证凹性约束，适用于资源分配问题中值函数的结构化近似。

### 13. 练习题
**Q1**：为什么需要保证凹性？
**A1**：资源边际价值递减（经济学边际效用递减）。凹性保证优化问题是凸的，有唯一最优解，且贪心策略有效。

### 14. 学习路径
**前置**：VFA、资源分配 | **进阶**：平整算法、分段线性近似
**资源**：原书Ch 18.3.2
