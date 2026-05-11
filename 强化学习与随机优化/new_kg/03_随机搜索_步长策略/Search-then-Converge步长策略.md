# Search-then-Converge步长策略 学习文档

> 前期固定大步长学习，后期切换到衰减步长收敛。

> 来源线索：本节内容根据原书中关于"Search-then-Converge"的相关章节(Ch 6.1.2.5)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：STC策略在前期用固定步长$\alpha_0$广泛搜索，经过$\tau$步后切换到$1/n$衰减，实现搜索与收敛的平滑过渡。

**公式**：

$$\alpha_n = \alpha_0 \frac{1 + n/\tau}{(1 + n/\tau)^2 + n/\tau}$$

- $n \ll \tau$时：$\alpha_n \approx \alpha_0$（固定步长，搜索阶段）
- $n \gg \tau$时：$\alpha_n \approx \alpha_0 \tau / n$（调和衰减，收敛阶段）
- $\tau$控制搜索持续多久

**直觉**：像找水源——先在大范围快跑（固定步长），感觉接近目标后慢慢缩小搜索圈（递减步长）。

## 4-8. 核心实现

```python
"""Search-then-Converge步长策略"""
import numpy as np

class STCStepsize:
    def __init__(self, alpha0=0.1, tau=100):
        self.alpha0 = alpha0
        self.tau = tau

    def __call__(self, n):
        ratio = n / self.tau
        return self.alpha0 * (1 + ratio) / ((1 + ratio)**2 + ratio)

if __name__ == "__main__":
    np.random.seed(42)
    stc = STCStepsize(alpha0=0.1, tau=100)
    for n in [0, 10, 50, 100, 200, 500, 1000]:
        print(f"n={n:4d}: α={stc(n):.5f}")
```

## 9-14. 简要

### 12. 学习总结
STC：$\alpha_n = \alpha_0(1+n/\tau)/((1+n/\tau)^2+n/\tau)$。前期搜索（固定步长），后期收敛（$1/n$衰减）。$\tau$控制切换时机。

### 13. 练习题
**Q1**：$\tau$太大和太小分别有什么问题？
**A1**：$\tau$太大→搜索太久，浪费时间在不精确估计上；$\tau$太小→过早切换到收敛，可能陷入局部最优。

### 14. 学习路径
**前置**：调和步长、固定步长 | **进阶**：McClain公式、自适应步长
**资源**：原书Ch 6.1.2.5
