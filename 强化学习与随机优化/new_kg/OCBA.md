# OCBA 最优计算预算分配 学习文档

> 在有限仿真预算下，最优地分配模拟次数以找到最优方案。

> 来源线索：本节内容根据原书中关于"Optimal Computing Budget Allocation"的相关章节(Ch 7.10.2)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：OCBA在$K$个候选方案中，最优分配仿真预算$N$使得正确选择最优方案的概率（PCS）最大。

**核心思想**：不需要等精度估计每个方案，而是集中预算区分最优和次优。

**最优分配规则**（Chen et al. 2000）：

$$\frac{N_i}{N_j} = \left(\frac{\sigma_i/\delta_{i,b}}{\sigma_j/\delta_{j,b}}\right)^2, \quad \forall i,j \neq b$$

$$N_b = \sigma_b\sqrt{\sum_{i\neq b}\frac{N_i^2}{\sigma_i^2}}$$

其中$b$是当前估计最优，$\delta_{i,b} = \bar{\mu}_b - \bar{\mu}_i$，$\sigma_i$是方案$i$的标准差。

**关键洞察**：对远离最优的方案，少分配（不需要精确估计就知道它差）。对接近最优的方案，多分配（需要精确区分谁更好）。

## 4-8. 核心实现

```python
"""OCBA最优计算预算分配"""
import numpy as np

def ocba_allocate(means, stds, total_budget):
    """OCBA预算分配"""
    K = len(means)
    b = np.argmax(means)  # 当前最优
    # 计算比率
    ratios = np.zeros(K)
    delta = np.abs(means[b] - means)
    delta[b] = 1.0  # 避免除0

    for i in range(K):
        if i != b:
            ratios[i] = (stds[i] / delta[i])**2

    ratios[b] = stds[b] * np.sqrt(np.sum(ratios**2 / stds**2))
    # 归一化
    allocations = np.round(ratios / ratios.sum() * total_budget).astype(int)
    allocations = np.maximum(allocations, 2)
    return allocations

if __name__ == "__main__":
    np.random.seed(42)
    means = np.array([10, 9.5, 8, 6, 4])  # 方案真实均值
    stds = np.array([2, 3, 1, 2, 1.5])
    alloc = ocba_allocate(means, stds, total_budget=100)
    print(f"方案均值: {means}")
    print(f"方案标准差: {stds}")
    print(f"OCBA分配: {alloc}")
    print(f"等精度分配: {[20]*5}")
```

## 9-14. 简要

### 12. 学习总结
OCBA：$N_i/N_j \propto (\sigma_i/\delta_i)^2 / (\sigma_j/\delta_j)^2$。集中预算区分最优和次优，而非等精度估计所有方案。

### 13. 练习题
**Q1**：OCBA相比等精度分配的优势？
**A1**：等精度分配浪费预算在明显差的方案上。OCBA集中资源区分接近最优的方案，在相同预算下PCS更高。实验中OCBA可节省50-80%的预算。

### 14. 学习路径
**前置**：多臂赌博机、仿真优化 | **进阶**：知识梯度、最优实验设计
**资源**：原书Ch 7.10.2、Chen et al. (2000)
