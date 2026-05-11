# 逆CDF采样(Inverse CDF Sampling) 学习文档

> 用均匀随机数通过逆累积分布函数生成任意分布的样本。

> 来源线索：本节内容根据原书中关于"Inverse Cumulative Distribution Sampling"的相关章节(Ch 10.4.3)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：逆CDF采样通过$X = F^{-1}(U)$将均匀随机数$U \sim \text{Uniform}(0,1)$转换为任意分布$F$的样本。

**核心定理（逆变换定理）**：若$U \sim \text{Uniform}(0,1)$，$F$是连续CDF，则$X = F^{-1}(U)$服从分布$F$。

**证明**：$\mathbb{P}(X \leq x) = \mathbb{P}(F^{-1}(U) \leq x) = \mathbb{P}(U \leq F(x)) = F(x)$

**常用逆CDF**：

| 分布 | CDF $F(x)$ | 逆CDF $F^{-1}(u)$ |
|------|-----------|-------------------|
| 指数 | $1-e^{-\lambda x}$ | $-\ln(1-U)/\lambda$ |
| 均匀$(a,b)$ | $(x-a)/(b-a)$ | $a + (b-a)U$ |
| 正态 | $\Phi(x)$ | 需数值方法 |
| 几何 | $1-(1-p)^{k+1}$ | $\lceil\ln(1-U)/\ln(1-p)\rceil - 1$ |

**从数据采样**：排序观测$X_{(1)} \leq \cdots \leq X_{(N)}$，$F^{-1}(u) = X_{(\lceil uN \rceil)}$。

## 4-8. 核心实现

```python
"""逆CDF采样"""
import numpy as np

class InverseCDFSampler:
    @staticmethod
    def sample_exponential(lam, size=1):
        U = np.random.uniform(0, 1, size)
        return -np.log(U) / lam

    @staticmethod
    def sample_geometric(p, size=1):
        U = np.random.uniform(0, 1, size)
        return np.ceil(np.log(1 - U) / np.log(1 - p)).astype(int)

    @staticmethod
    def sample_from_data(data, size=1):
        """从经验分布采样"""
        sorted_data = np.sort(data)
        U = np.random.uniform(0, 1, size)
        indices = np.clip((U * len(sorted_data)).astype(int), 0, len(sorted_data)-1)
        return sorted_data[indices]

if __name__ == "__main__":
    np.random.seed(42)
    sampler = InverseCDFSampler()
    # 指数分布
    exp_samples = sampler.sample_exponential(2.0, 10000)
    print(f"Exp(2)采样: 均值={exp_samples.mean():.3f} (理论={1/2:.3f})")
    # 几何分布
    geo_samples = sampler.sample_geometric(0.3, 10000)
    print(f"Geo(0.3)采样: 均值={geo_samples.mean():.2f} (理论={1/0.3-1:.2f})")
    # 经验分布
    data = np.random.normal(5, 2, 100)
    emp_samples = sampler.sample_from_data(data, 5000)
    print(f"经验分布采样: 均值={emp_samples.mean():.2f} (数据均值={data.mean():.2f})")
```

## 9-14. 简要

### 12. 学习总结
逆CDF：$X = F^{-1}(U)$，$U\sim\text{Uniform}(0,1)$。将均匀随机数转换为任意分布样本。蒙特卡洛仿真的基础工具。

### 13. 练习题
**Q1**：为什么正态分布不能直接用逆CDF？
**A1**：正态CDF $\Phi(x)$没有解析逆函数。需要数值方法（如Box-Muller变换或近似公式）生成正态样本。

### 14. 学习路径
**前置**：概率论、CDF | **进阶**：拒绝采样、MCMC
**资源**：原书Ch 10.4.3
