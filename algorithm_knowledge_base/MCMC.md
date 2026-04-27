# 马尔可夫链蒙特卡洛 (MCMC) 学习文档

> 用随机采样近似复杂概率分布的"金标准"。

> 来源线索：本节内容根据原书中关于"MCMC"的相关章节（第13章13.4.5节）整理、扩展与教学化改写。

---

## 1. 算法基础认知

**一句话定义：** MCMC 通过构建马尔可夫链从目标分布中采样，用样本均值近似难以解析计算的期望。

**直觉类比：** 想计算一个不规则湖泊的平均深度。精确测量不可能，但可以随机在湖面上撒点测量。MCMC 就是一种"聪明地撒点"的方法——它会倾向在深水区多撒点、浅水区少撒点，使采样点分布反映真实深度分布。

**历史背景：** Metropolis 算法于 1953 年提出，Hastings 于 1970 年推广为 Metropolis-Hastings。Gibbs 采样由 Geman 等人于 1984 年提出。MCMC 是贝叶斯统计的核心计算工具。

**算法定位：** 采样方法、近似推断、贝叶斯计算。

**前置知识：** 马尔可夫链、概率分布、蒙特卡罗方法。

---

## 2-3. 核心原理与数学公式

### Metropolis-Hastings 算法

1. 从当前状态 $x_t$ 提出新状态 $x' \sim q(x'|x_t)$
2. 计算接受率 $\alpha = \min\left(1, \frac{p(x')q(x_t|x')}{p(x_t)q(x'|x_t)}\right)$
3. 以概率 $\alpha$ 接受 $x_{t+1} = x'$，否则 $x_{t+1} = x_t$

### Gibbs 采样

当条件分布 $p(x_i | x_{-i})$ 容易采样时，轮流从条件分布采样：

$$x_i^{(t+1)} \sim p(x_i | x_1^{(t+1)}, \ldots, x_{i-1}^{(t+1)}, x_{i+1}^{(t)}, \ldots)$$

### 用样本近似期望

$$\mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{T}\sum_{t=1}^{T} f(x^{(t)})$$

---

## 4-8. 代码实现

```python
import numpy as np

def metropolis_hastings(target_log_pdf, proposal_std=1.0, n_samples=10000, burn_in=1000):
    """通用 Metropolis-Hastings 采样器"""
    x = 0.0
    samples = []
    for _ in range(n_samples + burn_in):
        x_new = x + np.random.randn() * proposal_std
        log_alpha = target_log_pdf(x_new) - target_log_pdf(x)
        if np.log(np.random.rand()) < log_alpha:
            x = x_new
        samples.append(x)
    return np.array(samples[burn_in:])

# 从双峰分布采样
def bimodal_log_pdf(x):
    return np.log(0.5 * np.exp(-0.5*(x-3)**2) + 0.5 * np.exp(-0.5*(x+3)**2) + 1e-10)

samples = metropolis_hastings(bimodal_log_pdf, proposal_std=2.0, n_samples=10000)
print(f"采样均值: {samples.mean():.3f}")
print(f"采样标准差: {samples.std():.3f}")
print(f"采样范围: [{samples.min():.2f}, {samples.max():.2f}]")
```

---

## 9-14. 练习与路径

**题1：** MCMC 的 burn-in 阶段为什么必要？

**参考答案：** 马尔可夫链需要一定时间才能收敛到目标分布（平稳分布）。burn-in 前的样本来自非平稳分布，不能用于估计。丢弃 burn-in 样本确保后续样本近似来自目标分布。

### 学习路径
- 前置：马尔可夫链、蒙特卡罗方法
- 进阶：HMC（哈密顿蒙特卡洛）、NUTS、变分推断
- 推荐：Brooks et al., "Handbook of Markov Chain Monte Carlo"
