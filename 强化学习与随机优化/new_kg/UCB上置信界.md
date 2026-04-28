# UCB上置信界(Upper Confidence Bound) 学习文档

> 选择"均值+不确定性"最大的臂，是最经典的乐观探索策略。

> 来源线索：本节内容根据原书中关于"UCB"的相关章节(Ch 7)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：UCB为每条臂计算$\bar{\mu}_k + c\sqrt{\ln n / n_k}$，选择上界最大的臂——"乐观面对不确定性"。

**UCB1公式**：

$$k^{UCB} = \arg\max_k \left[\bar{\mu}_k + c\sqrt{\frac{\ln n}{n_k}}\right]$$

- $\bar{\mu}_k$：臂$k$的样本均值
- $n_k$：臂$k$的拉取次数
- $n$：总拉取次数
- $c$：探索参数（通常$c=\sqrt{2}$）

**直觉**：第二项是"不确定性奖金"——拉取少的臂不确定性大，获得额外加分鼓励探索。随着拉取增多，奖金减小，策略趋向利用。

**遗憾界**：UCB1的累积遗憾为$O(\sqrt{Kn\ln n})$，接近理论最优。

## 4-8. 核心实现

```python
"""UCB1算法"""
import numpy as np

class UCB1:
    def __init__(self, n_arms, c=2.0):
        self.K = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)

    def select(self):
        if np.any(self.counts == 0):
            return np.argmin(self.counts)
        n = self.counts.sum()
        ucb = self.rewards / self.counts + self.c * np.sqrt(np.log(n) / self.counts)
        return np.argmax(ucb)

    def update(self, k, reward):
        self.counts[k] += 1
        self.rewards[k] += reward

if __name__ == "__main__":
    np.random.seed(42)
    K = 5
    true_means = [1.0, 0.8, 0.6, 0.5, 0.3]
    ucb = UCB1(K)
    regrets = []
    for t in range(1000):
        k = ucb.select()
        r = np.random.random() < true_means[k]
        ucb.update(k, float(r))
        regrets.append(max(true_means) - true_means[k])
    print(f"UCB1 (1000步): 每臂拉取{ucb.counts.astype(int)}, 累积遗憾={sum(regrets):.1f}")
```

## 9-14. 简要

### 12. 学习总结
UCB1：选$\arg\max[\bar{\mu}_k + c\sqrt{\ln n/n_k}]$。乐观策略——不确定的臂获得探索奖金。遗憾$O(\sqrt{Kn\ln n})$。

### 13. 练习题
**Q1**：UCB中$c$参数增大/减小的效果？
**A1**：$c$大→更多探索（不确定性权重大），$c$小→更利用（均值主导）。$c=\sqrt{2}$是理论最优，实践中常调优。

### 14. 学习路径
**前置**：多臂赌博机 | **进阶**：Thompson采样、知识梯度
**资源**：原书Ch 7、Auer et al. (2002)
