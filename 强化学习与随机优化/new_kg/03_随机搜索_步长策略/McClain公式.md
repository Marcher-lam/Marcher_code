# McClain公式 学习文档

> 从固定步长平滑过渡到调和步长，兼顾跟踪能力和收敛精度。

> 来源线索：本节内容根据原书中关于"McClain公式"的相关章节(Ch 6.1.2.4)整理、扩展与教学化改写。

## 1-3. 核心认知与公式

**一句话定义**：McClain公式$\alpha_n = \alpha_{n-1}/(1+\alpha_{n-1}-\bar{\alpha})$从初始值$\alpha_0$平滑过渡到目标步长$\bar{\alpha}$。

**公式**：

$$\alpha_n = \frac{\alpha_{n-1}}{1 + \alpha_{n-1} - \bar{\alpha}}$$

- 当$\bar{\alpha} = 0$：退化为$1/n$（调和步长）
- 当$\bar{\alpha} > 0$：收敛到固定步长$\bar{\alpha}$（适合非平稳）
- 过渡速度由$\alpha_0$和$\bar{\alpha}$控制

**优势**：在训练初期用较大步长快速接近最优，后期收敛到固定步长保持跟踪能力。

## 4-8. 核心实现

```python
"""McClain公式步长策略"""
import numpy as np

def mcclain_sequence(alpha_0, alpha_bar, n_steps):
    """生成McClain步长序列"""
    alphas = [alpha_0]
    for _ in range(n_steps - 1):
        a_prev = alphas[-1]
        alphas.append(a_prev / (1 + a_prev - alpha_bar))
    return np.array(alphas)

if __name__ == "__main__":
    for alpha_bar in [0.0, 0.01, 0.05]:
        seq = mcclain_sequence(0.5, alpha_bar, 20)
        print(f"α_bar={alpha_bar}: {seq.round(4)}")
```

## 9-14. 简要

### 12. 学习总结
McClain：$\alpha_n = \alpha_{n-1}/(1+\alpha_{n-1}-\bar{\alpha})$。从$\alpha_0$平滑过渡到$\bar{\alpha}$，兼顾初期学习速度和后期稳定性。

### 13. 练习题
**Q1**：$\bar{\alpha}=0$时，$\alpha_n$退化为？
**A1**：$\alpha_n = \alpha_{n-1}/(1+\alpha_{n-1})$。若$\alpha_0=1$，则$\alpha_n=1/(n+1)$（调和步长）。

### 14. 学习路径
**前置**：调和步长 | **进阶**：Kesten规则、自适应步长
**资源**：原书Ch 6.1.2.4
