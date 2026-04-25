# Distributional DQN 学习文档

> 学习Q值分布而非单一期望值

---

## 1. 算法基础认知

**一句话定义**：Distributional DQN不学习Q的期望值，而是学习Q值在整个分布上的分布，能够更精确地理解风险。

**直觉类比**：普通DQN说"这道菜值9分"，Distributional DQN说"这道菜：30%概率8分，50%概率9分，20%概率10分"——后者信息更丰富。

**历史背景**：由Bellemare等人在2017年提出。

---

## 2. 核心原理

### 2.1 分布Q学习

不是学习 $Q(s,a)$，而是学习 $Z(s,a)$ 的分布：

$$Z(s,a) = \text{Categorical}(\text{support}, \text{probs})$$

支持集：$\{V_{min}, V_{min}+\delta, ..., V_{max}\}$

### 2.2 投影

```python
# 将TD目标投影到支持集
def project(dist, support):
    projection =.zeros_like(dist)
    for i, v in enumerate(support):
        idx = min(i, len(support)-1)
        projection[idx] = dist[i]
    
    return projection
```

---

## 3. 总结

✓ 捕捉风险
✓ 更稳定的训练
✓ Atari显著提升