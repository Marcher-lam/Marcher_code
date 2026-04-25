# IMPALA（Importance Weighted Actor-Learner Architecture）学习文档

> 大规模并行的Actor-Learner架构

---

## 1. 算法基础认知

**一句话定义**：IMPALA是一种大规模并行的强化学习架构，多个Actor并行收集经验，一个Learner集中训练，效率和扩展性都很高。

**历史背景**：由Espeholt等人在2018年提出，用于DeepMind的超大规模训练。

---

## 2. 核心原理

### 2.1 架构

```
Actor 1 ─┐
Actor 2 ─┼──→ 经验队列 ──→ Learner
Actor 3 ─┘
...
Actor N ─┘
```

### 2.2 V-trace

解决Actor和Learner之间的策略差异：

$$v_{target} = V(s_\tau) + \sum_{t=\tau}^{T-1} \gamma^{t-\tau} \cdot c_t \cdot \delta_t$$

其中 $c_t = \min(1, \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)})$ 是重要性权重。

---

## 3. 总结

✓ 大规模并行
✓ V-trace修正
✓ DeepMind常用架构