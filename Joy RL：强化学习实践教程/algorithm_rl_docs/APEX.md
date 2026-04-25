# Ape-X（Prioritized Experience Replay with Distributed Workers）

> 大规模分布式 prioritized DQN

---

## 1. 算法基础认知

**一句话定义**：Ape-X将优先级回放与分布式架构结合，使用多个分布式Worker收集经验，实现大规模高效训练。

---

## 2. 核心组件

```
Worker 1 ──┐
Worker 2 ──┼──→ Replay Buffer ←→ Learner
...   ────┤
Worker N ──┘
```

### 核心改进

1. **分布式收集**：多个Actor并行动行
2. **优先级回放**：优先学习高TD误差
3. **GPU加速**：Learner使用GPU训练

---

## 3. 总结

✓ 超大规模训练
✓ 高样本效率
✓ 比A3C更快