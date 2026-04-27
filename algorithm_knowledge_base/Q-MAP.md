# Q-MAP 学习文档

> 用于移动自组织网络的多播路由算法。

## 1. 算法基础认知

Q-MAP是基于多智能体强化学习的移动自组织网络（MANET）多播路由算法。

**前置知识**：Q-Learning、网络路由

## 2. 核心原理

- 每个节点维护Q值表
- 多智能体协作学习路由
- 适应网络动态变化

## 3. 数学公式

**Q值更新**：
$$Q(n, d) \leftarrow (1-\alpha)Q(n,d) + \alpha[R + \gamma \min_{n'} Q(n', d)]$$

## 4-14. 其他章节

参考原书中关于"Q-MAP Multicast Routing"的章节。

> 来源线索：本节内容根据原书中关于"Q-MAP multicast routing"的相关章节整理。