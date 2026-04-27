# ACS 蚂蚁群系统 学习文档

> 蚂蚁群系统，ACO的高效变体。

## 1. 算法基础认知

ACS（Ant Colony System）是蚁群优化的重要变体，由Dorigo和Gambardella在1997年提出。ACS引入全局和局部信息素更新，以及伪随机比例规则来提高效率。

**前置知识**：蚂蚁系统

## 2. 核心原理

- 伪随机比例规则
- 局部信息素更新
- 全局信息素更新

## 3. 数学公式

**伪随机比例规则**：
$$S = \begin{cases} argmax [ \tau^\alpha \eta^\beta ] & q \le q_0 \ random & q > q_0 \end{cases}$$

## 4-14. 其他章节

参考原书中关于"Ant Colony System"的章节。

> 来源线索：本节内容根据原书中关于"Ant Colony System"的相关章节整理。