# VRPTW 带时间窗的车辆路径问题 学习文档

> VRP with Time Windows，每个客户有服务时间窗限制。

## 1. 算法基础认知

VRPTW（Vehicle Routing Problem with Time Windows）是VRP的扩展，每个客户只能在指定的时间窗内接受服务。

**直觉类比**：快递只能在客户在家的时间段送达（上午9:00-12:00，下午2:00-6:00）。

## 2. 核心原理

- 每个客户有时间窗[t_start, t_end]
- 车辆必须在时间窗内到达
- 早到需等待，晚到不可

## 3. 数学公式

$$t_i^{arrival} \in [t_i^{start}, t_i^{end}]$$

## 4-14. 其他章节

参考原书第5章关于VRPTW的内容。

> 来源线索：本节内容根据原书中关于"VRPTW"的相关章节整理。