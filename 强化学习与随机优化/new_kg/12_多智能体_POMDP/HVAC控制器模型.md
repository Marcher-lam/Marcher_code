# HVAC控制器模型(HVAC Controller Model) 学习文档

> 多独立智能体的典型应用——多个区域各自控制温度，共同优化能耗与舒适度。

> 来源线索：本节内容根据原书中关于"Multiple Independent Agents – An HVAC Controller Model"的相关章节(Ch 20.5)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：HVAC控制器模型将建筑暖通空调控制建模为多智能体序贯决策问题，每个区域(agent)独立控制温度以最小化总能耗。

## 2. 核心原理

**模型要素**：
- **智能体**：每个房间/区域一个agent
- **状态**：房间温度、室外温度、占用状态
- **决策**：制冷/制热功率
- **目标**：最小化总能耗 + 惩罚不舒适

**独立性**：
- 各区域温度独立演化（近似）
- 只通过总预算约束耦合
- 可以分解为独立子问题

**与多智能体系统的关系**：
- 独立智能体（无通信需求）
- 集中约束（总预算）
- 分布式执行（每个agent局部决策）

**策略**：
- 每个agent可以用独立的PFA/CFA/VFA
- 集中训练、分布式执行

## 3. Python 实现

```python
import numpy as np

class HVACAgent:
    """单个HVAC区域控制器"""
    def __init__(self, target_temp=22.0, area=50):
        self.temp = np.random.uniform(18, 28)
        self.target = target_temp
        self.area = area

    def step(self, power, outside_temp, dt=0.1):
        """更新温度"""
        # 热传导：与室外的热交换
        heat_transfer = 0.1 * (outside_temp - self.temp)
        # HVAC功率效果
        hvac_effect = power * 0.5 / self.area
        # 更新温度
        self.temp += dt * (heat_transfer + hvac_effect)

    def discomfort(self):
        return max(0, abs(self.temp - self.target) - 1.0) ** 2

class HVACSystem:
    """多区域HVAC系统"""
    def __init__(self, n_zones=5):
        self.agents = [HVACAgent(target_temp=22+np.random.randn()*2)
                       for _ in range(n_zones)]
        self.max_total_power = 10.0

    def step(self, powers, outside_temp):
        total_power = sum(powers)
        if total_power > self.max_total_power:
            scale = self.max_total_power / total_power
            powers = [p * scale for p in powers]

        for agent, power in zip(self.agents, powers):
            agent.step(power, outside_temp)

        energy = sum(powers)
        discomfort = sum(a.discomfort() for a in self.agents)
        return energy, discomfort

# 简单控制策略
system = HVACSystem(n_zones=5)
for t in range(100):
    outside = 30 + 5 * np.sin(t * 0.1)
    powers = [max(0, -(a.temp - a.target) * 0.5) for a in system.agents]
    energy, discomfort = system.step(powers, outside)

print(f"最终温度: {[round(a.temp, 1) for a in system.agents]}")
print(f"目标温度: {[a.target for a in system.agents]}")
```

## 4. 与其他方法的关系

- **多智能体RL**：每个agent独立学习
- **资源分配**：总功率约束下的分配
- **分布式控制**：去中心化决策

## 5. 参考文献

- Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization*, Ch 20.5
