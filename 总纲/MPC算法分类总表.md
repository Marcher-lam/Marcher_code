# MPC（Model Predictive Control，模型预测控制）算法分类总表

> 说明：MPC 是一种“滚动时域优化 / 预测控制”框架，不是单一算法。它的核心思想是在每个控制时刻，根据系统模型预测未来一段时间的状态，求解一个带约束的优化问题，只执行第一步控制动作，然后进入下一时刻重新预测、重新优化。  
> 本文按 **建模对象、优化形式、不确定性处理、实时求解方式、学习方式、系统结构、应用场景** 对目前常见 MPC 算法体系进行系统分类。

---

## 0. MPC 总体分类框架

MPC 可以从以下几个维度分类：

| 分类维度 | 主要类别 |
|---|---|
| 系统模型 | 线性 MPC、非线性 MPC、混合系统 MPC、混合整数 MPC、数据驱动 MPC、学习型 MPC |
| 不确定性处理 | 名义 MPC、鲁棒 MPC、随机 MPC、分布鲁棒 MPC、机会约束 MPC |
| 目标函数 | 跟踪 MPC、经济 MPC、多目标 MPC、风险敏感 MPC、安全约束 MPC |
| 计算方式 | 在线 MPC、显式 MPC、快速 MPC、嵌入式 MPC、近似 MPC、神经网络 MPC |
| 控制结构 | 集中式 MPC、分布式 MPC、去中心化 MPC、层级 MPC、协同 MPC |
| 时间尺度 | 离散时间 MPC、连续时间 MPC、多速率 MPC、事件触发 MPC |
| 应用场景 | 过程控制、机器人、自动驾驶、无人机、能源、电力、金融、医疗、交通、强化学习等 |

---

# 1. 按系统模型分类

## 1.1 线性 MPC（Linear MPC, LMPC）

线性 MPC 假设系统动力学为线性形式：

```text
x_{t+1} = A x_t + B u_t
```

常见形式：

- 线性二次 MPC（Linear Quadratic MPC）
- 线性约束 MPC
- 状态空间线性 MPC
- 输出反馈线性 MPC
- 线性时不变 MPC（LTI-MPC）
- 线性时变 MPC（LTV-MPC）
- 线性参数变化 MPC（LPV-MPC）
- 增量式线性 MPC
- 线性跟踪 MPC
- 线性调节 MPC

适用场景：

- 工业过程控制
- 化工过程
- 电机控制
- 温控系统
- 能源调度
- 简化车辆控制

优点：

- 优化问题通常是二次规划（QP）
- 求解速度快
- 稳定性理论成熟
- 适合嵌入式部署

---

## 1.2 非线性 MPC（Nonlinear MPC, NMPC）

NMPC 处理非线性系统：

```text
x_{t+1} = f(x_t, u_t)
```

主要类型：

- 通用非线性 MPC
- 非线性跟踪 MPC
- 非线性经济 MPC
- 非线性鲁棒 MPC
- 非线性随机 MPC
- 非线性输出反馈 MPC
- 非线性约束 MPC
- 连续时间 NMPC
- 离散时间 NMPC
- 实时迭代 NMPC（Real-Time Iteration NMPC, RTI-NMPC）
- 多重射击 NMPC（Multiple Shooting NMPC）
- 单射击 NMPC（Single Shooting NMPC）
- 直接配点 NMPC（Direct Collocation NMPC）
- 伪谱法 NMPC（Pseudospectral NMPC）

适用场景：

- 机器人控制
- 自动驾驶
- 无人机控制
- 航天器控制
- 化工反应器
- 生物系统
- 机械臂控制

优点：

- 可处理真实非线性动力学
- 控制性能高
- 适合高动态系统

缺点：

- 计算量大
- 对模型精度敏感
- 稳定性证明更复杂

---

## 1.3 混合系统 MPC（Hybrid MPC）

混合系统同时包含连续状态和离散状态。

典型系统：

```text
连续动力学 + 离散模式切换
```

主要算法：

- 混合 MPC（Hybrid MPC）
- 混合整数 MPC（Mixed-Integer MPC, MI-MPC）
- 混合整数线性 MPC（MILP-MPC）
- 混合整数二次 MPC（MIQP-MPC）
- 分段仿射系统 MPC（Piecewise Affine MPC, PWA-MPC）
- 逻辑动态系统 MPC（Mixed Logical Dynamical MPC, MLD-MPC）
- 自动机约束 MPC
- 开关系统 MPC
- 模式调度 MPC
- 逻辑约束 MPC

适用场景：

- 车辆换挡控制
- 电力系统开关控制
- 楼宇 HVAC 控制
- 生产调度
- 交通信号控制
- 机器人接触切换
- 电池管理系统

---

## 1.4 离散事件系统 MPC

适用于以事件驱动为主的系统。

代表方法：

- 离散事件 MPC
- Petri 网 MPC
- 自动机 MPC
- 排队网络 MPC
- 生产系统 MPC
- 调度 MPC
- 事件触发 MPC

应用：

- 制造系统
- 仓储物流
- 交通系统
- 网络调度
- 云计算资源调度

---

## 1.5 数据驱动 MPC（Data-Driven MPC）

不完全依赖显式物理模型，而是从数据中学习系统行为。

主要方法：

- 数据驱动 MPC
- 无模型 MPC（Model-Free MPC）
- 基于 Hankel 矩阵的 DeePC（Data-enabled Predictive Control）
- 子空间辨识 MPC
- Koopman MPC
- 高斯过程 MPC
- 神经网络 MPC
- 稀疏辨识 MPC
- SINDy-MPC
- 经验模型 MPC
- 在线辨识 MPC
- 行为系统理论 MPC

适用场景：

- 难以建立精确模型的工业系统
- 机器人
- 智能建筑
- 能源系统
- 生物医学控制
- 复杂动力系统

---

# 2. 按不确定性处理分类

## 2.1 名义 MPC（Nominal MPC）

假设模型完全准确，扰动可以忽略。

特点：

- 只优化一个确定性预测模型
- 不显式考虑模型误差
- 实现简单
- 常作为其他 MPC 的基础

常见算法：

- 名义线性 MPC
- 名义非线性 MPC
- 名义跟踪 MPC
- 名义经济 MPC

缺点：

- 对模型误差和扰动敏感
- 在安全关键系统中风险较高

---

## 2.2 鲁棒 MPC（Robust MPC, RMPC）

鲁棒 MPC 显式考虑扰动和模型不确定性，目标是在最坏情况下仍满足约束。

主要类型：

### 2.2.1 Min-Max MPC

- 极小极大 MPC
- 最坏情况 MPC
- 对抗扰动 MPC
- 鲁棒优化 MPC

形式：

```text
min_u max_w cost(x,u,w)
```

适用：

- 安全关键控制
- 模型误差较大系统
- 风险保守场景

---

### 2.2.2 Tube MPC

Tube MPC 让真实状态围绕名义轨迹形成一个“管状集合”。

主要算法：

- Tube MPC
- 约束收缩 Tube MPC
- 刚性 Tube MPC
- 同伦 Tube MPC
- 弹性 Tube MPC
- 自适应 Tube MPC
- 非线性 Tube MPC
- 随机 Tube MPC
- 鲁棒 Tube MPC
- LPV Tube MPC

核心思想：

- 优化名义轨迹
- 用反馈控制抵消扰动
- 对约束进行收紧
- 保证真实轨迹始终在安全管内

---

### 2.2.3 Set-Membership MPC

基于集合描述不确定性。

主要类型：

- 多面体不确定集 MPC
- 椭球不确定集 MPC
- 区间不确定集 MPC
- 可达集 MPC
- 不变集 MPC
- 终端不变集 MPC
- 鲁棒正不变集 MPC

---

### 2.2.4 Constraint Tightening MPC

通过收缩状态或输入约束来保证鲁棒性。

常见方法：

- 固定约束收缩 MPC
- 自适应约束收缩 MPC
- 场景约束收缩 MPC
- 概率约束收缩 MPC
- 学习型约束收缩 MPC

---

## 2.3 随机 MPC（Stochastic MPC, SMPC）

随机 MPC 将扰动建模为概率分布，而不是最坏情况。

主要类型：

- 随机 MPC
- 机会约束 MPC（Chance-Constrained MPC）
- 场景 MPC（Scenario MPC）
- 采样 MPC
- 概率 Tube MPC
- 随机 NMPC
- 马尔可夫跳变系统 MPC
- 随机混合系统 MPC
- 蒙特卡洛 MPC
- 贝叶斯 MPC
- 风险敏感随机 MPC

适用场景：

- 可以获得扰动分布的系统
- 交通预测
- 能源调度
- 金融控制
- 自动驾驶
- 供需不确定系统

---

## 2.4 分布鲁棒 MPC（Distributionally Robust MPC, DRMPC）

分布鲁棒 MPC 不假设精确概率分布，而是假设真实分布属于某个分布集合。

主要方法：

- Wasserstein 分布鲁棒 MPC
- 矩约束分布鲁棒 MPC
- φ-divergence 分布鲁棒 MPC
- KL 散度分布鲁棒 MPC
- CVaR 分布鲁棒 MPC
- 数据驱动分布鲁棒 MPC
- 样本外鲁棒 MPC

适用：

- 数据有限
- 分布漂移明显
- 安全与经济性都重要的系统

---

## 2.5 风险敏感 MPC（Risk-Sensitive MPC）

显式考虑风险指标。

常见算法：

- CVaR-MPC
- VaR-MPC
- Entropic Risk MPC
- Mean-Variance MPC
- Worst-Case Risk MPC
- Coherent Risk MPC
- 多目标风险 MPC
- 安全概率约束 MPC

应用：

- 自动驾驶
- 金融投资
- 能源交易
- 医疗决策
- 机器人避障

---

# 3. 按目标函数分类

## 3.1 跟踪 MPC（Tracking MPC）

目标是让系统状态或输出跟踪参考轨迹。

主要类型：

- 轨迹跟踪 MPC
- 设定点跟踪 MPC
- 输出跟踪 MPC
- 路径跟踪 MPC
- 速度跟踪 MPC
- 姿态跟踪 MPC
- 参考治理 MPC
- 伺服 MPC

应用：

- 车辆路径跟踪
- 无人机轨迹控制
- 机械臂控制
- 电机控制
- 工业过程跟踪

---

## 3.2 调节 MPC（Regulation MPC）

目标是让系统收敛到平衡点。

主要类型：

- 稳态调节 MPC
- 原点稳定 MPC
- 终端约束 MPC
- 终端成本 MPC
- Lyapunov MPC
- 稳定化 MPC

---

## 3.3 经济 MPC（Economic MPC, EMPC）

经济 MPC 不再简单跟踪参考轨迹，而是直接优化经济目标。

目标函数可能包括：

- 能耗最小
- 产量最大
- 利润最大
- 排放最小
- 运行成本最小
- 资源利用率最大

主要算法：

- 经济 MPC
- 非线性经济 MPC
- 鲁棒经济 MPC
- 随机经济 MPC
- 分布式经济 MPC
- 多目标经济 MPC
- 带耗散性约束的经济 MPC
- 带终端约束的经济 MPC
- 无终端约束经济 MPC

应用：

- 化工过程优化
- 电力系统
- 智能电网
- 供应链
- 数据中心能耗优化
- 工业节能

---

## 3.4 多目标 MPC（Multi-Objective MPC）

同时优化多个目标。

常见形式：

- 加权和多目标 MPC
- Pareto MPC
- 层级多目标 MPC
- 词典序 MPC
- 约束优先级 MPC
- 安全-性能折中 MPC
- 舒适性-能耗折中 MPC

应用：

- 自动驾驶：安全、舒适、效率
- HVAC：舒适、节能、空气质量
- 机器人：精度、能耗、避障
- 电网：成本、稳定性、排放

---

## 3.5 安全 MPC（Safety-Critical MPC）

强调约束永不违反。

主要方法：

- 安全约束 MPC
- 控制屏障函数 MPC（CBF-MPC）
- Lyapunov-MPC
- 可达性 MPC
- 安全集 MPC
- 递归可行 MPC
- 故障安全 MPC
- Shielded MPC
- Safe Learning MPC

应用：

- 自动驾驶
- 医疗设备
- 航空航天
- 机器人
- 人机协作

---

# 4. 按优化问题形式分类

## 4.1 QP-MPC（二次规划 MPC）

最常见的线性 MPC 形式。

特点：

- 二次目标函数
- 线性约束
- 线性系统模型
- 可用高效 QP 求解器

代表算法：

- Active Set MPC
- Interior Point MPC
- ADMM-MPC
- Fast Gradient MPC
- Riccati-based MPC
- Condensed QP-MPC
- Sparse QP-MPC

---

## 4.2 LP-MPC（线性规划 MPC）

目标和约束均为线性。

应用：

- 资源分配
- 能源调度
- 供应链
- 简化过程控制

---

## 4.3 NLP-MPC（非线性规划 MPC）

NMPC 通常会转化为 NLP。

代表方法：

- SQP-MPC
- Interior Point NMPC
- Multiple Shooting NMPC
- Direct Collocation NMPC
- Real-Time Iteration MPC
- Sequential Convex Programming MPC
- Augmented Lagrangian MPC

---

## 4.4 SOCP-MPC（二阶锥规划 MPC）

用于包含二阶锥约束的控制问题。

应用：

- 鲁棒控制
- 机器人力控制
- 电力系统
- 航天控制

---

## 4.5 SDP-MPC（半定规划 MPC）

用于矩阵不等式约束和鲁棒控制。

应用：

- 鲁棒稳定性
- LMI-MPC
- 椭球约束 MPC
- H∞ MPC

---

## 4.6 MILP / MIQP-MPC

含整数变量的 MPC。

主要类型：

- MILP-MPC
- MIQP-MPC
- MINLP-MPC
- 逻辑约束 MPC
- 混合整数经济 MPC
- 混合整数调度 MPC

应用：

- 开关控制
- 交通灯控制
- 能源设备启停
- 生产排程
- 车辆换挡
- 机器人接触模式选择

---

# 5. 按求解与实时实现方式分类

## 5.1 在线 MPC（Online MPC）

每个控制周期在线求解优化问题。

常见求解器：

- qpOASES
- OSQP
- IPOPT
- ACADOS
- FORCES Pro
- CVXGEN
- CasADi
- do-mpc
- MATLAB MPC Toolbox
- MPCTools
- GRAMPC

---

## 5.2 显式 MPC（Explicit MPC）

离线求解参数化优化问题，在线只查表或计算分段仿射控制律。

主要类型：

- 显式线性 MPC
- 显式二次规划 MPC
- 分段仿射显式 MPC
- 显式鲁棒 MPC
- 显式混合系统 MPC
- 显式神经网络近似 MPC

优点：

- 在线速度极快
- 适合嵌入式系统

缺点：

- 维度灾难严重
- 离线计算复杂

---

## 5.3 快速 MPC（Fast MPC）

面向实时系统的快速求解。

代表算法：

- Fast Gradient MPC
- Accelerated Gradient MPC
- Primal-Dual MPC
- ADMM-MPC
- Real-Time Iteration MPC
- Riccati Recursion MPC
- Sparse MPC
- Condensing MPC
- Partial Condensing MPC
- Warm-Start MPC
- Move Blocking MPC
- Early Termination MPC
- Inexact MPC

---

## 5.4 嵌入式 MPC（Embedded MPC）

面向 MCU、DSP、FPGA、车规芯片、机器人控制器。

主要技术：

- 固定点 MPC
- 代码生成 MPC
- 显式 MPC
- 快速 QP MPC
- 稀疏结构 MPC
- 低精度 MPC
- 神经网络近似 MPC
- FPGA-MPC
- GPU-MPC

---

## 5.5 近似 MPC（Approximate MPC）

用近似模型或近似求解器降低计算量。

主要方法：

- 神经网络近似 MPC
- 策略蒸馏 MPC
- 学习型显式 MPC
- 模仿 MPC
- 低阶模型 MPC
- 降维 MPC
- Koopman 线性化 MPC
- 局部线性 MPC
- 代理模型 MPC
- 模型降阶 MPC

---

# 6. 按控制结构分类

## 6.1 集中式 MPC（Centralized MPC）

一个控制器统一优化整个系统。

优点：

- 全局最优性更强
- 结构简单

缺点：

- 大规模系统计算量大
- 通信压力大
- 单点故障风险

应用：

- 小规模工业过程
- 单机器人
- 单车辆
- 小型能源系统

---

## 6.2 分布式 MPC（Distributed MPC, DMPC）

多个子系统各自求解局部 MPC，并通过通信协调。

主要类型：

- 分布式 MPC
- 协同分布式 MPC
- 非合作分布式 MPC
- 迭代式分布式 MPC
- 非迭代式分布式 MPC
- ADMM 分布式 MPC
- 双层分布式 MPC
- 分布式经济 MPC
- 分布式鲁棒 MPC
- 分布式随机 MPC

应用：

- 智能电网
- 多机器人
- 交通网络
- 大型化工过程
- 楼宇群控制
- 多智能体系统

---

## 6.3 去中心化 MPC（Decentralized MPC）

每个子系统独立控制，通信较少或没有通信。

特点：

- 简单可靠
- 扩展性强
- 全局性能可能下降

应用：

- 大规模弱耦合系统
- 工业装置群
- 多区域 HVAC
- 多车辆系统

---

## 6.4 层级 MPC（Hierarchical MPC）

不同层级负责不同时间尺度或决策粒度。

常见结构：

- 上层规划 + 下层 MPC
- 经济调度 + 跟踪控制
- 路径规划 + 轨迹跟踪
- 任务规划 + 动作控制
- 慢时间尺度优化 + 快时间尺度控制

应用：

- 自动驾驶
- 机器人
- 电力系统
- 生产调度
- 智能交通

---

## 6.5 多智能体 MPC（Multi-Agent MPC）

多个智能体协同控制。

主要方法：

- 多智能体 MPC
- 博弈 MPC
- 协同 MPC
- 非合作 MPC
- 均值场 MPC
- 编队 MPC
- 避碰 MPC
- 分布式多机器人 MPC
- 车队 MPC
- 群体 MPC

---

# 7. 按学习与数据结合分类

## 7.1 自适应 MPC（Adaptive MPC）

在线估计模型参数并更新控制器。

主要方法：

- 参数自适应 MPC
- 递归最小二乘 MPC
- 贝叶斯自适应 MPC
- 双控制 MPC
- 自校正 MPC
- 在线辨识 MPC
- LPV 自适应 MPC

应用：

- 参数随时间变化系统
- 老化设备
- 负载变化系统
- 机器人与未知环境交互

---

## 7.2 学习型 MPC（Learning-Based MPC）

将机器学习与 MPC 结合。

主要类别：

- Gaussian Process MPC
- Neural Network MPC
- Reinforcement Learning MPC
- Koopman MPC
- SINDy-MPC
- Bayesian MPC
- Safe Learning MPC
- Imitation Learning MPC
- Meta-Learning MPC
- Offline Learning MPC
- Online Learning MPC

---

## 7.3 Gaussian Process MPC（GP-MPC）

使用高斯过程建模系统误差或未知动力学。

常见形式：

- GP-NMPC
- GP-Robust MPC
- GP-Safe MPC
- GP-Chance-Constrained MPC
- GP-Residual MPC

优点：

- 能给出预测均值和不确定性
- 适合小数据系统
- 适合安全学习控制

---

## 7.4 Neural Network MPC（NN-MPC）

使用神经网络建模系统动力学或直接近似控制律。

主要类型：

- 神经网络动力学 MPC
- 神经网络残差 MPC
- 神经网络显式 MPC
- 深度 MPC
- LSTM-MPC
- Transformer-MPC
- Physics-Informed Neural MPC
- Diffusion-MPC
- World-Model MPC
- Latent Dynamics MPC

应用：

- 机器人
- 自动驾驶
- 复杂非线性系统
- 高维感知控制

---

## 7.5 Reinforcement Learning + MPC

强化学习与 MPC 的结合方式：

- MPC 作为 RL 策略
- RL 调参 MPC
- RL 学习 MPC 代价函数
- RL 学习终端价值函数
- RL 学习动力学模型
- MPC 作为安全层
- MPC 作为规划器，RL 作为低层控制
- Model-Based RL with MPC
- PETS
- PlaNet
- Dreamer-style latent MPC
- MPPI with learned dynamics
- Policy Distillation from MPC
- Imitation of MPC Expert

---

## 7.6 Koopman MPC

用 Koopman 算子将非线性系统提升到高维线性空间。

主要类型：

- EDMD-MPC
- Deep Koopman MPC
- Koopman Linear MPC
- Koopman Robust MPC
- Koopman NMPC

优点：

- 将非线性控制转化为近似线性控制
- 便于使用 QP 求解

---

## 7.7 Safe Learning MPC

在学习过程中保证安全。

主要方法：

- Safe MPC
- Safe Learning MPC
- Reachability-based MPC
- CBF-MPC
- Lyapunov Safe MPC
- GP Safe MPC
- Shielded MPC
- Backup Controller MPC
- Recursive Feasibility MPC

应用：

- 自动驾驶
- 机器人
- 人机交互
- 医疗设备
- 航空航天

---

# 8. 按预测模型来源分类

## 8.1 物理模型 MPC

基于机理方程。

例子：

- 牛顿力学模型
- 电路模型
- 热力学模型
- 化工反应模型
- 流体模型
- 刚体动力学模型

---

## 8.2 系统辨识 MPC

从输入输出数据中辨识模型。

方法：

- ARX-MPC
- ARMAX-MPC
- OE-MPC
- State-Space Identification MPC
- Subspace MPC
- N4SID-MPC
- PEM-MPC

---

## 8.3 灰箱 MPC

结合物理知识和数据学习。

方法：

- 物理模型 + 残差学习
- 参数辨识 MPC
- Physics-Informed MPC
- Hybrid Model MPC
- Semi-Parametric MPC

---

## 8.4 黑箱 MPC

完全依赖数据模型。

方法：

- 神经网络 MPC
- 高斯过程 MPC
- 随机森林 MPC
- 支持向量回归 MPC
- Koopman 学习 MPC
- Transformer 动力学 MPC

---

# 9. 按预测时域和采样方式分类

## 9.1 有限时域 MPC

最常见形式。

- 有限预测时域 MPC
- 有限控制时域 MPC
- 终端约束 MPC
- 终端成本 MPC

---

## 9.2 无限时域 MPC

理论上考虑无限未来。

- 无限时域 MPC
- 折扣无限时域 MPC
- 平均成本 MPC
- 稳态 MPC

---

## 9.3 移动阻塞 MPC（Move Blocking MPC）

减少控制变量数量。

方法：

- 固定 move blocking
- 自适应 move blocking
- 分段常数控制
- 分段线性控制

---

## 9.4 多速率 MPC（Multi-Rate MPC）

不同变量使用不同采样周期。

应用：

- 化工过程
- 电力系统
- 机器人
- 传感器异步系统

---

## 9.5 事件触发 MPC（Event-Triggered MPC）

只有事件发生时才更新控制。

优点：

- 降低计算与通信
- 适合网络控制系统

方法：

- Self-triggered MPC
- Event-triggered MPC
- Asynchronous MPC
- Communication-aware MPC

---

# 10. 按稳定性和可行性设计分类

## 10.1 终端约束 MPC

通过终端状态约束保证稳定性。

方法：

- 终端等式约束 MPC
- 终端集合约束 MPC
- 终端不变集 MPC
- 鲁棒终端集 MPC

---

## 10.2 终端成本 MPC

引入终端价值函数。

方法：

- LQR terminal cost MPC
- Lyapunov terminal cost MPC
- Learned terminal value MPC
- Approximate Dynamic Programming terminal cost MPC

---

## 10.3 递归可行 MPC

保证如果当前优化可行，下一个时刻也可行。

方法：

- 递归可行约束 MPC
- Backup policy MPC
- Invariant set MPC
- Constraint tightening MPC

---

## 10.4 Lyapunov MPC

使用 Lyapunov 函数保证稳定性。

方法：

- Lyapunov constraint MPC
- Control Lyapunov Function MPC
- CLF-MPC
- CLF-CBF-MPC

---

## 10.5 Suboptimal / Inexact MPC

允许优化问题不完全求解。

方法：

- Suboptimal MPC
- Inexact MPC
- Early-stopping MPC
- Real-time feasible MPC
- Anytime MPC
- Warm-start MPC

---

# 11. 按应用领域分类

## 11.1 工业过程 MPC

代表算法：

- DMC（Dynamic Matrix Control）
- QDMC（Quadratic Dynamic Matrix Control）
- GPC（Generalized Predictive Control）
- IDCOM
- RMPCT
- Shell MPC
- Aspen DMCplus
- 工业多变量 MPC
- 约束过程 MPC
- 经济过程 MPC

应用：

- 炼油
- 化工
- 制药
- 造纸
- 食品加工
- 水处理

---

## 11.2 车辆与自动驾驶 MPC

主要算法：

- 车辆路径跟踪 MPC
- 车辆横向控制 MPC
- 纵向速度 MPC
- 横纵向耦合 MPC
- 非线性车辆动力学 MPC
- 轮胎约束 MPC
- 避障 MPC
- 车队 MPC
- 自动泊车 MPC
- 漂移控制 MPC
- 稳定性控制 MPC
- 风险敏感自动驾驶 MPC

---

## 11.3 机器人 MPC

主要算法：

- 机械臂 NMPC
- 移动机器人 MPC
- 四足机器人 MPC
- 人形机器人 MPC
- 接触丰富 MPC
- Whole-Body MPC
- Legged Locomotion MPC
- Manipulation MPC
- Visual MPC
- Latent MPC
- Diffusion-MPC
- Sampling-Based MPC
- MPPI

---

## 11.4 无人机与航空航天 MPC

主要算法：

- UAV NMPC
- Quadrotor MPC
- 航迹跟踪 MPC
- 姿态 MPC
- 编队飞行 MPC
- 避障 MPC
- 航天器轨道 MPC
- 再入制导 MPC
- 推力约束 MPC

---

## 11.5 电力系统与能源 MPC

主要算法：

- 智能电网 MPC
- 微电网 MPC
- 储能 MPC
- 电池管理 MPC
- 电动车充电 MPC
- HVAC-MPC
- 建筑能耗 MPC
- 风电 MPC
- 光伏 MPC
- 经济调度 MPC
- 鲁棒能源 MPC
- 随机能源 MPC

---

## 11.6 交通系统 MPC

主要算法：

- 交通信号 MPC
- 高速路匝道控制 MPC
- 城市路网 MPC
- 公交调度 MPC
- 轨道交通 MPC
- 车队协同 MPC
- 动态路径诱导 MPC
- 拥堵控制 MPC

---

## 11.7 医疗与生物系统 MPC

主要算法：

- 人工胰腺 MPC
- 胰岛素给药 MPC
- 麻醉控制 MPC
- 肿瘤治疗 MPC
- 药物剂量 MPC
- 生物反应器 MPC
- 个性化医疗 MPC

---

## 11.8 金融与经济 MPC

主要算法：

- 投资组合 MPC
- 风险约束 MPC
- CVaR-MPC
- 库存金融 MPC
- 动态定价 MPC
- 经济政策 MPC
- 供应链 MPC

---

## 11.9 计算机系统与网络 MPC

主要算法：

- 数据中心能耗 MPC
- 云资源调度 MPC
- 网络拥塞控制 MPC
- 边缘计算 MPC
- CPU/GPU 资源 MPC
- 缓存控制 MPC
- 视频码率 MPC

---

# 12. 采样型 MPC 与随机优化型 MPC

## 12.1 MPPI（Model Predictive Path Integral Control）

MPPI 是一种采样型 MPC，常用于机器人和自动驾驶。

特点：

- 使用大量随机轨迹采样
- 根据轨迹代价加权控制
- 适合非线性和非凸问题
- GPU 并行友好

变体：

- Standard MPPI
- Tube-MPPI
- Robust MPPI
- Tsallis-MPPI
- Risk-Sensitive MPPI
- Constrained MPPI
- Learned Dynamics MPPI

---

## 12.2 CEM-MPC（Cross-Entropy Method MPC）

用交叉熵方法优化控制序列。

特点：

- 采样候选控制序列
- 选择 elite samples
- 更新采样分布
- 常用于模型强化学习

变体：

- CEM-MPC
- iCEM-MPC
- CEM with colored noise
- CEM with policy prior
- CEM with learned dynamics

---

## 12.3 Random Shooting MPC

随机采样控制序列并选择最优。

类型：

- Random Shooting MPC
- Guided Random Shooting MPC
- Neural-Guided Shooting MPC
- Parallel Shooting MPC

---

## 12.4 iLQR / DDP-MPC

基于局部二次近似和动态规划。

主要算法：

- iLQR-MPC
- DDP-MPC
- Constrained iLQR-MPC
- Sequential LQ MPC
- SLQ-MPC
- Real-time iLQR MPC

应用：

- 机器人
- 自动驾驶
- 腿足控制
- 轨迹优化

---

# 13. 与经典预测控制相关的历史算法

## 13.1 工业预测控制早期算法

- IDCOM
- DMC（Dynamic Matrix Control）
- QDMC
- GPC（Generalized Predictive Control）
- EPSAC
- PFC（Predictive Functional Control）
- RMPCT
- SMOC
- MPC based on FIR step response model
- MPC based on impulse response model

---

## 13.2 广义预测控制（GPC）

基于 CARIMA 模型。

特点：

- 适合单输入单输出和多输入多输出系统
- 曾广泛用于工业过程控制
- 与自适应控制联系紧密

---

## 13.3 动态矩阵控制（DMC）

基于阶跃响应模型。

特点：

- 工业 MPC 代表方法
- 适合多变量约束过程
- 在炼油化工领域应用广泛

---

# 14. MPC 与现代 AI / 大模型 / Agent 的结合

## 14.1 MPC + 世界模型

主要类型：

- World Model MPC
- Latent Dynamics MPC
- Dreamer-style MPC
- PlaNet-style MPC
- MuZero-style planning with MPC
- Neural Simulation MPC

---

## 14.2 MPC + 大模型 Agent

可能方向：

- LLM 高层任务规划 + MPC 低层控制
- VLM 感知理解 + MPC 轨迹控制
- LLM 生成约束/目标 + MPC 求解
- Agent 选择子任务 + MPC 执行
- 大模型辅助代价函数设计
- 大模型辅助动力学建模
- 大模型解释 MPC 决策

---

## 14.3 MPC + 扩散模型

主要类型：

- Diffusion Policy + MPC
- Diffusion Planner MPC
- Diffusion Trajectory MPC
- Score-based MPC
- Generative MPC
- Sampling-based Diffusion MPC

应用：

- 机器人操作
- 自动驾驶轨迹生成
- 多模态路径规划

---

## 14.4 MPC + 强化学习

主要组合：

- MPC as Policy
- MPC-Guided RL
- RL-Tuned MPC
- MPC Safety Layer
- Model-Based RL with MPC
- Offline RL + MPC
- Safe RL + MPC
- Differentiable MPC
- Policy Distillation from MPC

---

# 15. 按数学理论基础分类

## 15.1 最优控制型 MPC

- LQR-MPC
- LQG-MPC
- H∞ MPC
- Differential Dynamic Programming MPC
- Pontryagin MPC
- Hamilton-Jacobi MPC

---

## 15.2 动态规划型 MPC

- Value Function MPC
- Approximate Dynamic Programming MPC
- Rollout MPC
- Adaptive Dynamic Programming MPC
- Reinforcement Learning MPC

---

## 15.3 鲁棒优化型 MPC

- Min-Max MPC
- Tube MPC
- Set-Invariance MPC
- Adjustable Robust MPC
- Robust Counterpart MPC

---

## 15.4 随机优化型 MPC

- Chance-Constrained MPC
- Scenario MPC
- Sample Average Approximation MPC
- Distributionally Robust MPC
- Risk-Sensitive MPC

---

## 15.5 凸优化型 MPC

- QP-MPC
- LP-MPC
- SOCP-MPC
- SDP-MPC
- ADMM-MPC
- Primal-Dual MPC

---

## 15.6 非凸优化型 MPC

- NMPC
- MINLP-MPC
- Sampling MPC
- CEM-MPC
- MPPI
- DDP-MPC
- Mixed-Integer MPC

---

# 16. MPC 算法总清单

## 16.1 基础 MPC

- Model Predictive Control
- Receding Horizon Control
- Linear MPC
- Nonlinear MPC
- Tracking MPC
- Regulation MPC
- Output Feedback MPC
- State Feedback MPC
- Continuous-Time MPC
- Discrete-Time MPC

## 16.2 线性与非线性 MPC

- LTI-MPC
- LTV-MPC
- LPV-MPC
- Linear Quadratic MPC
- Constrained Linear MPC
- NMPC
- RTI-NMPC
- Direct Collocation MPC
- Multiple Shooting MPC
- Single Shooting MPC
- Pseudospectral MPC

## 16.3 鲁棒与随机 MPC

- Robust MPC
- Min-Max MPC
- Tube MPC
- Elastic Tube MPC
- Homothetic Tube MPC
- Constraint Tightening MPC
- Set-Membership MPC
- Stochastic MPC
- Chance-Constrained MPC
- Scenario MPC
- Distributionally Robust MPC
- Risk-Sensitive MPC
- CVaR-MPC

## 16.4 经济与多目标 MPC

- Economic MPC
- Nonlinear Economic MPC
- Robust Economic MPC
- Stochastic Economic MPC
- Distributed Economic MPC
- Multi-Objective MPC
- Lexicographic MPC
- Pareto MPC

## 16.5 分布式与多智能体 MPC

- Distributed MPC
- Decentralized MPC
- Cooperative MPC
- Noncooperative MPC
- Hierarchical MPC
- Multi-Agent MPC
- Game-Theoretic MPC
- Mean-Field MPC
- Formation MPC
- Platoon MPC

## 16.6 学习型 MPC

- Learning-Based MPC
- Adaptive MPC
- Data-Driven MPC
- DeePC
- GP-MPC
- NN-MPC
- Koopman MPC
- SINDy-MPC
- Safe Learning MPC
- Bayesian MPC
- RL-MPC
- Imitation Learning MPC
- Meta-Learning MPC

## 16.7 快速与嵌入式 MPC

- Explicit MPC
- Fast MPC
- Embedded MPC
- ADMM-MPC
- Fast Gradient MPC
- Active Set MPC
- Interior Point MPC
- Sparse MPC
- Condensed MPC
- Partial Condensing MPC
- Warm-Start MPC
- Inexact MPC
- Anytime MPC

## 16.8 采样型 MPC

- MPPI
- CEM-MPC
- Random Shooting MPC
- Guided Shooting MPC
- iLQR-MPC
- DDP-MPC
- SLQ-MPC
- Monte Carlo MPC
- Particle MPC

## 16.9 混合与离散 MPC

- Hybrid MPC
- Mixed-Integer MPC
- MILP-MPC
- MIQP-MPC
- MINLP-MPC
- PWA-MPC
- MLD-MPC
- Switching MPC
- Automata MPC
- Petri-Net MPC

## 16.10 安全 MPC

- Safety-Critical MPC
- CBF-MPC
- CLF-MPC
- CLF-CBF-MPC
- Reachability MPC
- Shielded MPC
- Backup MPC
- Fault-Tolerant MPC
- Recursive Feasibility MPC

---

# 17. 推荐学习路线

## 17.1 入门路线

1. 线性系统基础
2. 最优控制基础
3. 线性 MPC
4. QP 求解
5. 约束处理
6. 稳定性与终端约束
7. NMPC
8. 鲁棒 MPC
9. 随机 MPC
10. 学习型 MPC

---

## 17.2 工程路线

1. 从 LQR 理解 MPC
2. 学习线性 MPC 建模
3. 使用 OSQP / qpOASES / CasADi 实现
4. 学习约束建模
5. 学习 NMPC 与 ACADOS
6. 加入扰动和鲁棒约束
7. 做仿真测试
8. 转嵌入式部署
9. 做在线调参和稳定性检查
10. 与学习模型结合

---

## 17.3 自动驾驶 / 机器人路线

1. 车辆或机器人动力学
2. 轨迹跟踪 MPC
3. 非线性 MPC
4. 约束建模
5. 避障约束
6. iLQR / DDP
7. MPPI / CEM
8. 学习型动力学
9. Safe MPC
10. MPC + RL / Diffusion Policy

---

# 18. 总结

MPC 算法可以概括为以下几个核心家族：

1. **基础 MPC**：线性 MPC、非线性 MPC、跟踪 MPC、调节 MPC  
2. **不确定性 MPC**：鲁棒 MPC、随机 MPC、分布鲁棒 MPC、风险敏感 MPC  
3. **优化形式 MPC**：QP-MPC、NLP-MPC、MILP-MPC、SOCP-MPC、SDP-MPC  
4. **实时实现 MPC**：显式 MPC、快速 MPC、嵌入式 MPC、近似 MPC  
5. **结构化 MPC**：集中式、分布式、去中心化、层级、多智能体 MPC  
6. **学习型 MPC**：数据驱动 MPC、GP-MPC、NN-MPC、Koopman MPC、RL-MPC  
7. **安全 MPC**：CBF-MPC、CLF-MPC、Reachability MPC、Safe Learning MPC  
8. **采样型 MPC**：MPPI、CEM-MPC、Random Shooting、iLQR、DDP  
9. **应用专用 MPC**：工业过程、自动驾驶、机器人、电力、交通、医疗、金融等  

一句话总结：

> MPC 是一个“模型 + 预测 + 约束优化 + 滚动执行”的控制框架。  
> 它的算法分类，本质上取决于系统模型、优化目标、不确定性建模、约束形式、求解器和应用场景。

---

# 参考资料

- J. B. Rawlings, D. Q. Mayne, M. Diehl, *Model Predictive Control: Theory, Computation, and Design*.
- Alberto Bemporad & Manfred Morari, *Robust Model Predictive Control: A Survey*.
- Timm Faulwasser et al., *Economic Nonlinear Model Predictive Control*.
- Matthias A. Müller et al., *Economic and Distributed Model Predictive Control*.
- K. Zhang et al., *A Survey on Learning-Based Model Predictive Control*.
- Springer, *Handbook of Model Predictive Control*.
- Recent surveys and papers on robust MPC, stochastic MPC, learning-based MPC, explicit MPC, distributed MPC, and MPC-RL integration.
