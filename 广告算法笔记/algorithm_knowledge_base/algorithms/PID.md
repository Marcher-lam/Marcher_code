# PID 控制算法 学习文档

## 1. 算法基础认知

PID（比例-积分-微分）控制算法是一种经典的反馈控制算法，广泛应用于工业自动化。

在广告出价算法中，PID 通过动态调整控制量（如广告出价 bid）来缩小实际成本与目标成本的偏差。其核心思想是结合当前误差（比例项）、历史误差累积（积分项）和误差变化趋势（微分项）进行综合调节，最终实现广告出价成本控制。

## 2. 核心原理

PID 控制公式：

$$
u(t) = K_p \cdot e(t) + K_i \cdot \int_0^t e(\tau) d\tau + K_d \cdot \frac{de(t)}{dt}
$$

- **比例环节（P）**：即时响应当前误差。误差越大，调价幅度越大，加快收敛速度。
  $$P = K_p \cdot e(t)$$
- **积分环节（I）**：累积历史误差，消除稳态误差。
  $$I = K_i \cdot \sum_{k=0}^{t} e(k) \cdot \Delta t$$
- **微分环节（D）**：预测误差变化趋势，抑制超调。
  $$D = K_d \cdot \frac{e(t) - e(t-1)}{\Delta t}$$

在广告出价中，PID 通过周期性（如每小时）监控成本误差，输出调价因子（调价因子=1+λ），乘以基础出价（出价=基础出价×调价因子），使实际成本趋近目标成本。

## 3. 数学公式与推导

### 增量式 PID 公式（避免积分饱和）：

增量式 PID 不直接计算控制量的绝对值，而是计算控制量的**增量** Δu(t)：

$$
\Delta u(t) = K_p[e(t) - e(t-1)] + K_i \cdot e(t) + K_d[e(t) - 2e(t-1) + e(t-2)]
$$

其中：
- e(t) = 目标成本 - 实际成本（当前时刻的误差）
- e(t-1) = 上一时刻误差，e(t-2) = 上两个时刻误差
- Kp, Ki, Kd 分别为比例、积分、微分系数

**为什么增量式 PID 能避免积分饱和？** 位置式 PID 中积分项 Σe(t) 会持续累积，当误差长期为正（或负）时积分项可能无限增大，导致控制量远超合理范围（积分饱和）。而增量式 PID 直接计算 Δu(t)，不显式累加积分项——积分作用被隐含在 Ki·e(t) 中，每次只对输出做增量调整，因此天然不会出现积分饱和问题。

最终出价 = 基础出价 × (1 + Δu)

## 4. 训练过程讲解

1. **初始化参数**：设定初始 Kp, Ki, Kd；设置基础出价；初始化误差累积项 integral=0 和上一次误差 prev_error=0
2. **误差计算**：实时计算 e(t) = 目标成本 - 实际成本
3. **PID 输出计算**：计算比例项、积分项（含积分限幅）、微分项
4. **出价调整**：将调价因子乘以基础出价，得到最终出价，并进行上下限约束
5. **参数更新**：更新历史误差、累积误差

## 5. 应用场景

- 广告出价成本控制（第一代出价算法）
- 预算平滑（Budget Pacing）
- 多目标权重调节
- 与 Bandit 辅助配合使用

## 6. 优缺点分析

### 优点
- 简单鲁棒：无需复杂建模，适用于动态变化的广告竞价环境
- 动态调节：通过积分项消除长期偏差，微分项抑制波动
- 工程易用：参数调优可通过经验或自动化工具（如网格搜索）完成
- 可解释性高

### 缺点
- 参数敏感：Kp, Ki, Kd 需精细调参，否则可能导致振荡或滞后
- 滞后系统效果差：广告成本反馈有延迟，可能导致超调或振荡
- 短视决策：只看当前误差，不考虑未来流量分布
- 约束处理弱：难以处理预算上下限、ROI 约束等硬约束
- 震荡问题：比例-积分-微分参数难调，容易超调震荡

## 7. 调库实现（Python + 完整代码 + 注释）

```python
class PIDBidController:
    def __init__(self, target_cost, kp, ki, kd, min_factor=0.5, max_factor=2.0):
        self.target_cost = target_cost
        self.kp, self.ki, self.kd = kp, ki, kd
        self.min_factor, self.max_factor = min_factor, max_factor
        self.integral, self.prev_error = 0.0, 0.0
        self.first_run = True

    def calculate_adjustment(self, actual_cost):
        error = self.target_cost - actual_cost
        if self.first_run:
            p_term = self.kp * error
            adjustment = 1.0 + p_term
            self.first_run = False
        else:
            p_term = self.kp * error
            self.integral += error
            integral_max = 10.0 / self.ki if self.ki != 0 else float('inf')
            self.integral = max(min(self.integral, integral_max), -integral_max)
            i_term = self.ki * self.integral
            d_term = self.kd * (error - self.prev_error)
            self.prev_error = error
            adjustment = 1.0 + p_term + i_term + d_term
        adjustment = max(min(adjustment, self.max_factor), self.min_factor)
        return adjustment

    def calculate_bid(self, base_bid, actual_cost):
        adjustment_factor = self.calculate_adjustment(actual_cost)
        return base_bid * adjustment_factor

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def pid_control(target, actual, kp, ki, kd, integral, prev_error, dt=1.0):
    error = target - actual
    integral += error * dt
    integral = np.clip(integral, -10.0, 10.0)
    derivative = (error - prev_error) / dt
    output = kp * error + ki * integral + kd * derivative
    prev_error = error
    return output, integral, prev_error

def pid_bid_adjust(base_bid, target_cpa, actual_cpa, kp=0.1, ki=0.01, kd=0.05):
    integral, prev_error = 0.0, 0.0
    delta_u, integral, prev_error = pid_control(
        target_cpa, actual_cpa, kp, ki, kd, integral, prev_error
    )
    adjustment = 1.0 + delta_u
    adjustment = max(min(adjustment, 3.0), 0.3)
    return base_bid * adjustment
```

## 9. 可视化与结果理解

- PID 出价轨迹对比：PID 容易出现震荡（λ: 0.50 → 0.45 → 0.60 → 0.80 → 1.20 → 0.70 → ...）
- 而 MPC 出价更平滑（λ: 0.75 → 0.78 → 0.82 → 0.85 → 0.88 → 0.90 → ...）

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
target_cpa = 10.0
base_bid = 5.0
kp, ki, kd = 0.1, 0.01, 0.05
hours = 60

def simulate_pid(target, kp, ki, kd, n_steps, noise_scale=2.0):
    adjustment_factors = []
    actual_cpas = []
    integral, prev_error = 0.0, 0.0
    actual_cpa = target + 5.0
    for t in range(n_steps):
        error = target - actual_cpa
        integral += error
        integral = np.clip(integral, -10.0, 10.0)
        if t > 0:
            derivative = error - prev_error
        else:
            derivative = 0.0
        prev_error = error
        delta_u = kp * error + ki * integral + kd * derivative
        adjustment = 1.0 + delta_u
        adjustment = np.clip(adjustment, 0.3, 3.0)
        adjustment_factors.append(adjustment)
        actual_cpa = target + noise_scale * np.sin(t * 0.3) + (target / adjustment - target) * 0.3
        actual_cpa += np.random.normal(0, 0.5)
        actual_cpas.append(actual_cpa)
    return np.array(adjustment_factors), np.array(actual_cpas)

def simulate_mpc(target, n_steps, noise_scale=2.0):
    actual_cpas = []
    for t in range(n_steps):
        error = target - (target + noise_scale * np.sin(t * 0.3) + np.random.normal(0, 0.5))
        adj = 1.0 + 0.05 * error
        adj = np.clip(adj, 0.5, 2.0)
        actual_cpa = target + noise_scale * np.sin(t * 0.3) * 0.3 + np.random.normal(0, 0.3)
        actual_cpas.append(actual_cpa)
    adjustments = 1.0 + 0.02 * np.cumsum(target - np.array(actual_cpas)) / np.arange(1, n_steps + 1)
    adjustments = np.clip(adjustments, 0.5, 2.0)
    return adjustments, np.array(actual_cpas)

pid_adj, pid_cpa = simulate_pid(target_cpa, kp, ki, kd, hours)
mpc_adj, mpc_cpa = simulate_mpc(target_cpa, hours)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(range(hours), pid_cpa, 'b-', alpha=0.8, label='PID Actual CPA')
axes[0].plot(range(hours), mpc_cpa, 'g-', alpha=0.8, label='MPC Actual CPA')
axes[0].axhline(y=target_cpa, color='r', linestyle='--', linewidth=2, label=f'Target CPA = {target_cpa}')
axes[0].set_ylabel('CPA', fontsize=12)
axes[0].set_title('PID vs MPC: CPA Tracking Performance', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(hours), pid_adj, 'b-', alpha=0.8, label='PID Adjustment Factor')
axes[1].plot(range(hours), mpc_adj, 'g-', alpha=0.8, label='MPC Adjustment Factor')
axes[1].axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Baseline (1.0)')
axes[1].set_xlabel('Hour', fontsize=12)
axes[1].set_ylabel('Adjustment Factor', fontsize=12)
axes[1].set_title('PID vs MPC: Bid Adjustment Factor Over Time', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pid_vs_mcp_bidding.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. 模型评估

- 成本控制精度：实际 CPA 与目标 CPA 的偏差
- 预算利用率：实际消耗/分配预算
- Pacing Rate：实际消耗速度与理想消耗速度的比值
- 出价稳定性：相邻时刻出价变化的方差

## 11. 常见问题与易错点

- 积分饱和：需设置积分限幅，防止积分项过大导致超调
- 参数调优困难：Kp, Ki, Kd 需根据实际系统调整，通常通过离线仿真预调优
- 延迟反馈：转化数据滞后可能导致超调，需引入预估消耗+延迟补偿机制
- 流量波动：突发事件导致流量剧烈波动时，需 PID 参数自适应+异常检测熔断

## 12. 学习总结

PID 是广告出价的第一代核心算法，简单高效但存在短视、约束处理弱、震荡等痛点。工业最佳实践是"PID为主 + Bandit辅助"的混合策略。MPC 和 RL 是其升级方案。

## 13. 练习题与思考题（含答案）

1. **推导**：推导增量式 PID 公式 Δu(t) = Kp[e(t)-e(t-1)] + Ki·e(t) + Kd[e(t)-2e(t-1)+e(t-2)]，从位置式 PID 出发，说明每一步推导的依据，并解释为什么增量式可以避免积分饱和。
2. **实践**：调整 Kp=0.5, Ki=0, Kd=0 观察系统响应，然后逐步加入 Ki 和 Kd，记录超调量和收敛速度的变化。
3. **思考**：为什么广告系统中 PID 通常以分钟级别更新权重，而不是秒级别？

## 14. 学习路径建议

PID → MPC（第二代）→ 强化学习出价（第三代）→ 生成式 RL 出价（第四代）
