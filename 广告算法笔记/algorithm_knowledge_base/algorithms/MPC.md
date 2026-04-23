# MPC 出价算法 学习文档

## 1. 算法基础认知

MPC（Model Predictive Control，模型预测控制）是广告智能出价领域的第二代核心算法。它源自工业控制领域，核心思想是：在每个决策时刻，利用预测模型对未来多个时间步的系统状态进行预测，然后在约束条件下求解最优控制序列，但只执行第一步动作，下一时刻重新预测、重新优化——即"滚动优化 + 反馈校正"。

在广告出价场景中，MPC 解决了第一代 PID 控制器的核心缺陷：PID 只看当前误差，无法预见未来流量变化，导致预算分配不均、尾部超投/欠投。

### PID 的三大痛点 vs MPC 解法

| 痛点 | PID 具体表现 | MPC 解法 |
|------|-------------|---------|
| 短视决策 | 只看当前误差 e(t)，不考虑未来流量分布 | 预测未来 H 步流量，全局优化 |
| 约束处理弱 | 难以处理预算上下限、ROI 约束等硬约束 | 显式约束优化（线性/二次规划） |
| 震荡问题 | 比例-积分-微分参数难调，容易超调震荡 | 滚动优化天然平滑，自带稳定性 |

## 2. 核心原理

### MPC 三大核心要素

1. **预测模型**：利用系统模型预测未来 H 个时间步的状态变化
2. **滚动优化**：在每个时间步，求解有限时域的最优化问题
3. **反馈校正**：只执行第一步，下一时刻用实际观测重新优化

### 状态空间定义

| 状态变量 | 含义 |
|---------|------|
| B(t) | 时刻 t 的剩余预算 |
| C(t) | 时刻 t 的累计转化数 |
| ROI(t) | 时刻 t 的累计 ROI |

**控制变量**：λ(t) = 出价调节系数（bid multiplier），实际出价 = 基础出价 b₀ × λ(t)

### 预测模型

**预算消耗预测**：
$$\hat{B}(t+k) = B(t) - \sum_{j=0}^{k-1} \hat{V}(t+j) \cdot \hat{w}(t+j) \cdot \hat{c}(t+j, \lambda_j)$$

**转化数预测**：
$$\hat{C}(t+k) = C(t) + \sum_{j=0}^{k-1} \hat{V}(t+j) \cdot \hat{w}(t+j) \cdot \hat{r}(t+j)$$

**竞胜率模型**（Logistic 近似）：
$$\hat{w}(\lambda) = \frac{1}{1 + \exp(-\beta \cdot (\lambda \cdot b_0 - \hat{p}))}$$

## 3. 数学公式与推导

### 优化目标一：预算均匀消耗（Budget Pacing）

$$
\min_{\lambda(t:t+H-1)} \sum_{k=0}^{H-1} \left(\hat{B}(t+k) - B_{target}(t+k)\right)^2 + \mu \sum_{k=0}^{H-2} (\lambda_{k+1} - \lambda_k)^2
$$

### 优化目标二：ROI 约束下转化最大化

$$
\max_{\lambda(t:t+H-1)} \sum_{k=0}^{H-1} \hat{V}_k \cdot \hat{w}_k \cdot \hat{r}_k
$$

约束条件：
- 预算约束：∑cost ≤ B_remain
- ROI 约束：ROI(t+H) ≥ ROI_target
- 出价范围：λ_min ≤ λ(t+k) ≤ λ_max
- 平滑约束：|λ_{k+1} - λ_k| ≤ Δ_max

## 4. 训练过程讲解

1. **预测模型构建**：利用历史数据训练流量预测、竞胜率预测等模型
2. **预测时域选择**：推荐 H=10~20，控制周期 1~5 分钟
3. **优化求解**：分段线性近似 + QP（二次规划）求解
4. **执行与反馈**：只执行第一步动作，下一时刻重新预测优化

## 5. 应用场景

| 场景 | 适用度 | 原因 |
|------|--------|------|
| 预算均匀消耗（Pacing） | ★★★★★ | MPC 的经典应用场景 |
| ROI 约束出价 | ★★★★ | 显式约束优化的天然优势 |
| 多目标出价（成本+量+ROI） | ★★★★ | 多约束优化框架 |
| 实时竞价（RTB） | ★★★ | 需要快速求解，适合线性 MPC |
| 长周期策略优化 | ★★ | 有限时域，不如 RL |

## 6. 优缺点分析

### 优势
1. 前瞻性决策：利用预测信息，避免 PID 的短视问题
2. 显式约束处理：天然支持预算、ROI、出价范围等多种约束
3. 平滑控制：滚动优化天然产生平滑的出价轨迹
4. 可解释性强：每一步的决策都有明确的数学依据
5. 鲁棒性好：反馈校正机制容忍预测误差

### 局限性
1. 依赖预测精度：预测模型不准时，MPC 效果下降
2. 计算成本：每个控制周期需求解优化问题（但线性 MPC 可毫秒级求解）
3. 模型简化：实际竞价环境高度非线性，线性化可能损失精度
4. 无法学习长期策略：MPC 是有限时域优化，不具备 RL 的长期策略学习能力

## 7. 调库实现（Python + 完整代码 + 注释）

```python
import numpy as np
from scipy.optimize import minimize

class MPCBidController:
    def __init__(self, total_budget, base_bid, H=12, dt_minutes=5):
        self.total_budget = total_budget
        self.base_bid = base_bid
        self.H = H
        self.dt = dt_minutes

    def predict_traffic(self, t, H):
        return np.random.uniform(800, 1200, H)

    def win_rate(self, lam, p_hat, beta=1.0):
        return 1.0 / (1.0 + np.exp(-beta * (lam * self.base_bid - p_hat)))

    def objective(self, lambdas, B_target, traffic, p_hat, cvr, mu=0.01):
        B = self.total_budget
        cost = 0
        loss = 0
        for k in range(self.H):
            w = self.win_rate(lambdas[k], p_hat[k])
            c = traffic[k] * w * lambdas[k] * self.base_bid / 1000
            cost += c
            B -= c
            loss += (B - B_target[k]) ** 2
        for k in range(self.H - 1):
            loss += mu * (lambdas[k+1] - lambdas[k]) ** 2
        return loss

    def solve(self, current_state, traffic_forecast, price_forecast):
        B_target = np.linspace(current_state['budget'], 0, self.H)
        x0 = np.ones(self.H)
        bounds = [(0.3, 3.0)] * self.H
        result = minimize(
            self.objective, x0,
            args=(B_target, traffic_forecast, price_forecast, 0.03),
            bounds=bounds, method='SLSQP'
        )
        return result.x[0]

    def compute_bid(self, base_bid, current_state, traffic_forecast, price_forecast):
        lam = self.solve(current_state, traffic_forecast, price_forecast)
        return base_bid * lam
```

## 8. 手工代码实现（核心算法手写 + 注释）

```python
import numpy as np

def mpc_bid_control(remaining_budget, total_budget, time_elapsed, total_time,
                    traffic_forecast, price_forecast, base_bid, H=12):
    ideal_budget_per_step = remaining_budget / (total_time - time_elapsed)
    lambdas = []
    budget_left = remaining_budget
    for k in range(H):
        traffic_k = traffic_forecast[k]
        price_k = price_forecast[k]
        ideal_spend_k = ideal_budget_per_step
        win_rate = 1.0 / (1.0 + np.exp(-(base_bid - price_k)))
        estimated_spend = traffic_k * win_rate * base_bid / 1000
        lam = ideal_spend_k / max(estimated_spend, 0.01)
        lam = np.clip(lam, 0.3, 3.0)
        lambdas.append(lam)
        budget_left -= estimated_spend * lam
    return base_bid * lambdas[0]
```

## 9. 可视化与结果理解

### MPC vs PID 出价轨迹对比

**PID 做法**（10:00 超投 1,250 元）：
- 看到超投 → 立即大幅降低出价（λ=0.50）
- 后续流量低谷期严重欠投 → 再次大幅提高出价
- 出价震荡：λ: 0.50 → 0.45 → 0.60 → 0.80 → 1.20 → 0.70 → ...
- 预算分配不均

**MPC 做法**（同样超投 1,250 元）：
- 预测到高峰即将过去 → 适度降低出价（而非大幅）
- 高峰期少花钱，低谷期正常出价
- 预算平滑消耗
- 出价平稳：λ: 0.75 → 0.78 → 0.82 → 0.85 → 0.88 → 0.90 → ...

## 10. 模型评估

- 预算利用率（目标 95%~100%）
- Pacing Rate（目标 ≈1.0）
- 出价平滑度（相邻时刻出价变化方差）
- CPA/ROAS 达成率

## 11. 常见问题与易错点

- 预测模型精度不足时效果下降
- 预测时域 H 选择不当（过短短视，过长计算量大）
- 控制周期与流量波动周期不匹配
- 线性化近似在高非线性场景下精度不够

## 12. 学习总结

MPC = 预测 + 滚动优化 + 反馈校正，是 PID 的全面升级。定位：第二代出价算法，介于 PID（太简单）和 RL（太复杂）之间的最佳平衡点。

核心公式速查：
- 滚动优化：min Σ L(x, u) + Vf
- 预算跟踪：min Σ(B - B_target)² + μΣ(Δλ)²
- 竞胜率：w(λ) = σ(β(λb₀ - p))
- 线性 MPC：u^THu + f^Tu

## 13. 练习题与思考题（含答案）

1. **推导**：推导 MPC 预算均匀消耗目标函数的梯度。
2. **实践**：对比不同预测时域 H（6/12/24）下的出价效果。
3. **思考**：为什么 MPC 适合 ROI 约束出价场景？

## 14. 学习路径建议

PID → **MPC** → 强化学习出价（DQN/PPO）→ 生成式 RL（Decision Transformer/Diffusion）
