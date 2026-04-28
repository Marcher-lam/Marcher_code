# 模型预测控制(MPC) 学习文档

> MPC在每个时刻求解一个有限视野的最优控制问题，是化工、自动驾驶和机器人领域的主流控制方法。

> 来源线索：本节内容根据原书中关于"Model Predictive Control"的相关章节(Ch 2.1.13)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：模型预测控制（MPC）在每个决策时刻，基于系统模型预测未来有限步的行为，求解一个开环最优控制问题，但只执行第一步控制，然后滚动向前。

**直觉类比**：你开车在弯道上行驶。你根据路况模型预测未来10秒的最佳路线，计算出方向盘转角序列。但你只执行第一秒的转向，然后重新预测和规划。这种"边走边规划"的方式让你能实时应对路况变化。

**历史背景**：MPC起源于1970年代的化工过程控制（Richalet et al., Cutler & Ramaker）。由于其能显式处理约束，在化工、石油等行业迅速普及。近年来在自动驾驶（模型预测轨迹跟踪）和机器人（运动规划）中广泛应用。

**算法定位**：直接前瞻策略(DLA)/最优控制方法。在原书框架中，MPC属于直接前瞻策略（Ch 19），通过求解前瞻模型的最优控制来近似最优策略。

**前置知识**：最优控制、线性系统、优化基础、Python。

## 2. 核心原理

**核心思想**：MPC的核心是"滚动优化"。每一步：(1)测量当前状态；(2)用系统模型预测未来$H$步的行为；(3)求解最优控制序列$\{u_t, u_{t+1}, ..., u_{t+H-1}\}$；(4)只执行$u_t$；(5)滚动到下一步重复。

**为什么只执行第一步？**因为模型有误差、干扰存在。执行一步后重新测量实际状态，可以校正预测误差。

**工作流程**：

1. 测量当前状态$x_t$
2. 基于模型$\dot{x}=f(x,u)$，预测未来$H$步
3. 求解优化问题：$\min_{u_{t:t+H-1}} \sum_{k=0}^{H-1} \ell(x_{t+k}, u_{t+k}) + V_f(x_{t+H})$
4. 约束：$x \in \mathcal{X}$, $u \in \mathcal{U}$
5. 执行$u_t^*$（最优序列的第一步）
6. $t \leftarrow t+1$，回到步骤1

**关键概念**：

- **预测视野(Horizon)**$H$：向前预测的步数
- **阶段成本**$\ell(x,u)$：每一步的代价
- **终端成本**$V_f(x)$：视野结束时的惩罚
- **约束处理**：MPC能显式处理输入和状态约束

```
时刻t: 当前状态 x_t
         ↓ 预测
    x_t → u_t → x_{t+1} → u_{t+1} → ... → x_{t+H}
         ↓ 求解最优 u_{t:t+H-1}
         ↓ 只执行 u_t*
    时刻t+1: 测量实际 x_{t+1}，重新规划
```

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $x_t$ | 系统状态 |
| $u_t$ | 控制输入 |
| $f(x,u)$ | 系统动力学模型 |
| $\ell(x,u)$ | 阶段成本 |
| $V_f(x)$ | 终端成本 |
| $H$ | 预测视野 |
| $\mathcal{X}, \mathcal{U}$ | 状态和输入约束集 |

### MPC优化问题

在每个时刻$t$求解：

$$\min_{u_{t:t+H-1}} \sum_{k=0}^{H-1} \ell(x_{t+k}, u_{t+k}) + V_f(x_{t+H})$$

subject to:
$$x_{t+k+1} = f(x_{t+k}, u_{t+k}), \quad k=0,...,H-1$$
$$x_{t+k} \in \mathcal{X}, \quad u_{t+k} \in \mathcal{U}$$

### 线性MPC

当系统是线性的（$x_{k+1} = Ax_k + Bu_k$）、成本是二次的时，MPC退化为二次规划（QP）：

$$\min_{U} \sum_{k=0}^{H-1} (x_k^T Q x_k + u_k^T R u_k) + x_H^T Q_f x_H$$

这是一个凸优化问题，可以高效求解。

### 稳定性保证

通过适当选择终端约束$V_f$和终端集$\mathcal{X}_f$，可以保证闭环稳定性。

## 4. 训练过程讲解

### 超参数表

| 参数 | 含义 | 推荐范围 |
|------|------|----------|
| $H$ | 预测视野 | [10, 100] |
| $Q$ | 状态惩罚矩阵 | 根据应用 |
| $R$ | 输入惩罚矩阵 | 根据应用 |
| $dt$ | 离散化步长 | 根据系统 |

## 5. 应用场景

1. **自动驾驶**：轨迹跟踪、避障
2. **化工过程**：温度、压力控制
3. **机器人**：运动规划、平衡控制
4. **能源管理**：储能调度

## 6. 优缺点分析

### 优点
1. **显式约束处理**：直接处理输入和状态约束
2. **多变量控制**：天然处理MIMO系统
3. **前瞻性**：考虑未来行为
4. **灵活性**：支持非线性和时变系统

### 缺点
1. **计算量大**：每步求解优化问题
2. **需要模型**：依赖系统动力学模型
3. **视野有限**：H有限可能错过长期最优

## 7. 调库实现

```python
"""线性MPC：二次规划求解"""
import numpy as np
from scipy.optimize import minimize

def linear_mpc(A, B, Q, R, x0, H=10, umin=None, umax=None):
    """线性MPC控制器"""
    n, m = B.shape

    def cost(U_flat):
        U = U_flat.reshape(H, m)
        x = x0.copy()
        J = 0
        for k in range(H):
            J += x @ Q @ x + U[k] @ R @ U[k]
            x = A @ x + B @ U[k]
        J += x @ Q @ x  # 终端成本
        return J

    bounds = None
    if umin is not None and umax is not None:
        bounds = [(umin[i], umax[i]) for _ in range(H) for i in range(m)]

    U0 = np.zeros(H * m)
    result = minimize(cost, U0, bounds=bounds, method='SLSQP')
    return result.x[:m]  # 返回第一步控制

# 测试：双积分系统（小车位置控制）
if __name__ == "__main__":
    dt = 0.1
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0], [dt]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    x0 = np.array([5.0, 0.0])

    # 仿真MPC控制
    x = x0.copy()
    trajectory = [x.copy()]
    for t in range(50):
        u = linear_mpc(A, B, Q, R, x, H=10, umin=np.array([-2.0]), umax=np.array([2.0]))
        x = A @ x + B.flatten() * u
        trajectory.append(x.copy())

    trajectory = np.array(trajectory)
    print(f"初始位置: {x0.round(2)}")
    print(f"最终位置: {x.round(2)}")
    print(f"位置偏差: {abs(x[0]):.4f}")
```

## 8. 手工代码实现

```python
"""从零实现MPC（无优化库，用梯度下降）"""
import numpy as np

class SimpleMPC:
    def __init__(self, A, B, Q, R, H=10, lr=0.01, n_iters=100):
        self.A, self.B = A, B
        self.Q, self.R = Q, R
        self.H = H
        self.lr = lr
        self.n_iters = n_iters

    def compute_trajectory(self, x0, U):
        """计算给定控制序列下的状态轨迹"""
        n, m = self.B.shape
        X = np.zeros((self.H + 1, n))
        X[0] = x0
        for k in range(self.H):
            X[k+1] = self.A @ X[k] + self.B @ U[k]
        return X

    def cost(self, X, U):
        """总成本"""
        J = 0
        for k in range(self.H):
            J += X[k] @ self.Q @ X[k] + U[k] @ self.R @ U[k]
        J += X[self.H] @ self.Q @ X[self.H]
        return J

    def solve(self, x0):
        """用梯度下降求解MPC"""
        n, m = self.B.shape
        U = np.zeros((self.H, m))

        for _ in range(self.n_iters):
            X = self.compute_trajectory(x0, U)

            # 计算梯度（通过链式法则）
            grad_U = np.zeros_like(U)
            lam = 2 * self.Q @ X[self.H]
            for k in range(self.H - 1, -1, -1):
                grad_U[k] = 2 * self.R @ U[k] + self.B.T @ lam
                lam = 2 * self.Q @ X[k] + self.A.T @ lam

            U -= self.lr * grad_U

        return U[0]  # 返回第一步控制

if __name__ == "__main__":
    np.random.seed(42)
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [0.1]])
    Q = np.diag([10, 1])
    R = np.array([[0.1]])

    mpc = SimpleMPC(A, B, Q, R, H=20, lr=0.001, n_iters=200)

    x = np.array([3.0, 0.0])
    print("MPC控制仿真:")
    for t in range(30):
        u = mpc.solve(x)
        x = A @ x + B.flatten() * u
        if (t+1) % 5 == 0:
            print(f"  t={t+1}: 位置={x[0]:.3f}, 速度={x[1]:.3f}, 控制={u[0]:.3f}")
    print(f"最终位置偏差: {abs(x[0]):.4f}")
```

## 9-14. 简要补充

### 9. 可视化
绘制MPC控制下的状态轨迹和预测轨迹。

### 10. 评估
比较MPC与LQR的控制性能和约束满足情况。

### 11. 常见问题
1. **计算太慢**：缩短视野或用更快的QP求解器
2. **模型不准**：使用鲁棒MPC或自适应MPC
3. **视野选择**：H太短不够前瞻，太长计算量大

### 12. 学习总结
MPC通过"滚动优化"实现约束感知的最优控制：每步求解$H$步前瞻问题，只执行第一步。核心是$\min_{U}\sum \ell(x,u) + V_f(x_H)$ s.t. $x_{k+1}=f(x_k,u_k)$。

### 13. 练习题
**Q1**：MPC与LQR都是最优控制方法，主要区别是什么？
**A1**：LQR求解无限视野无约束问题，得到固定增益矩阵K。MPC求解有限视野带约束问题，每步重新求解。LQR是MPC在无约束、无限视野时的特例。

### 14. 学习路径
**前置**：最优控制、LQR | **进阶**：非线性MPC(NMPC)、鲁棒MPC、随机MPC
**资源**：原书Ch 2.1.13、Borrelli et al. "Predictive Control for Linear and Hybrid Systems"
