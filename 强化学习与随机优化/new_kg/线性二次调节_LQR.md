# 线性二次调节(LQR) 学习文档

> LQR是线性系统最优控制的解析解，是控制理论的基石和MPC的理论特例。

> 来源线索：本节内容根据原书中关于"Linear Quadratic Regulation"的相关章节(Ch 14.11)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：LQR在系统动力学为线性、成本函数为二次的设定下，通过求解代数Riccati方程得到最优反馈控制律$u = -Kx$。

**直觉类比**：你要平衡一个倒立摆。摆的角度和角速度构成状态，推力是控制。LQR告诉你：最优策略是"用与状态成比例的力把摆推回去"，比例系数$K$通过解一个矩阵方程（Riccati方程）得到。

**历史背景**：LQR的理论由Kalman (1960)和Bellman (1957)奠定。离散时间LQR的解由代数Riccati方程(DARE)给出。LQR是现代控制理论的核心成果之一。

**算法定位**：最优控制/精确解析解。在原书Ch 14.11中，LQR是连续状态空间中少数有精确解的问题之一。

**前置知识**：线性系统、矩阵运算、Bellman方程、优化。

## 2. 核心原理

**核心思想**：对于线性系统$x_{k+1} = Ax_k + Bu_k$和二次成本$J = \sum_k (x_k^T Q x_k + u_k^T R u_k)$，最优控制是状态的线性反馈$u_k = -K x_k$。增益$K$通过求解Riccati方程获得。

**工作流程**：

1. 给定系统矩阵$A, B$和权重矩阵$Q, R$
2. 求解代数Riccati方程：$P = A^TPA - A^TPB(R+B^TPB)^{-1}B^TPA + Q$
3. 计算最优增益：$K = (R + B^TPB)^{-1}B^TPA$
4. 最优控制：$u_k = -Kx_k$

## 3. 数学公式与推导

### 问题设定

系统：$x_{k+1} = Ax_k + Bu_k$

成本：$J = \sum_{k=0}^{\infty} (x_k^T Q x_k + u_k^T R u_k)$

其中$Q \geq 0$（半正定），$R > 0$（正定）。

### Bellman方程

$$V(x) = \min_u [x^T Q x + u^T R u + V(Ax + Bu)]$$

### 值函数形式

猜测$V(x) = x^T P x$（二次函数），代入Bellman方程：

$$x^TPx = \min_u [x^TQx + u^TRu + (Ax+Bu)^TP(Ax+Bu)]$$

对$u$求导令其为0：

$$2Ru + 2B^TP(Ax+Bu) = 0$$
$$u = -(R + B^TPB)^{-1}B^TPAx = -Kx$$

### Riccati方程

将$u = -Kx$代回，得离散代数Riccati方程(DARE)：

$$P = Q + A^TPA - A^TPB(R + B^TPB)^{-1}B^TPA$$

### 最优增益

$$K = (R + B^TPB)^{-1}B^TPA$$

### 迭代求解

从$P_0 = Q$开始，迭代$P_{k+1} = Q + A^TP_kA - A^TP_kB(R+B^TP_kB)^{-1}B^TP_kA$，直到收敛。

## 4. 训练过程讲解

### 超参数表

| 参数 | 含义 | 选择建议 |
|------|------|---------|
| $Q$ | 状态惩罚 | 对重要状态加大权重 |
| $R$ | 控制惩罚 | 控制代价大则加大$R$ |

## 5-6. 应用场景与优缺点

**应用**：倒立摆控制、飞行器姿态控制、电机控制、经济调控。

**优点**：解析最优解、计算高效（一次求解）、稳定性保证。
**缺点**：仅适用于线性系统、无法处理约束、需要精确模型。

## 7. 调库实现

```python
"""LQR控制：使用scipy求解Riccati方程"""
import numpy as np
from scipy.linalg import solve_discrete_are

def lqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K, P

# 测试：双积分系统
A = np.array([[1, 0.1], [0, 1]])
B = np.array([[0], [0.1]])
Q = np.diag([10, 1])
R = np.array([[1.0]])

K, P = lqr(A, B, Q, R)
print(f"最优增益 K = {K.round(4)}")
print(f"Riccati解 P =\n{P.round(4)}")

# 仿真
x = np.array([5.0, 0.0])
for t in range(30):
    u = -K @ x
    x = A @ x + B.flatten() * u
print(f"30步后状态: {x.round(4)}")
```

## 8. 手工代码实现

```python
"""从零实现LQR（迭代Riccati求解）"""
import numpy as np

def solve_lqr(A, B, Q, R, max_iter=1000, epsilon=1e-8):
    """迭代求解离散Riccati方程"""
    P = Q.copy()
    for i in range(max_iter):
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        if np.max(np.abs(P_new - P)) < epsilon:
            print(f"Riccati方程在第{i+1}轮收敛")
            break
        P = P_new
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K, P

if __name__ == "__main__":
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [0.1]])
    Q = np.diag([10, 1])
    R = np.array([[1.0]])

    K, P = solve_lqr(A, B, Q, R)
    print(f"K = {K.round(4)}")

    x = np.array([3.0, 1.0])
    traj = [x.copy()]
    for _ in range(20):
        u = (-K @ x)[0]
        x = A @ x + B.flatten() * u
        traj.append(x.copy())
    traj = np.array(traj)
    print(f"初始: {traj[0].round(2)}, 最终: {traj[-1].round(4)}")
```

## 9-14. 简要补充

### 9. 可视化
绘制状态轨迹和相平面图。

### 10. 评估
比较LQR与随机控制的成本。

### 11. 常见问题
1. **$R$太小**：控制量过大 → 增大$R$
2. **系统不可控**：Riccati方程不收敛 → 检查可控性
3. **连续vs离散**：连续LQR用微分Riccati方程

### 12. 学习总结
LQR的解析最优控制为$u=-Kx$，$K=(R+B^TPB)^{-1}B^TPA$，$P$由Riccati方程求解。是值迭代的连续状态特例。

### 13. 练习题
**Q1**：双积分系统$A=[[1,1],[0,1]], B=[[0],[1]], Q=I, R=1$。求K的第一步迭代（从$P_0=Q$）。
**A1**：$P_1 = Q + A^TQA - A^TQB(R+B^TQB)^{-1}B^TQA$。$B^TQB = 1$, $(R+1)^{-1}=0.5$, $A^TQB = [[0],[1]]$, 代入计算。

### 14. 学习路径
**前置**：值迭代、线性系统 | **进阶**：MPC（LQR+约束）、LQG（+噪声）、H-infinity控制
**资源**：原书Ch 14.11、Bertsekas "Dynamic Programming and Optimal Control"
