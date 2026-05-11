# 采样模型/样本平均近似(Sampled Models / SAA) 学习文档

> 将随机优化问题转化为确定性采样问题，是随机规划和大规模随机优化的核心求解方法论。

> 来源线索：本节内容根据原书中关于"Sampled Models"、"Formulating a Sampled Model"、"Static Sampling: Solving a Sampled Model"的相关章节(Ch 10, Stochastic Modeling章节)整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义**：SAA（Sample Average Approximation）从分布中采样$N$个场景，将随机问题的期望目标替换为样本平均，得到可求解的确定性近似问题。

**算法定位**：不确定性建模 / 求解方法论。是蒙特卡洛仿真的"优化版"——不是仿真评估策略，而是用采样构建可求解的确定性模型。

**前置知识**：蒙特卡洛仿真、随机规划、大数定律。

## 2. 核心原理

**核心思想**：原问题含期望$\mathbb{E}[f(x, W)]$，直接优化困难（需要积分）。SAA用样本平均近似：

$$\mathbb{E}[f(x, W)] \approx \frac{1}{N}\sum_{n=1}^N f(x, W^{(n)})$$

其中$W^{(1)}, \ldots, W^{(N)}$是从分布中采样的场景。近似问题是确定性的，可用现成的优化求解器。

**与蒙特卡洛仿真的区别**：
- MC仿真：给定策略$x$，用采样估计$V(x) = \mathbb{E}[f(x,W)]$
- SAA：用采样构建近似问题$\min_x \frac{1}{N}\sum f(x,W^{(n)})$，然后求解最优$x$

**与随机规划的关系**：SAA是求解随机规划的计算方法论——两阶段随机规划就是SAA的一个特例（采样场景后求解大规模确定性LP）。

## 3. 数学公式

### 原始随机优化问题

$$\min_{x \in X} \quad \mathbb{E}[f(x, W)]$$

### SAA近似问题

$$\min_{x \in X} \quad \hat{f}_N(x) = \frac{1}{N}\sum_{n=1}^N f(x, W^{(n)})$$

### 收敛性（大数定律保证）

$$\hat{f}_N(x) \xrightarrow{a.s.} \mathbb{E}[f(x, W)], \quad N \to \infty$$

设$x^*$是原问题最优解，$\hat{x}_N$是SAA最优解：

$$\mathbb{E}[f(\hat{x}_N, W)] \to f(x^*, W), \quad N \to \infty$$

### SAA的方差

最优值的方差估计：

$$\text{Var}[\hat{v}_N] = \frac{1}{N}\text{Var}[f(\hat{x}_N, W)]$$

可用批量SAA（求解$M$个独立的SAA问题）估计。

### 置信区间

最优值的$(1-\alpha)$置信区间：

$$\hat{v}_N \pm z_{\alpha/2} \cdot \frac{S_v}{\sqrt{M}}$$

其中$M$是SAA重复次数，$S_v$是$M$个最优值的样本标准差。

## 4. 求解过程

1. **采样**：从分布中采样$N$个场景$W^{(1)}, \ldots, W^{(N)}$
2. **构建SAA问题**：将期望替换为样本平均
3. **求解SAA问题**：得到$\hat{x}_N$（确定性优化问题）
4. **评估**：用大量独立场景评估$\hat{x}_N$的真实目标值
5. **可选：重复**：做$M$次SAA，取最好的$\hat{x}_N$

## 5. 应用场景

- 两阶段随机规划（原书Ch 19.9）
- 机会约束规划的采样近似
- 随机线性/非线性规划
- 电力系统调度（场景生成）
- 供应链网络设计
- 金融风险管理（CVaR优化）

## 6. 优缺点

**优点**：
- 将随机问题转化为确定性问题，可用现成求解器
- 理论保证强（大数定律→一致性）
- 实现简单
- 可并行（每个SAA独立求解）

**缺点**：
- $N$大时SAA问题规模大（$N$个场景×问题规模）
- 只保证渐近最优（有限$N$有偏差）
- 需要能从分布中采样
- 非凸问题时SAA也可能是非凸的

## 7. 库实现

Python中 `cvxpy` + `scipy` 可实现。大规模问题可用 `Pyomo` 或 `Gurobi` 的随机规划接口。

## 8. 从零实现

```python
"""采样模型 / 样本平均近似(SAA)"""
import numpy as np
from scipy.optimize import minimize

class SAA:
    def __init__(self, objective_fn, n_scenarios, bounds=None):
        """
        objective_fn(x, scenario) -> 成本值
        n_scenarios: 采样场景数
        """
        self.obj_fn = objective_fn
        self.N = n_scenarios
        self.bounds = bounds
        self.scenarios = None
        self.solutions = []
        self.values = []

    def sample_scenarios(self, sample_fn):
        """从分布中采样N个场景"""
        self.scenarios = [sample_fn() for _ in range(self.N)]

    def saa_objective(self, x):
        """SAA目标函数：样本平均"""
        total = sum(self.obj_fn(x, w) for w in self.scenarios)
        return total / self.N

    def solve(self, x0, n_reps=1, sample_fn=None):
        """求解SAA（可选重复M次取最优）"""
        self.solutions = []
        self.values = []

        for rep in range(n_reps):
            # 采样场景
            if sample_fn is not None:
                self.sample_scenarios(sample_fn)

            # 求解确定性优化
            result = minimize(self.saa_objective, x0,
                            method='L-BFGS-B', bounds=self.bounds)
            self.solutions.append(result.x)
            self.values.append(result.fun)

        # 取最优
        best_idx = np.argmin(self.values)
        return self.solutions[best_idx], self.values[best_idx]

    def evaluate_true(self, x, n_eval=10000, sample_fn=None):
        """用大量独立场景评估解的真实目标值"""
        eval_scenarios = [sample_fn() for _ in range(n_eval)]
        total = sum(self.obj_fn(x, w) for w in eval_scenarios)
        return total / n_eval

if __name__ == "__main__":
    np.random.seed(42)

    # 示例：报童问题的SAA求解
    # min E[(c_h * max(x - D, 0) + c_p * max(D - x, 0))]
    c_h, c_p = 1.0, 3.0  # 持有成本、缺货成本

    def newsvendor_cost(x, demand):
        """x=[订货量], demand=需求场景"""
        return c_h * max(x[0] - demand, 0) + c_p * max(demand - x[0], 0)

    def sample_demand():
        """需求分布：N(100, 20)"""
        return max(0, np.random.normal(100, 20))

    saa = SAA(newsvendor_cost, n_scenarios=500,
              bounds=[(50, 150)])

    best_x, best_v = saa.solve(
        x0=np.array([100.0]),
        n_reps=10,
        sample_fn=sample_demand
    )

    true_v = saa.evaluate_true(best_x, n_eval=50000, sample_fn=sample_demand)

    print(f"SAA求解（N=500, 10次重复）:")
    print(f"  最优订货量: {best_x[0]:.2f}")
    print(f"  SAA目标值: {best_v:.2f}")
    print(f"  真实期望成本（评估）: {true_v:.2f}")

    # 解析最优
    F_optimal = c_p / (c_h + c_p)
    from scipy.stats import norm
    x_optimal = norm.ppf(F_optimal, 100, 20)
    print(f"\n  解析最优订货量: {x_optimal:.2f}")
    print(f"  SAA与解析的偏差: {abs(best_x[0] - x_optimal):.2f}")
```

## 9. 可视化

```python
import matplotlib.pyplot as plt
# 绘制SAA目标值随N的变化（应收敛到真实最优值）
# x轴：场景数N，y轴：目标函数值
# 上界/下界用阴影区域表示置信区间
```

## 10. 评估方法

- SAA解与真实最优的偏差（如已知解析解）
- 目标值的置信区间（多次SAA重复的方差）
- 解的稳定性（不同随机种子下解的变异程度）

## 11. 常见错误

- 场景数$N$太小，SAA解偏差大
- 场景数$N$太大，求解SAA问题计算成本过高
- 采样分布与真实分布不一致（模型误差）
- 评估用和训练用同一组场景（过拟合）
- 非凸问题中SAA可能找到局部最优

## 12. 学习总结

SAA：$\min_x \frac{1}{N}\sum_{n=1}^N f(x, W^{(n)})$。用采样将随机问题转化为确定性问题。$N\to\infty$时SAA解收敛到真实最优。实践核心是平衡场景数$N$与计算成本。

## 13. 练习题

**Q1**：SAA和蒙特卡洛仿真有何区别？
**A1**：MC仿真是给定策略$x$后用采样评估$V(x)$——不改变$x$。SAA是用采样构建近似问题后**求解最优$x$**——改变决策。MC是评估工具，SAA是求解工具。

**Q2**：为什么SAA需要重复多次（$M$次）？
**A2**：单次SAA的最优值$\hat{v}_N$是随机变量（依赖采样场景）。重复$M$次可以估计最优值的方差，构建置信区间，并取$M$次中最好的解（降低碰巧采到差场景的风险）。

## 14. 学习路径

**前置**：蒙特卡洛仿真、随机规划 | **进阶**：多切割SAA、内部采样策略
**资源**：原书Ch 10 (Stochastic Modeling)、Shapiro et al. (2009) Chap 5
