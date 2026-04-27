# 粒子滤波 学习文档

> 用一句话说明这个算法的核心价值：粒子滤波是一种基于蒙特卡洛方法的非线性非高斯状态估计技术，通过大量随机采样（粒子）近似表示概率分布，能够处理任意复杂的非线性系统和非高斯噪声。

## 1. 算法基础认知

粒子滤波（Particle Filter），也称为序列蒙特卡洛方法（Sequential Monte Carlo, SMC），是一种使用一组随机样本（称为粒子）来近似表示概率分布的递归贝叶斯估计算法。它特别适用于处理非线性、非高斯的状态空间模型。

**一句话定义**：通过大量随机粒子及其权重来近似表示后验概率分布，实现非线性非高斯系统的状态估计。

**直觉类比**：就像你在一个黑暗房间里寻找掉落的钥匙，虽然不知道确切位置，但你可以让一群朋友（粒子）在房间里随机搜索，每个人报告自己认为钥匙在哪里的可能性（权重），然后根据这些报告集中搜索最可能的位置。

**历史背景**：
- 1993年：Gordon等人提出基础的粒子滤波算法
- 1996年：Liu和Chen引入重要采样重采样技术
- 1998年：Arulampalam给出系统化的粒子滤波综述
- 2000年：PF在SLAM（同步定位与建图）领域广泛应用
- 2005年： Rao-Blackwellized粒子滤波提出，结合了PF和EKF的优点
- 2010s：应用于机器人导航、计算机视觉、金融工程等领域

**算法定位**：
- 属于蒙特卡洛方法和贝叶斯估计
- 非线性非高斯系统的通用解决方案
- 采样方法的典型代表
- 能够处理多模态分布

**典型应用场景**：
- 机器人定位与SLAM（尤其在非高斯噪声下）
- 目标跟踪（特别是机动目标跟踪）
- 金融状态估计（具有跳跃的金融时间序列）
- 生物信息学（基因序列分析）
- 语音识别
- 图像处理和计算机视觉

**前置知识**：
- 概率论与统计（贝叶斯理论、重要性采样）
- 随机过程
- 蒙特卡洛方法
- 状态空间模型基础
- 数值计算基础

## 2. 核心原理

粒子滤波基于**序贯蒙特卡洛方法**，核心思想是：**用一组带权重的随机样本（粒子）来近似表示后验概率分布，通过重要性采样和重采样机制保持有效粒子集**。

**2.1 概率表示**：
在粒子滤波中，概率分布用粒子集合及其权重表示：
$$
p(x) \approx \sum_{i=1}^N w_i \delta(x - x_i)
$$
其中 $x_i$ 是粒子位置，$w_i$ 是对应权重，$\delta$ 是狄拉克δ函数。

**2.2 递归贝叶斯估计**：
粒子滤波实现了递归贝叶斯估计公式：
$$
p(x_k | z_{1:k}) = \eta p(z_k | x_k) \int p(x_k | x_{k-1}) p(x_{k-1} | z_{1:k-1}) dx_{k-1}
$$

**2.3 算法步骤**：
1. **初始化**：在状态空间中随机采样N个粒子
2. **预测**：根据系统动态模型传播粒子
3. **重要性加权**：根据观测数据更新粒子权重
4. **重采样**：根据粒子权重重新采样，避免退化
5. **状态估计**：基于加权粒子计算估计值

## 3. 数学公式与推导

**3.1 重要性采样**：
当直接采样后验分布困难时，使用重要性采样：选择一个容易采样的提议分布 $q(x)$，则：
$$
E_{p(x)}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx \approx \frac{1}{N} \sum_{i=1}^N f(x_i) \frac{p(x_i)}{q(x_i)}
$$
其中 $x_i \sim q(x)$。

在PF中，提议分布通常选择为 $q(x_k) = p(x_k | x_{k-1}^{(i)})$，则权重更新为：
$$
w_k^{(i)} \propto w_{k-1}^{(i)} p(z_k | x_k^{(i)})
$$

**3.2 重采样**：
为避免权重退化问题（大部分权重集中在一个或少数几个粒子上），执行重采样：
1. 计算有效样本大小：$N_{eff} = \frac{1}{\sum_{i=1}^N (w_i)^2}$
2. 当 $N_{eff} < N_{threshold}$ 时，进行重采样
3. 重采样方法：
   - **多项式重采样**：根据权重概率选择粒子
   - **分层重采样**：将权重范围分层，逐层采样
   - **系统重采样**：更高效的单循环重采样方法

**3.3 状态估计**：
估计状态和协方差：
$$
\hat{x}_k = \sum_{i=1}^N w_i^{(k)} x_i^{(k)}
$$
$$
P_k = \sum_{i=1}^N w_i^{(k)} (x_i^{(k)} - \hat{x}_k)(x_i^{(k)} - \hat{x}_k)^T
$$

## 4. 训练过程讲解

粒子滤波的训练过程主要涉及参数调整和模型选择，包括：
1. 粒子数量选择
2. 重采样策略选择
3. 提议分布设计
4. 重要性权重归一化

## 5. 应用场景

- 机器人定位与导航
- 多目标跟踪
- SLAM问题
- 金融状态估计
- 语音识别
- 计算机视觉

## 6. 优缺点分析

**优点**：
- 能够处理非线性非高斯系统
- 并行化潜力大
- 适用于多模态分布
- 理论上可以任意逼近真实分布

**缺点**:
- 计算复杂度高（粒子数多）
- 粒子退化问题
- 对提议分布敏感
- 重采样可能导致多样性丧失

## 7. 调库实现

import numpy as np
from scipy.stats import norm

def particle_filter(observations, num_particles=1000):
    """简单的粒子滤波实现"""
    # 初始化粒子
    particles = np.random.randn(num_particles, 2)
    weights = np.ones(num_particles) / num_particles
    
    estimates = []
    
    for obs in observations:
        # 预测步骤（简单随机游走）
        particles += np.random.randn(num_particles, 2) * 0.1
        
        # 重要性加权
        weights *= norm.pdf(obs[0], loc=particles[:, 0], scale=0.5)
        weights *= norm.pdf(obs[1], loc=particles[:, 1], scale=0.5)
        weights /= weights.sum()  # 归一化
        
        # 重采样
        indices = np.random.choice(num_particles, size=num_particles, p=weights)
        particles = particles[indices]
        weights.fill(1.0 / num_particles)
        
        # 状态估计
        estimate = np.average(particles, weights=weights, axis=0)
        estimates.append(estimate)
    
    return np.array(estimates)

## 8. 手工代码实现

import numpy as np

class SimpleParticleFilter:
    def __init__(self, num_particles=1000):
        self.num_particles = num_particles
        self.particles = None
        self.weights = None
    
    def initialize(self, initial_state, initial_cov):
        """初始化粒子"""
        self.particles = np.random.multivariate_normal(
            initial_state, initial_cov, self.num_particles
        )
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def predict(self, dt):
        """预测步骤：随机游走模型"""
        process_noise = np.random.randn(self.num_particles, 2) * 0.1
        self.particles += process_noise * dt
    
    def update(self, observation):
        """更新步骤：重要性加权"""
        # 计算观测似然（假设高斯观测噪声）
        distances = np.linalg.norm(self.particles - observation, axis=1)
        self.weights = np.exp(-0.5 * distances**2)
        self.weights /= self.weights.sum()  # 归一化
    
    def resample(self):
        """重采样步骤"""
        indices = np.random.choice(
            self.num_particles, self.num_particles, p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)
    
    def estimate(self):
        """状态估计"""
        return np.average(self.particles, weights=self.weights, axis=0)
    
    def run(self, observations):
        """运行粒子滤波器"""
        estimates = []
        for obs in observations:
            self.predict(1.0)
            self.update(obs)
            self.resample()
            estimates.append(self.estimate())
        return np.array(estimates)

## 9. 可视化与结果理解

import matplotlib.pyplot as plt

# 可视化粒子分布和估计结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 粒子分布
axes[0].scatter(particles[:, 0], particles[:, 1], c=weights, cmap='viridis', s=10)
axes[0].set_title('粒子分布与权重')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

# 估计轨迹
axes[1].plot(observations[:, 0], observations[:, 1], 'k.', label='观测', alpha=0.3)
axes[1].plot(estimates[:, 0], estimates[:, 1], 'b-', linewidth=2, label='粒子滤波估计')
axes[1].set_title('轨迹估计')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].legend()
plt.tight_layout()
plt.show()

## 10. 模型评估

评估粒子滤波性能的指标：
- 均方根误差（RMSE）
- 估计误差的平均值
- 权重多样性
- 有效样本大小（N_eff）

## 11. 常见问题与易错点

- **粒子退化**：权重集中在少数粒子上，失去多样性
  - 解决方法：及时重采样
- **提议分布选择不当**：导致重要性权重差异过大
  - 解决方法：使用系统重采样或分层重采样
- **粒子数量不足**：无法准确表示真实分布
  - 解决方法：增加粒子数量，但注意计算成本
- **重采样过早或过晚**：影响估计质量
  - 解决方法：根据有效样本大小动态调整重采样时机

## 12. 学习总结

粒子滤波提供了一种灵活的非参数贝叶斯推断方法，特别适合处理非线性非高斯系统。通过粒子集合和权重机制，可以在复杂的后验分布中进行有效的采样和估计。

## 13. 练习题与思考题

1. 如何选择合适的重要提议分布？
2. 重采样方法的选择对算法性能有何影响？
3. 如何处理高维状态空间的粒子滤波（采样效率问题）？
4. 为什么说有效样本大小（N_eff）是重要的诊断指标？

## 14. 学习路径建议

建议按以下顺序学习：
1. 概率论与统计基础
2. 贝叶斯推断基础
3. 蒙特卡洛方法
4. 重要性采样
5. 粒子滤波基本原理
6. 改进的粒子滤波算法（如Rao-Blackwellized PF）
7. 粒子滤波在SLAM中的应用
8. 与卡尔曼滤波的比较