# ε-greedy 学习文档

> 强化学习中平衡探索与利用的经典动作选择策略。

## 1. 算法基础认知

ε-greedy（Epsilon-Greedy）是强化学习中用于平衡探索（Exploration）与利用（Exploitation）的经典策略。它以概率ε进行随机探索，以概率1-ε选择当前认为最优的动作。

**直觉类比**：想象你在一家餐厅点菜。如果你总是点之前吃过的菜（利用），你不会发现新的美味；如果你总是随机点菜（探索），你可能吃到难吃的菜。ε-greedy的策略是：80%的时候点常吃的菜（利用），20%的时候随机尝试新菜（探索）。

**前置知识**：强化学习基础、Q-Learning

## 2. 核心原理

ε-greedy在探索和利用之间取得平衡：
- 随机探索：防止陷入局部最优
- 利用：最大化即时奖励

## 3. 数学公式与推导

动作选择概率：
$$P(a|s) = \begin{cases} epsilon / |A| + (1 - epsilon) & text{if } a = argmax_{a'} Q(s,a') \ epsilon / |A| & text{otherwise} end{cases}$$

## 4. 训练过程讲解

参数：
- ε：探索率，通常随训练衰减
- ε_min：最小探索率
- decay_rate：衰减率

## 5. 应用场景

- 所有探索-利用权衡问题
-  bandits
- 强化学习

## 6. 优缺点分析

**优点**：简单易实现、有效
**缺点**：ε固定时不高效

## 7. 调库实现

```python
"""
ε-greedy策略实现
"""
import numpy as np

class EpsilonGreedy:
    """ε-greedy动作选择策略"""
    def __init__(self, n_actions, epsilon=0.1, epsilon_min=0.01, decay_rate=0.995):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
    
    def select_action(self, Q_values, training=True):
        """选择动作"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(Q_values)
    
    def decay(self):
        """衰减ε"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

# 测试
np.random.seed(42)
eg = EpsilonGreedy(4, epsilon=0.2)

Q = np.array([0.5, 0.8, 0.3, 0.6])
for i in range(10):
    action = eg.select_action(Q)
    print(f"ε={eg.epsilon:.3f}, 选择动作: {action}, Q值: {Q[action]:.3f}")
    eg.decay()
```

## 8. 手工代码实现

```python
def epsilon_greedy(Q, n_actions, epsilon):
    """ε-greedy动作选择"""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q)

# 示例
Q_values = np.array([1.0, 2.0, 0.5, 1.5])
print("ε-greedy结果:", epsilon_greedy(Q_values, 4, 0.1))
```

## 9-14. 其他章节

代码已包含以上内容。学习总结：ε-greedy是探索-利用平衡的基础策略。

> 来源线索：本节内容根据原书中关于"ε-greedy transition policy"的相关章节整理。

## 9. 可视化与结果理解

### 推荐可视化
1. **训练曲线**：损失随训练轮次变化，观察收敛趋势
2. **性能对比**：epsilon-greedy与基准方法对比
3. **特征重要性**（如适用）：各特征贡献度

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Iteration'); plt.ylabel('Loss')
plt.title('epsilon-greedy Training Loss')
plt.show()
```

### 结果解读
- 训练损失持续下降表明模型在学习
- 验证损失上升可能过拟合
- 差距过大需考虑正则化


## 10. 模型评估

### 评估指标
- **准确率(Accuracy)**：正确预测比例
- **精确率/召回率/F1**：综合评估分类质量
- **AUC-ROC**：分类器整体性能
- **损失值**：训练收敛关键指标

### 评估方法
1. **K折交叉验证**：稳健的性能估计
2. **留出法**：独立训练/验证/测试集
3. **时间序列验证**：滚动窗口（金融场景）

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```


## 11. 常见问题与易错点

### 常见问题
**Q1: 训练不收敛？**
- 检查学习率是否合适
- 检查数据是否正确归一化
- 确认损失函数是否匹配任务

**Q2: 过拟合严重？**
- 增加数据量或使用数据增强
- 添加正则化（L1/L2/Dropout）
- 使用早停策略

**Q3: 超参数如何选？**
- 网格搜索或随机搜索
- 贝叶斯优化
- 参考论文推荐值

### 易错点
1. 数据泄露：预处理时使用测试集信息
2. 随机种子：忘记设置导致不可复现
3. 维度错误：输入shape与模型不匹配
4. 梯度问题：需要适当初始化和裁剪
5. 评估偏差：在训练集上评估


## 12. 学习总结

### 核心要点
1. **基本原理**：epsilon-greedy的核心思想和数学基础
2. **实现方法**：从调库到手工实现
3. **应用场景**：epsilon-greedy适合的问题类型
4. **优化技巧**：超参数调优和正则化
5. **评估方法**：客观评估性能

### 关键概念
- epsilon-greedy的损失函数设计原理
- 参数优化的数学推导
- 泛化能力与过拟合的平衡

### 进阶方向
掌握epsilon-greedy后，可进一步学习相关的进阶方法和变体。


## 13. 练习题与思考题（含答案）

### 练习1：概念理解
题目：简述epsilon-greedy的核心思想及适用场景。
<details><summary>参考答案</summary>
epsilon-greedy通过数据驱动学习输入到输出的映射，适用于强化学习中的模式识别、预测和决策等任务。
</details>

### 练习2：公式推导
题目：写出epsilon-greedy的损失函数并推导梯度。
<details><summary>参考答案</summary>
$$L(\theta) = \frac{1}{N} \sum_{i} \ell(y_i, f(x_i; \theta))$$
$$\nabla_\theta L = \frac{1}{N} \sum_{i} \nabla_\theta \ell(y_i, f(x_i; \theta))$$
</details>

### 练习3：代码实现
题目：用Python实现epsilon-greedy核心逻辑并测试。
<details><summary>参考答案</summary>
参考第8章手工代码实现部分。
</details>

### 思考题
1. epsilon-greedy在什么情况下会失效？
2. 训练数据很少时，epsilon-greedy还能有效工作吗？
3. 如何将epsilon-greedy与其他方法结合？


## 14. 学习路径建议

### 前置知识
概率论、MDP、Python、NumPy

### 学习顺序
1. 先理解原理：掌握epsilon-greedy核心思想和数学基础
2. 动手实践：运行代码，观察实验结果
3. 深入理解：阅读原始论文，理解设计动机
4. 项目实战：真实数据集上应用epsilon-greedy

### 进阶方向
多智能体RL、RLHF

### 推荐资源
- 搜索epsilon-greedy原始论文和综述
- GitHub优秀实现
- Coursera/Stanford相关课程

