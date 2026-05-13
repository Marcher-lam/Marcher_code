# ACP自动与受控加工理论 学习文档
> 来源线索：本节内容根据原书第1章关于"自动与受控加工理论"的相关章节整理、扩展与教学化改写。

## 1. 算法基础认知

**一句话定义：** ACP（Automatic and Controlled Processes）理论由Schneider和Shiffrin于1977年提出，认为注意力分配存在自动化加工（Automatic Processing）和受控加工（Controlled Processing）两种机制——前者高效、无意识、不消耗注意资源，后者低效、有意识、消耗注意资源。

**直觉类比：** 开车时，走熟悉路段可以听音乐聊天（自动化加工），走到陌生路段必须专注（受控加工）。自动化加工是通过反复练习达到的"肌肉记忆"状态——就像运动员的高难度动作，一开始需要高度专注，练到极致后"不假思索"。

**历史背景：** 1977年，Walter Schneider和Richard Shiffrin在Psychological Review上发表了一篇里程碑式的论文《Controlled and Automatic Human Information Processing》，提出了ACP理论。该理论在卡内曼资源理论基础上，进一步阐述了注意力的分配机制，解释了"熟能生巧"的认知本质。

**算法定位：** 注意力资源理论的延伸，解释了技能学习和自动化的认知机制。

---

## 2. 核心原理

### 2.1 两种加工模式对比

| 维度 | 自动化加工 | 受控加工 |
|------|-----------|---------|
| 执行速度 | 快 | 慢 |
| 意识参与 | 无意识 | 有意识 |
| 注意资源消耗 | 少/无 | 多 |
| 灵活性 | 低（一旦启动难以停止） | 高（可灵活调整） |
| 并行能力 | 可多任务并行 | 串行，一次只能处理一个 |
| 练习依赖性 | 需要大量练习 | 可不经练习 |
| 抗干扰性 | 强 | 弱 |
| 可中断性 | 难中断 | 可随时中断 |

### 2.2 加工模式的获取过程

```
新手阶段（全受控）:
任务 -> 高度注意 -> 慢速加工 -> 易出错

练习阶段（受控->自动）:
任务 -> 注意逐渐减少 -> 速度逐渐提升 -> 错误逐渐减少

专家阶段（全自动）:
任务 -> 无需注意 -> 快速加工 -> 准确
```

### 2.3 经典实验：视觉搜索实验

Schneider和Shiffrin的实验范式：
- **一致映射（Consistent Mapping, CM）：** 目标和干扰物来自不同类别（如目标永远是数字，干扰物永远是字母）
- **变异映射（Varied Mapping, VM）：** 目标和干扰物来自相同类别（如目标和干扰物都是字母，角色随机交换）

实验结果：
- CM条件下：经过大量练习后，搜索速度与集合大小无关（自动化加工）
- VM条件下：搜索速度随集合大小增加而线性增加（受控加工）

---

## 3. 数学公式与推导

### 3.1 自动化程度量化

定义自动化程度 $A \in [0, 1]$，其中 $0$ 表示完全受控，$1$ 表示完全自动：

$$A = 1 - \frac{R_{current}}{R_{initial}}$$

其中 $R_{current}$ 是当前任务的注意资源消耗，$R_{initial}$ 是初始资源消耗。

### 3.2 练习曲线（Power Law of Practice）

练习次数 $n$ 与任务表现 $P(n)$ 的关系：

$$P(n) = P_0 \cdot n^{-\beta} + P_\infty$$

其中：
- $P_0$：初始表现水平
- $P_\infty$：自动化后的最终表现水平
- $\beta$：学习速率（$\beta \in [0.1, 0.6]$）

反应时间与练习次数：

$$RT(n) = RT_\infty + \frac{RT_0 - RT_\infty}{(n + 1)^\beta}$$

### 3.3 资源消耗函数

任务 $i$ 的注意资源消耗随练习水平的变化：

$$R_i(p_i) = R_i^0 \cdot e^{-\alpha_i p_i}$$

其中 $p_i$ 是练习量，$\alpha_i$ 是自动化的速率参数。

### 3.4 双任务干扰中的自动化效应

当任务1达到部分自动化时，双任务表现：

$$P_1 = f_1(R_1) = f_1(R_{total} - R_2 \cdot (1 - A_1))$$

其中 $A_1$ 是任务1的自动化程度，$R_1$ 是任务1实际消耗的资源。

### 3.5 自动化加工的特征方程

自动化加工的速度-准确率平衡：

$$\text{Speed} = \frac{1}{\tau_{auto} + \epsilon}$$

$$\text{Accuracy} = 1 - \exp(-\gamma \cdot \text{practice})$$

其中 $\tau_{auto}$ 是自动化加工的最小处理时间。

### 3.6 加工模式切换条件

从受控加工切换到自动化加工的条件：

$$\frac{dP}{dn} < \theta \quad \text{且} \quad P > P_{threshold}$$

即表现增长趋于平稳且已达到足够高水平的准确率。

---

## 4. 训练过程讲解（技能自动化的认知过程）

### 4.1 一致映射（CM）训练流程

```
1. 初始状态：受控加工
   - 需要记住目标类别（如"数字是目标"）
   - 速度慢，但准确率高

2. 练习阶段：自动化建立
   - 每次试验都遵循相同规则
   - 目标检测逐渐变得"自动"
   - 注意资源需求逐渐下降

3. 最终状态：自动化加工
   - 看到数字自动弹出（pop-out）
   - 不消耗注意资源
   - 即使分心也能完成任务
```

### 4.2 变异映射（VM）训练的困境

```
1. 每次试验规则变化
   - 这次目标是"B"，下次数可能变成干扰物
   - 无法形成稳定的"目标特征"记忆

2. 始终需要受控加工
   - 必须记住当前目标是什么
   - 每次都需要"主动搜索"

3. 练习无法带来自动化
   - 反应时间不会随练习显著下降
   - 始终消耗注意资源
```

### 4.3 自动化的计算模拟

```python
def practice_curve(n_trials, initial_RT=1000, final_RT=200, beta=0.4):
    """模拟练习曲线"""
    RT = [final_RT + (initial_RT - final_RT) / ((n + 1) ** beta)
          for n in range(n_trials)]
    return RT
```

---

## 5. 应用场景

### 5.1 认知心理学
- **技能学习研究：** 从新手到专家的认知机制
- **注意训练：** 通过一致性训练提高注意效率
- **阅读研究：** 阅读从受控（逐字辨认）到自动（整体理解）的过程

### 5.2 人机交互
- **用户界面设计：** 设计一致的交互模式以促进自动化
- **快捷键设计：** 一致性映射帮助形成肌肉记忆
- **游戏设计：** 通过一致性操作降低认知负荷

### 5.3 交通运输
- **驾驶员培训：** 基本操作自动化的训练设计
- **飞行员训练：** 在模拟器中建立自动化处理能力
- **驾驶分心研究：** 自动化程度高的驾驶员分心风险更低

### 5.4 人工智能
- **技能迁移学习：** 预训练（自动化）+微调（受控）的类比
- **元学习：** 学习如何自动化学习过程
- **强化学习中的策略蒸馏：** 将受控策略转化为自动化的快策略

---

## 6. 优缺点分析

### 6.1 优点
1. **解释"熟能生巧"的认知机制：** 清晰说明了自动化加工如何通过练习建立
2. **实验证据扎实：** CM vs VM范式提供了清晰的实验证据
3. **与资源理论互补：** 解释了"为什么有些任务不消耗注意资源"
4. **实践指导性强：** 直接指导技能训练设计

### 6.2 缺点
1. **二分法过于简单：** 自动化不是"全或无"的，而是连续渐变的过程
2. **自动化条件不完整：** 一致映射是充分条件但不是必要条件
3. **未解释"去自动化"：** 当环境变化时自动化加工如何被抑制
4. **个体差异大：** 不同个体达到自动化所需练习量差异巨大

---

## 7. 调库实现（Python + NumPy模拟）

```python
"""
ACP自动与受控加工理论的计算模拟
模拟练习曲线、CM vs VM实验和自动化的双任务效应
"""
import numpy as np
import matplotlib.pyplot as plt


class ACPModel:
    """ACP自动与受控加工模型"""

    def __init__(self, initial_RT=1000, final_RT=200, learning_rate=0.4):
        self.initial_RT = initial_RT          # 初始反应时间(ms)
        self.final_RT = final_RT              # 自动化后反应时间(ms)
        self.learning_rate = learning_rate    # 学习速率beta
        self.automation_level = 0.0           # 自动化程度(0-1)
        self.practice_count = 0               # 练习次数

    def practice(self, n_trials=1, consistent=True):
        """
        进行练习
        consistent: 是否为一致映射条件
        """
        RTs = []
        for _ in range(n_trials):
            if consistent:
                # CM条件：可以建立自动化
                self.practice_count += 1
                n = self.practice_count
                RT = self.final_RT + (self.initial_RT - self.final_RT) / ((n + 1) ** self.learning_rate)
                # 自动化程度提高
                self.automation_level = 1 - (RT - self.final_RT) / (self.initial_RT - self.final_RT)
            else:
                # VM条件：无法自动化
                RT = self.initial_RT * (0.8 + 0.4 * np.random.random())
                self.automation_level = 0.1  # 始终保持低自动化

            RTs.append(RT)

        return np.array(RTs)

    def dual_task_performance(self, task2_difficulty=0.5):
        """
        双任务表现：在自动化加工条件下执行第二任务
        自动化程度越高，双任务干扰越小
        """
        # 自动化减少第一任务的资源消耗
        resource_saved = self.automation_level
        # 可用资源增加
        available_for_task2 = 1.0 - (1.0 - resource_saved) * 0.6
        # 第二任务表现
        task2_performance = available_for_task2 / (available_for_task2 + task2_difficulty)
        return task2_performance

    def search_experiment(self, set_size, trial_type='CM'):
        """
        模拟视觉搜索实验
        set_size: 显示项数量
        trial_type: 'CM' 或 'VM'
        """
        if trial_type == 'CM':
            if self.practice_count > 200:
                # 充分练习后：自动弹出，与集合大小无关
                RT = self.final_RT + 20 * np.random.random()
            else:
                # 未充分练习
                RT = self.final_RT + (self.initial_RT - self.final_RT) / (set_size ** 0.5)
        else:  # VM
            # 受控搜索：RT与集合大小线性相关
            RT = self.initial_RT * 0.3 + 20 * set_size + 30 * np.random.random()

        accuracy = 0.95 if trial_type == 'CM' else 0.85
        return RT, accuracy


def run_acp_demo():
    """运行ACP理论演示"""
    np.random.seed(42)

    model = ACPModel(initial_RT=1000, final_RT=250, learning_rate=0.35)

    print("=" * 50)
    print("ACP自动与受控加工理论 - 完整模拟")
    print("=" * 50)

    # CM条件练习
    print("\n[实验1] 一致映射(CM)条件下的练习曲线")
    cm_RTs = model.practice(n_trials=100, consistent=True)
    print(f"  初始: {cm_RTs[0]:.0f}ms, 第50次: {cm_RTs[49]:.0f}ms, "
          f"第100次: {cm_RTs[-1]:.0f}ms")
    print(f"  自动化程度: {model.automation_level:.2%}")

    # VM条件练习
    print("\n[实验2] 变异映射(VM)条件下的练习曲线")
    model2 = ACPModel(initial_RT=1000, final_RT=250, learning_rate=0.35)
    vm_RTs = model2.practice(n_trials=100, consistent=False)
    print(f"  初始: {vm_RTs[0]:.0f}ms, 第50次: {vm_RTs[49]:.0f}ms, "
          f"第100次: {vm_RTs[-1]:.0f}ms")
    print(f"  自动化程度: {model2.automation_level:.2%}")

    # 视觉搜索实验
    print("\n[实验3] 视觉搜索：集合大小效应")
    for size in [1, 2, 4, 8]:
        rt_cm, acc_cm = model.search_experiment(size, 'CM')
        rt_vm, acc_vm = model.search_experiment(size, 'VM')
        print(f"  集合大小={size}: CM_RT={rt_cm:.0f}ms, VM_RT={rt_vm:.0f}ms")

    # 双任务测试
    print("\n[实验4] 双任务表现与自动化")
    for practice_level in [0, 50, 200, 500]:
        m = ACPModel(initial_RT=1000, final_RT=250, learning_rate=0.35)
        m.practice(n_trials=practice_level, consistent=True)
        perf = m.dual_task_performance(0.4)
        print(f"  练习{practice_level}次: 自动化={m.automation_level:.0%}, "
              f"双任务表现={perf:.2%}")

    return model


if __name__ == "__main__":
    model = run_acp_demo()
```

---

## 8. 手工代码实现（核心算法手写）

```python
"""
ACP理论的手工实现
"""
import math
import random


class SimpleACPModel:
    """手工实现的ACP模型"""

    def __init__(self):
        self.practice_count = 0
        self.auto_level = 0.0

    def power_law_practice(self, n, RT0=1000, RTinf=200, beta=0.4):
        """幂定律练习曲线"""
        return RTinf + (RT0 - RTinf) / ((n + 1) ** beta)

    def cm_practice(self, n_trials=50):
        """CM条件练习"""
        RTs = []
        for i in range(n_trials):
            self.practice_count += 1
            RT = self.power_law_practice(self.practice_count)
            RTs.append(RT)
        # 计算自动化水平
        initial = self.power_law_practice(0)
        current = self.power_law_practice(self.practice_count)
        self.auto_level = 1.0 - (current - 200) / (initial - 200)
        return RTs

    def vm_practice(self, n_trials=50):
        """VM条件练习"""
        RTs = []
        for _ in range(n_trials):
            RT = 500 + random.randint(0, 400)
            RTs.append(RT)
        self.auto_level = 0.05
        return RTs

    def simulate_search(self, set_size, cm=True):
        """模拟视觉搜索"""
        if cm and self.auto_level > 0.7:
            # 自动弹出
            RT = 200 + random.randint(0, 50)
            acc = 0.95 + random.random() * 0.05
        elif cm:
            # CM但未充分练习
            RT = 200 + 50 * math.sqrt(set_size) + random.randint(0, 100)
            acc = 0.90
        else:
            # VM：串行搜索
            RT = 300 + 80 * set_size + random.randint(0, 100)
            acc = 0.80 + random.random() * 0.1
        return RT, acc

    def run_demo(self):
        """运行完整演示"""
        print("=" * 40)
        print("手工实现 - ACP理论模拟")
        print("=" * 40)

        print("\n1. CM条件练习曲线:")
        cm_RTs = self.cm_practice(30)
        print(f"   第1次: {cm_RTs[0]:.0f}ms, 第30次: {cm_RTs[-1]:.0f}ms")
        print(f"   自动化水平: {self.auto_level:.1%}")

        print("\n2. CM条件搜索（已自动化）:")
        for size in [1, 2, 4]:
            rt, acc = self.simulate_search(size, cm=True)
            print(f"   集合大小{size}: RT={rt:.0f}ms, Acc={acc:.1%}")

        print("\n3. VM条件搜索（受控）:")
        for size in [1, 2, 4]:
            rt, acc = self.simulate_search(size, cm=False)
            print(f"   集合大小{size}: RT={rt:.0f}ms, Acc={acc:.1%}")

        return cm_RTs


if __name__ == "__main__":
    random.seed(42)
    model = SimpleACPModel()
    model.run_demo()
```

---

## 9. 可视化与结果理解

```python
"""
ACP理论的可视化
"""
import numpy as np
import matplotlib.pyplot as plt


def visualize_acp_theory():
    fig = plt.figure(figsize=(16, 12))

    # 子图1: 练习曲线（CM vs VM）
    ax1 = fig.add_subplot(2, 3, 1)
    trials = np.arange(1, 101)
    cm_RT = 200 + 800 / ((trials + 1) ** 0.4)
    vm_RT = 500 + 200 * np.random.random(100)
    ax1.plot(trials, cm_RT, 'b-', label='CM (自动化)', linewidth=2)
    ax1.plot(trials, vm_RT, 'r-', alpha=0.5, label='VM (受控)', linewidth=1)
    ax1.set_xlabel('练习次数')
    ax1.set_ylabel('反应时间 (ms)')
    ax1.set_title('CM vs VM 练习曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 集合大小效应（CM vs VM）
    ax2 = fig.add_subplot(2, 3, 2)
    set_sizes = [1, 2, 4, 8, 16]
    cm_RTs = [250, 255, 260, 258, 262]
    vm_RTs = [380, 460, 540, 680, 820]
    ax2.plot(set_sizes, cm_RTs, 'bo-', label='CM (自动化)', linewidth=2, markersize=8)
    ax2.plot(set_sizes, vm_RTs, 'rs-', label='VM (受控)', linewidth=2, markersize=8)
    ax2.set_xlabel('集合大小')
    ax2.set_ylabel('反应时间 (ms)')
    ax2.set_title('集合大小效应')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 自动化程度随练习的变化
    ax3 = fig.add_subplot(2, 3, 3)
    trials = np.linspace(0, 500, 100)
    auto_level = 1 - np.exp(-0.008 * trials)
    ax3.plot(trials, auto_level, 'g-', linewidth=2)
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='半自动化')
    ax3.axhline(0.8, color='gray', linestyle=':', alpha=0.5, label='高度自动化')
    ax3.set_xlabel('练习次数')
    ax3.set_ylabel('自动化程度')
    ax3.set_title('自动化程度随练习的增长')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图4: 自动化对资源消耗的影响
    ax4 = fig.add_subplot(2, 3, 4)
    auto_levels = np.linspace(0, 1, 50)
    resource_cost = 1 - auto_levels * 0.9
    ax4.plot(auto_levels, resource_cost, 'purple', linewidth=2)
    ax4.fill_between(auto_levels, 0, resource_cost, alpha=0.2, color='purple')
    ax4.set_xlabel('自动化程度')
    ax4.set_ylabel('注意资源消耗')
    ax4.set_title('自动化降低资源消耗')
    ax4.grid(True, alpha=0.3)

    # 子图5: 双任务表现与自动化
    ax5 = fig.add_subplot(2, 3, 5)
    auto_levels = np.linspace(0, 1, 50)
    dual_task_perf = 0.3 + 0.7 * auto_levels
    ax5.plot(auto_levels, dual_task_perf, 'orange', linewidth=2)
    ax5.fill_between(auto_levels, 0, dual_task_perf, alpha=0.2, color='orange')
    ax5.set_xlabel('自动化程度（任务1）')
    ax5.set_ylabel('双任务表现（任务2）')
    ax5.set_title('自动化改善双任务表现')
    ax5.grid(True, alpha=0.3)

    # 子图6: 加工模式对比雷达图
    ax6 = fig.add_subplot(2, 3, 6, projection='polar')
    categories = ['速度', '意识参与', '资源消耗', '灵活性', '并行能力', '抗干扰']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    auto_vals = [0.9, 0.2, 0.2, 0.2, 0.8, 0.8]
    cont_vals = [0.3, 0.9, 0.9, 0.9, 0.2, 0.3]
    auto_vals += auto_vals[:1]
    cont_vals += cont_vals[:1]
    ax6.plot(angles, auto_vals, 'b-', label='自动化加工', linewidth=2)
    ax6.fill(angles, auto_vals, alpha=0.1, color='blue')
    ax6.plot(angles, cont_vals, 'r-', label='受控加工', linewidth=2)
    ax6.fill(angles, cont_vals, alpha=0.1, color='red')
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_title('加工模式特征对比')
    ax6.legend(loc='upper right')

    plt.suptitle('ACP自动与受控加工理论 - 核心概念可视化', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_acp_theory()
```

**可视化结果解读：**
- 子图1：CM条件下RT随练习指数下降（幂定律），VM条件下RT不降
- 子图2：CM条件下RT与集合大小无关（自动弹出），VM条件下RT线性增长（串行搜索）
- 子图3：自动化程度呈指数增长，早期进步快，后期趋于饱和
- 子图4：自动化程度越高，注意资源消耗越少
- 子图5：任务自动化后，双任务表现显著改善
- 子图6：雷达图展示了两种加工模式的全面对比

---

## 10. 模型评估

### 10.1 CM vs VM 实验预测验证

| 预测 | CM条件 | VM条件 |
|------|--------|--------|
| 集合大小效应 | 无（RT恒定） | 有（RT线性增长） |
| 练习效应 | 强（大幅下降） | 弱（几乎不变） |
| 双任务干扰 | 小 | 大 |
| 注意资源消耗 | 低 | 高 |
| 预期准确率 | 高 | 中 |

### 10.2 与资源理论的关系

| 维度 | 资源理论（Kahneman） | ACP理论 |
|------|--------------------|---------|
| 核心问题 | 资源如何分配 | 资源需求如何变化 |
| 动态机制 | 唤醒调节总资源 | 练习降低资源需求 |
| 任务差异 | 权重决定分配 | 一致性决定自动化潜力 |
| 互补性 | 静态分配框架 | 动态学习框架 |

### 10.3 影响自动化的关键因素

```python
def factors_affecting_automatization():
    factors = {
        "刺激-反应一致性": "高一致性促进自动化",
        "练习频率": "高频练习加速自动化",
        "反馈质量": "即时反馈有利于自动化",
        "任务复杂性": "简单任务更容易自动化",
        "个体差异": "工作记忆容量影响自动化速度"
    }
    for factor, desc in factors.items():
        print(f"{factor}: {desc}")
```

---

## 11. 常见问题与易错点

### Q1：自动化加工是否意味着"不需要注意"？
**答：** 不完全正确。自动化加工只需要极少量的注意资源，但完全"零注意"的情况很少。高度自动化的任务（如走路）仍然需要极少量的注意触发和监控。

### Q2：为什么VM条件下无法建立自动化？
**答：** 因为VM条件下，刺激-反应映射不断变化——这次是"目标"的刺激下次可能是"干扰物"。大脑无法形成稳定的"目标模板"，因此每次都需要受控加工来重新配置目标。

### Q3：自动化加工是否总比受控加工好？
**答：** 不是。自动化加工的缺点包括：（1）一旦启动难以中断（如开车时习惯性走老路）；（2）不灵活，当环境变化时可能产生错误；（3）难以有意识地控制。

### Q4：在深度学习中，预训练-微调是否类似ACP理论？
**答：** 是的。预训练可以类比为"自动化加工"的建立过程——通过大量数据训练让模型学到通用特征（自动化的"模板"）。微调类比为"受控加工"——针对特定任务的有意识调整。预训练越充分，微调需要的标注数据和计算资源越少。

---

## 12. 学习总结

### 12.1 核心贡献
ACP理论揭示了技能自动化的认知机制——通过一致映射条件下的反复练习，任务可以从资源消耗巨大的受控加工转变为几乎不消耗资源的自动化加工。这一理论为理解"熟能生巧"提供了认知层面的解释。

### 12.2 关键公式回顾
- 幂定律练习曲线：$RT(n) = RT_\infty + (RT_0 - RT_\infty) / (n+1)^\beta$
- 自动化程度：$A = 1 - R_{current} / R_{initial}$
- 资源消耗递减：$R(p) = R_0 \cdot e^{-\alpha p}$

### 12.3 理论联系
- Kahneman资源理论 -> 资源有限性基础
- Norman & Bobrow -> 任务类型分类
- **ACP理论 -> 解释资源需求变化的机制**
- 后续：Logan的实例理论（自动化依赖记忆提取）

---

## 13. 练习题与思考题（含答案）

**题目1：** 自动化加工和受控加工的核心区别是什么？
**答案：** 自动化加工速度快、无意识、资源消耗少、难中断；受控加工速度慢、有意识、资源消耗多、灵活。关键区别在于是否需要注意资源的持续投入。

**题目2：** 为什么一致映射（CM）条件可以建立自动化，而变异映射（VM）不行？
**答案：** CM条件下刺激-反应映射始终一致，大脑可以建立稳定的"目标模板"，从而在知觉阶段直接识别目标（自动弹出）。VM条件下映射不断变化，每次都需要重新配置目标模板，无法建立稳定的自动化加工。

**题目3：** 请举例说明自动化加工的"优势"和"劣势"。
**答案：** 优势：不需要注意资源，可以并行处理其他任务（如熟练打字时还能思考）。劣势：一旦启动难以控制（如走熟悉的错误路线），环境变化时容易出错（如开惯了自动挡换手动挡时的操作错误）。

**题目4：** 编程题——模拟CM和VM条件下的视觉搜索实验，验证集合大小效应。
**答案：**
```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_search(cm=True):
    set_sizes = np.arange(1, 13)
    RTs = []
    for s in set_sizes:
        if cm:
            RT = 250 + np.random.normal(0, 15)
        else:
            RT = 300 + 60 * s + np.random.normal(0, 30)
        RTs.append(RT)
    return set_sizes, np.array(RTs)

sizes, cm_rt = simulate_search(True)
_, vm_rt = simulate_search(False)

plt.plot(sizes, cm_rt, 'bo-', label='CM')
plt.plot(sizes, vm_rt, 'rs-', label='VM')
plt.xlabel('集合大小')
plt.ylabel('反应时间(ms)')
plt.legend()
plt.grid(True)
plt.show()
```

**题目5（思考题）：** 在当今的AI系统中（如GPT等大语言模型），预训练是否可以视为"自动化加工"，推理/微调是否可以视为"受控加工"？请分析两者的异同。
**答案：** 同：预训练通过大量一致数据训练（类似CM条件），使模型学会语言规律（自动化）；微调针对特定任务进行有意识调整（受控）。异：（1）AI的"自动化"是参数固化而非心理过程的自动化；（2）AI的"受控加工"没有意识参与；（3）AI可以在不同任务间灵活切换而不需要"重新练习"——这是数字系统与生物系统的本质区别。

---

## 14. 学习路径建议

**前置知识：**
- 卡内曼注意力资源理论（资源有限性）
- 诺曼与鲍勃罗多任务注意力理论（资源受限和数据受限）

**平行学习：**
- Logan的实例理论（自动化基于记忆提取）
- Anderson的ACT-R认知架构（程序性知识自动化）

**进阶方向：**
- 技能学习理论（Fitts & Posner三阶段模型）
- 工作记忆与长时工作记忆（Ericsson）
- 深度学习中的迁移学习和预训练

**推荐阅读顺序：**
1. 资源理论 -> 2. **ACP理论（本节）** -> 3. 实例理论 -> 4. ACT-R -> 5. 技能学习三阶段模型
