# KOCH模型 学习文档

> 第一个仿生计算注意力模型——用类神经元网络模拟视觉注意选择
>
> 来源线索：本节内容根据原书第2.2.1节"KOCH:第一个仿生计算模型"整理。

---

## 1. 算法基础认知

**一句话定义：** KOCH模型由Koch和Ullman于1985年提出，是第一个用于选择性视觉注意力的理论计算模型，通过特征图、显著性图和赢者通吃（WTA）网络模拟注意力的选择与转移。

**核心思想：** 模型由三个关键组件构成：
1. **特征图**：视觉场景的早期特征表示（颜色、方向、运动、距离）
2. **显著性图**：通过特征融合构建的全局显著性表示
3. **WTA网络**：从显著性图中选择唯一的注视点，并通过抑制已选位置实现注视点转移

**历史背景：** 1985年，Christof Koch和Shimon Ullman在MIT提出该模型，是计算机领域第一个注意力计算模型。它的生物学基础源于Treisman的特征整合理论（1980）和视觉注意的神经生理学发现。KOCH模型直接启发了ITTI模型（1998），可以说是所有计算注意力模型的始祖。

**为什么重要：** 在深度学习和强化学习出现之前，KOCH模型提供了第一个完整的注意力计算范式：特征分解 -> 显著性融合 -> WTA选择 -> 抑制返回。

---

## 2. 核心原理

### 2.1 模型架构

```
输入图像
  |
  v
特征提取 (颜色/方向/运动/距离)
  |     |     |     |
  v     v     v     v
特征图1 特征图2 ... 特征图n
  |     |     |     |
  +-----+-----+-----+
  |
  v
归一化与融合
  |
  v
显著性图
  |
  v
WTA网络 (赢者通吃)
  |
  v
唯一注视点 -> 中央表示
  |
  v
抑制已选位置
  |
  v
注视点转移 (临近偏好 / 相似偏好)
```

### 2.2 特征表示

在KOCH模型中，视觉场景被分解为多个特征维度：
- **颜色特征**：红/绿/蓝/黄双拮抗
- **方向特征**：不同朝向的Gabor滤波器响应
- **运动特征**：运动方向和速度
- **距离特征**：深度信息（立体视觉）

每个特征维度产生一张或多张特征图。

### 2.3 显著性图构建

多张特征图通过归一化和加权融合得到单一显著性图：

$$
S(x,y) = \sum_{k} w_k \cdot \mathcal{N}(F_k(x,y))
$$

其中 $F_k$ 是第 $k$ 张特征图，$w_k$ 是权重，$\mathcal{N}(\cdot)$ 是归一化函数。

### 2.4 WTA网络

WTA（Winner-Take-All）网络是一个竞争神经网络，其定义如下：

给定 $N$ 个输入神经元，每个神经元 $i$ 的激活为 $a_i$，WTA选择激活最大的神经元作为胜者：

$$
\text{Winner} = \arg\max_i a_i
$$

在KOCH模型中，WTA在显著性图上操作：

$$
\text{NextFixation} = \arg\max_{(x,y)} S(x,y)
$$

### 2.5 注视点转移规则

KOCH模型定义了两种转移机制：

1. **临近偏好 (Proximity Preference)**：倾向于选择当前注视点附近的显著位置
2. **相似偏好 (Similarity Preference)**：倾向于选择与当前点特征相似的位置

---

## 3. 数学公式与推导

### 3.1 WTA网络的神经动力学

Koch和Ullman使用神经振荡网络实现WTA。每个位置 $(x,y)$ 对应一个神经元，其膜电位动力学：

$$
\tau \frac{dV_{xy}}{dt} = -V_{xy} + S(x,y) - \alpha W_{xy} + \beta I_{xy}(t)
$$

其中 $W_{xy}$ 是抑制输入（来自其他神经元），$I_{xy}(t)$ 是外部输入。

当 $V_{xy}$ 首次达到阈值 $\theta$ 时，该神经元成为胜者。

### 3.2 抑制返回 (Inhibition of Return, IOR)

被选择的注视点在后续处理中被抑制，防止重复注意同一位置：

$$
S_{t+1}(x,y) = S_t(x,y) - \gamma \cdot G_\sigma(x - x_t^*, y - y_t^*)
$$

其中 $(x_t^*, y_t^*)$ 是当前注视点，$G_\sigma$ 是高斯基（抑制邻域），$\gamma$ 是抑制强度。

### 3.3 多尺度显著性

KOCH模型提出显著性可以在多个尺度上计算：

$$
S^{(s)}(x,y) = \sum_k \mathcal{N}(F_k^{(s)}(x,y))
$$

不同尺度的显著性图通过竞争或融合得到最终的注视点选择。

---

## 4. 训练过程讲解

KOCH模型是**无训练**的方法，完全基于预定义特征和WTA选择规则。

**处理流程：**
1. 提取多特征多尺度特征图
2. 特征图归一化（消除不同特征范围差异）
3. 加权融合得到显著性图
4. WTA选择最大显著性位置
5. 记录当前注视点
6. 在显著性图中抑制当前位置
7. 重复步4-6直到达到预定注视点数量

**没有"训练"只有"执行"的原因：** KOCH模型是纯理论计算模型，其所有参数（特征权重、抑制强度等）都是预设的，不需要从数据中学习。

---

## 5. 应用场景

| 场景 | 说明 |
|------|------|
| 视觉搜索 | 模拟人眼在场景中搜索目标 |
| 场景理解 | 按显著性次序处理图像区域 |
| 人机交互 | 预测用户的注意焦点 |
| 机器人视觉 | 驱动主动视觉系统的注视控制 |
| 计算神经科学 | 验证注意力的神经机制理论 |
| 图像压缩 | 基于注意力的非均匀编码 |

---

## 6. 优缺点分析

**优点：**
- ✅ **开创性**：第一个计算注意力模型，奠定理论基础
- ✅ **生物合理性**：基于神经生理学发现
- ✅ **框架清晰**：特征图 + 显著性图 + WTA 的三层架构
- ✅ **可扩展**：可以融入新的特征和转移规则

**缺点：**
- ❌ **特征手工**：需要手动设计特征提取器
- ❌ **WTA限制**：每次只能选择一个注视点，效率低
- ❌ **缺乏学习**：无法从任务中学习注意力策略
- ❌ **无自上而下**：只处理自下而上注意
- ❌ **实现不完整**：1985年论文是理论框架，缺乏完整实现细节

---

## 7. 调库实现

```python
"""KOCH模型 - 完整调库实现"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib.pyplot as plt


class KOCHModel:
    """Koch和Ullman的仿生注意力模型"""
    
    def __init__(self, n_features=3, inhibition_radius=20, inhibition_strength=0.5):
        """
        参数:
            n_features: 特征图数量
            inhibition_radius: 抑制返回半径
            inhibition_strength: 抑制强度
        """
        self.n_features = n_features
        self.inhibition_radius = inhibition_radius
        self.inhibition_strength = inhibition_strength
        self.attention_map = None
        self.fixations = []
    
    def compute_saliency(self, features):
        """
        从多张特征图融合为显著性图
        
        参数:
            features: 特征图数组 (n_features, H, W)
        
        返回:
            saliency: 融合后的显著性图 (H, W)
        """
        n, h, w = features.shape
        saliency = np.zeros((h, w))
        
        for k in range(n):
            fm = features[k]
            # 每个特征图独立归一化
            fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            # 全局提升：抑制响应均匀的特征图
            # 如果特征图的最大值远大于均值，说明特征图有明确峰值，给予更高权重
            peak_ratio = fm_norm.max() / (fm_norm.mean() + 1e-8)
            weight = np.clip(peak_ratio, 0.5, 2.0)
            saliency += weight * fm_norm
        
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        self.attention_map = saliency.copy()
        return self.attention_map
    
    def wta_select(self, saliency_map, inhibited_positions=None):
        """
        WTA赢者通吃：选择显著性最大的位置
        
        参数:
            saliency_map: 显著性图
            inhibited_positions: 已抑制位置列表
        
        返回:
            (y, x): 注视点坐标
        """
        working_map = saliency_map.copy()
        
        if inhibited_positions is not None:
            for y, x in inhibited_positions:
                # 高斯形状抑制
                h, w = working_map.shape
                yy, xx = np.mgrid[0:h, 0:w]
                mask = np.exp(-((yy-y)**2 + (xx-x)**2) / (2 * self.inhibition_radius**2))
                working_map -= self.inhibition_strength * mask
                working_map = np.clip(working_map, 0, 1)
        
        max_pos = np.unravel_index(working_map.argmax(), working_map.shape)
        return max_pos
    
    def inhibit_position(self, saliency_map, pos):
        """抑制已选位置（IOR）"""
        y, x = pos
        h, w = saliency_map.shape
        yy, xx = np.mgrid[0:h, 0:w]
        mask = np.exp(-((yy-y)**2 + (xx-x)**2) / (2 * self.inhibition_radius**2))
        inhibited = saliency_map - self.inhibition_strength * mask
        return np.clip(inhibited, 0, 1)
    
    def scan_image(self, features, n_fixations=5):
        """
        在图像上依次选择多个注视点
        
        参数:
            features: 特征图数组
            n_fixations: 注视点数量
        
        返回:
            fixations: 注视点坐标列表
        """
        saliency = self.compute_saliency(features)
        self.fixations = []
        current_map = saliency.copy()
        
        for _ in range(n_fixations):
            pos = self.wta_select(current_map, self.fixations)
            self.fixations.append(pos)
            current_map = self.inhibit_position(current_map, pos)
        
        return self.fixations
    
    def transfer_attention(self, current_pos, saliency_map, mode='proximity', features=None):
        """注意力转移"""
        h, w = saliency_map.shape
        working = saliency_map.copy()
        
        if mode == 'proximity':
            # 局部搜索：在当前点附近找局部极大值
            # 使用 maximum_filter 找所有局部极大值
            local_max = maximum_filter(working, size=11)
            maxima = (working == local_max) & (working > 0.1)
            
            # 找离当前点最近的局部极大值
            y, x = current_pos
            candidates = np.argwhere(maxima)
            if len(candidates) > 0:
                distances = np.sum((candidates - [y, x])**2, axis=1)
                # 排除当前点
                far_enough = distances > 4
                if far_enough.any():
                    best_idx = np.argmin(distances[far_enough])
                    return tuple(candidates[far_enough][best_idx])
        
        # 默认：全局搜索最大值
        working[current_pos] = 0
        return np.unravel_index(working.argmax(), working.shape)


def demo():
    model = KOCHModel(n_features=3, inhibition_radius=15, inhibition_strength=0.4)
    np.random.seed(42)
    
    h, w = 100, 100
    features = np.random.randn(3, h, w) * 0.2 + 0.5
    
    # 在特征图中加入显著目标
    features[0, 25:35, 30:40] = 1.8
    features[1, 60:70, 50:60] = 1.5
    features[2, 40:45, 70:75] = 0.1
    
    # 扫描
    fixations = model.scan_image(features, n_fixations=4)
    print("注视点序列:")
    for idx, pos in enumerate(fixations):
        print(f"  {idx+1}: ({pos[0]}, {pos[1]})")
    
    # 可视化
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(model.attention_map, cmap='hot')
    for idx, pos in enumerate(fixations):
        ax.plot(pos[1], pos[0], 'co', markersize=12, markeredgewidth=2, 
                markerfacecolor='none')
        ax.text(pos[1]+3, pos[0], str(idx+1), color='cyan', fontsize=12)
    ax.set_title('显著性图与注视点序列')
    plt.tight_layout()
    plt.savefig('koch_fixations.png', dpi=150)
    print("已保存至 koch_fixations.png")


if __name__ == "__main__":
    demo()
```

---

## 8. 手工代码实现

```python
"""KOCH模型 - 手工WTA实现"""
import numpy as np


def manual_maximum_filter(data, size):
    """手工最大值滤波（找局部极值）"""
    h, w = data.shape
    radius = size // 2
    result = np.zeros_like(data)
    for i in range(h):
        for j in range(w):
            y1 = max(0, i-radius); y2 = min(h, i+radius+1)
            x1 = max(0, j-radius); x2 = min(w, j+radius+1)
            result[i,j] = data[y1:y2, x1:x2].max()
    return result


def wta_manual(saliency_map):
    """手工WTA选择"""
    max_pos = np.unravel_index(saliency_map.argmax(), saliency_map.shape)
    return max_pos


def inhibit_manual(saliency_map, pos, radius=15, strength=0.4):
    """手工抑制返回"""
    y, x = pos
    h, w = saliency_map.shape
    result = saliency_map.copy()
    for i in range(h):
        for j in range(w):
            dist2 = (i-y)**2 + (j-x)**2
            inhibition = strength * np.exp(-dist2 / (2 * radius**2))
            result[i,j] = max(0, result[i,j] - inhibition)
    return result


def koch_scan_manual(features, n_fixations=5):
    """手工KOCH扫描"""
    n, h, w = features.shape
    saliency = np.zeros((h,w))
    for k in range(n):
        fm = features[k]
        fm_min, fm_max = fm.min(), fm.max()
        fm_norm = (fm - fm_min) / (fm_max - fm_min + 1e-8)
        peak = fm_norm.max() / (fm_norm.mean() + 1e-8)
        weight = max(0.5, min(2.0, peak))
        saliency += weight * fm_norm
    
    s_min, s_max = saliency.min(), saliency.max()
    saliency = (saliency - s_min) / (s_max - s_min + 1e-8)
    
    fixations = []
    current = saliency.copy()
    
    for _ in range(n_fixations):
        pos = wta_manual(current)
        fixations.append(pos)
        current = inhibit_manual(current, pos)
    
    return fixations, saliency


def test_koch():
    np.random.seed(42)
    h, w = 50, 50
    features = np.random.rand(3, h, w) * 0.3 + 0.4
    features[0, 12:18, 15:22] = 1.8
    features[1, 30:36, 25:31] = 1.5
    
    fix, smap = koch_scan_manual(features, 3)
    print("手工KOCH注视点:", fix)
    assert len(fix) == 3, "应生成3个注视点"
    # 第一个注视点应在最显著区域
    assert (10 <= fix[0][0] <= 20) and (13 <= fix[0][1] <= 24), \
        "第一个注视点应在最显著区域"
    print("测试通过")


if __name__ == "__main__":
    test_koch()
```

---

## 9. 可视化与结果理解

### 9.1 显著性图热力图

- 红色区域：显著性高
- 蓝色区域：显著性低
- 注视点序列：从最显著到次显著递减

### 9.2 抑制返回的效果

- 第一次注视：最显著目标
- 第二次注视：次显著目标（已被抑制的最显著目标降低）
- 后续注视：依次转向其他未访问的显著区域

### 9.3 注视路径

人类的自然扫描路径通常呈现出"就近转移"的特点，KOCH模型的临近偏好的模拟了这一现象。

---

## 10. 模型评估

```python
"""KOCH模型评估"""
import numpy as np


def evaluate_koch():
    model = KOCHModel(n_features=3, inhibition_radius=15, inhibition_strength=0.4)
    np.random.seed(42)
    h, w = 50, 50
    features = np.random.rand(3, h, w) * 0.3 + 0.4
    features[0, 20:30, 25:35] = 1.8
    features[1, 20:30, 25:35] = 1.6
    
    fixations = model.scan_image(features, n_fixations=5)
    
    # 第一个注视点应该在显著区域内
    first_in_target = (20 <= fixations[0][0] <= 30) and (25 <= fixations[0][1] <= 35)
    print(f"第一个注视点在目标内: {first_in_target}")
    print(f"注视点序列: {fixations}")
    print(f"注视点间距离均值: {np.mean([np.sqrt((fixations[i][0]-fixations[i+1][0])**2 + (fixations[i][1]-fixations[i+1][1])**2) for i in range(len(fixations)-1])):.2f}")


if __name__ == "__main__":
    evaluate_koch()
```

---

## 11. 常见问题与易错点

### Q1: KOCH模型与ITTI模型的关系？
**A:** ITTI（1998）是KOCH（1985）的具体实现和扩展。ITTI增加了多尺度处理、中心-周围差分的具体算法、特征图归一化的具体方案。KOCH是理论框架，ITTI是工程实现。

### Q2: WTA网络为什么是必要的？
**A:** 视觉系统在同一时刻只能聚焦于一个位置进行精细化处理。WTA模拟了这一"瓶颈"特性——从大量并行处理的前注意阶段切换到串行的注意聚焦阶段。

### Q3: 抑制返回（IOR）的生物学意义？
**A:** IOR防止视觉系统反复注意同一个位置，促进探索新位置。神经生理学上，IOR与上丘（Superior Colliculus）的神经元活动相关。

### Q4: 为什么注视点有"临近偏好"？
**A:** 从当前注视点转移到附近位置的成本低于跳转到远处。这是视觉搜索中的一种优化策略。

### Q5: KOCH模型能处理动态场景吗？
**A:** 理论上可以，通过加入运动特征图。但1985年的论文主要关注静态场景。

---

## 12. 学习总结

**核心贡献：**
1. 第一个计算注意力架构：特征图 -> 显著性图 -> WTA
2. 引入WTA和IOR作为注意力选择与转移的机制
3. 为所有后续计算注意力模型（ITTI、GBVS等）奠定基础

**历史地位：** KOCH模型是连接神经科学和计算机视觉的桥梁。它首次将注意力的生物学概念转化为可计算的算法框架。

---

## 13. 练习题与思考题

### 基础题

**1.** KOCH模型中"显著性图"和"特征图"有什么区别？

<details>
<summary>答案</summary>
特征图是单一维度的视觉特征表示（如红色响应），是多张并行图。显著性图是特征图经归一化、加权融合后得到的单一图，表示每个位置的全局显著性。
</details>

**2.** WTA网络的输入和输出是什么？

<details>
<summary>答案</summary>
输入：显著性图中所有位置的显著性值。输出：显著性最大位置的坐标（唯一的注视点）。
</details>

**3.** 为什么需要抑制已选择的注视点？

<details>
<summary>答案</summary>
防止反复注意同一位置，促进视觉系统探索新区域。如果没有抑制，WTA会反复选择同一个最显著的位置。
</details>

### 进阶题

**4.** 如果多张特征图之间有冲突（同一位置在不同特征图中显著性相反），KOCH模型如何处理？

<details>
<summary>答案</summary>
KOCH使用加权求和融合特征图，冲突的特征图通过权重调整：如果一张特征图的响应均匀（无明确峰值），其权重较低；如果特征图有明显峰值，权重较高。这是通过"全局提升"归一化实现的。
</details>

**5.** 设计一个实验来验证KOCH模型的预测与人眼注视的一致性。

<details>
<summary>答案</summary>
(1) 收集自然图像；(2) 用眼动仪记录被试的自由观看注视点；(3) 用KOCH模型计算显著性图和注视点序列；(4) 比较：注视点是否落在高显著性区域；前3个注视点的顺序是否一致；注视路径的统计特性是否相似。
</details>

---

## 14. 学习路径建议

### 预备知识
- 认知心理学中的注意理论
- 视觉神经生理学基础
- 数字图像处理基础

### 进阶方向
1. **KOCH -> ITTI**：从理论到实现的里程碑
2. **KOCH -> Guided Search**：Wolfe的引导搜索理论
3. **KOCH -> RAM/DRAM**：深度强化学习注意力模型

### 推荐阅读
- Koch & Ullman. "Shifts in Selective Visual Attention: Towards the Underlying Neural Circuitry." Human Neurobiology, 1985.
- Itti et al. "A Model of Saliency-Based Visual Attention for Rapid Scene Analysis." TPAMI 1998.
- Treisman & Gelade. "A Feature-Integration Theory of Attention." Cognitive Psychology, 1980.

### 项目实践
1. 实现KOCH + ITTI的完整特征提取pipeline
2. 在眼动数据集上比较不同WTA变体的性能
3. 扩展KOCH模型以支持自上而下的任务驱动注意
