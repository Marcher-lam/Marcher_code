# Token 压缩技术 学习文档

> 来源线索：本节内容根据原书中关于"多模态特征token压缩"（第12章）的相关章节整理、扩展与教学化改写。

> 用更少的token表达同样多的信息——为多模态大模型减轻长序列的计算负担。

## 1. 算法基础认知

**一句话定义**：将长序列的token表示压缩为更短的序列，同时尽量保留关键信息。

**直觉类比**：看一部2小时的电影——你不需要记住每一帧画面，只需要记住关键情节（约20分钟摘要）。Token压缩就是让模型学会"摘要"——用少量token代表大量信息。

**历史背景**：随着多模态大模型的兴起，图像可能产生数百个patch token，音频也产生大量帧特征。这些长序列严重拖慢了Transformer的计算。Token压缩技术在2022-2024年间快速发展，包括ToMe（Token Merging）、AvgPool投影器、Pixel-Shuffle等方法。

**算法定位**：深度学习 / 序列压缩 / 多模态特征工程。

**前置知识**：
- Transformer和自注意力机制
- 池化操作（Average Pooling, Max Pooling）
- 多模态特征表示

## 2. 核心原理

### 核心思想

多模态模型中，不同模态产生的token数量差异巨大：图像ViT可能产生196-576个token，而文本只有几十个。Token压缩的目的是将图像/音频的冗长token序列压缩到与文本相当的长度，减少注意力计算的开销。

### 主要压缩方法

**1. 平均池化投影器（AvgPool Projector）**

将token序列视为特征图，用AdaptiveAvgPool压缩空间维度：

```
原始: (batch, 576, d) → reshape → (batch, d, 24, 24) → AvgPool → (batch, d, 6, 6) → reshape → (batch, 36, d)
```

**2. Pixel-Shuffle（像素重组）**

将通道维度的信息重排到空间维度（或反之），实现上/下采样：

```
下采样: (batch, H, W, C) → (batch, H/r, W/r, C*r²)
```

**3. Token Merging（ToMe）**

基于token相似度合并最相似的token对：
1. 计算所有token之间的余弦相似度
2. 用二分图匹配找到最相似的token对
3. 将相似token对合并（平均或加权）
4. 重复直到达到目标token数

**4. 跨层Token融合（Cross-layer Token Fusion）**

在不同Transformer层之间合并token，利用深层特征更抽象的特点进行压缩。

### 工作流程

1. 接收原始token序列 $(N, d)$
2. 选择压缩策略（池化/匹配/重排）
3. 输出压缩后的token序列 $(M, d)$，其中 $M \ll N$
4. 压缩后的token送入后续Transformer层

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $X$ | 原始token序列 | $(N, d)$ |
| $X'$ | 压缩后序列 | $(M, d)$, $M \ll N$ |
| $r$ | 压缩比 | $N/M$ |

### AvgPool投影

$$X' = \text{Reshape}(\text{AvgPool}(\text{Reshape}(X)))$$

将 $(N, d)$ reshape为 $(d, \sqrt{N}, \sqrt{N})$，池化到 $(d, \sqrt{M}, \sqrt{M})$，再reshape为 $(M, d)$。

### Token Merging (ToMe)

**步骤1**：计算token间的余弦相似度

$$\text{sim}(x_i, x_j) = \frac{x_i \cdot x_j}{\|x_i\| \cdot \|x_j\|}$$

**步骤2**：二分图匹配——将token分为两组A和B，在A→B之间找最大相似度匹配

**步骤3**：合并匹配的token对

$$x_{merged} = \frac{x_a + x_b}{2}$$

或带权合并：$x_{merged} = \frac{s_a \cdot x_a + s_b \cdot x_b}{s_a + s_b}$，其中 $s$ 是token的重要性分数。

### Pixel-Shuffle

下采样方向：
$$X'_{h,w,c} = X_{h \cdot r + i, w \cdot r + j, c \cdot r^2 + i \cdot r + j}$$

其中 $i, j \in [0, r)$，将空间信息折叠进通道维度。

## 4. 训练过程讲解

### 数据预处理

- 原始模态特征（如图像patch token或音频帧token）需要统一维度
- 压缩模块通常接在特征提取器之后、LLM之前

### 参数初始化

- 池化层无需参数
- 投影层使用Xavier初始化
- ToMe的合并操作是确定性的，无需训练参数

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| 目标token数 $M$ | 压缩后的序列长度 | 36-144 | 64 |
| 压缩比 $r$ | 压缩倍数 | 4-16 | 9 |
| 合并策略 | token如何合并 | avg/weighted | avg |

## 5. 应用场景

1. **多模态LLM输入压缩**：DeepSeek-VL2等模型需要将数百个图像token压缩后送入LLM。AvgPool投影器是主流方案。

2. **视频理解**：视频产生数千帧特征，必须大幅压缩才能进行有效推理。

3. **语音识别**：长音频的帧特征序列过长，压缩后可加速Transformer推理。

## 6. 优缺点分析

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| AvgPool | 简单/无需训练参数/确定性强 | 丢失空间位置信息 | 均匀信息分布的token |
| Pixel-Shuffle | 可逆/信息保留好 | 只适合2D特征图 | 图像patch特征 |
| ToMe | 信息保留最好/可解释 | 计算开销/需要相似度计算 | 需要高保真压缩 |
| 跨层融合 | 利用深层特征 | 实现复杂 | 多层Transformer |

## 7. 调库实现

```python
"""Token压缩技术的PyTorch实现"""
import torch
import torch.nn as nn


class AvgPoolProjector(nn.Module):
    """平均池化投影器：将长token序列压缩为短序列"""
    
    def __init__(self, d_model, target_tokens=64):
        super().__init__()
        self.target_tokens = target_tokens
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """
        x: (batch, N, d) - 原始token序列
        返回: (batch, M, d) - 压缩后序列
        """
        batch, N, d = x.shape
        
        # 计算空间维度（假设N是完全平方数或接近）
        import math
        h = w = int(math.sqrt(N))
        if h * w != N:
            # 如果不是完全平方数，截断到最近的完全平方数
            h = w = int(math.sqrt(N))
            x = x[:, :h*w, :]
        
        # Reshape: (batch, N, d) -> (batch, d, h, w)
        x = x.transpose(1, 2).reshape(batch, d, h, w)
        
        # 计算目标大小
        target_h = target_w = int(math.sqrt(self.target_tokens))
        
        # 自适应平均池化
        pool = nn.AdaptiveAvgPool2d((target_h, target_w))
        x = pool(x)  # (batch, d, target_h, target_w)
        
        # Reshape回序列: (batch, target_tokens, d)
        x = x.reshape(batch, d, -1).transpose(1, 2)
        
        return self.proj(x)


class TokenMerger(nn.Module):
    """Token Merging (ToMe): 基于相似度合并token"""
    
    def __init__(self, target_tokens=64):
        super().__init__()
        self.target_tokens = target_tokens
    
    def forward(self, x):
        """
        x: (batch, N, d)
        返回: (batch, M, d), M = target_tokens
        """
        batch, N, d = x.shape
        if N <= self.target_tokens:
            return x
        
        tokens_to_merge = N - self.target_tokens
        
        # 按token的L2范数作为重要性分数
        importance = x.norm(dim=-1)  # (batch, N)
        
        # 简单策略：按重要性排序，合并最不重要的token对
        # 这里使用更高效的实现：将token两两配对合并
        for _ in range(tokens_to_merge):
            # 计算所有token对的余弦相似度
            x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
            sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (batch, N, N)
            # 排除自身
            sim.fill_diagonal_(float('-inf'))
            
            # 找到最相似的token对
            flat_idx = sim.argmax(dim=-1)  # 每个token最相似的另一个token
            
            # 找全局最相似的对
            max_sim_vals = sim.max(dim=-1).values  # (batch, N)
            # 取最相似的对（避免重复）
            best_idx = max_sim_vals.argmax(dim=-1)  # (batch,)
            
            # 合并这对token
            for b in range(batch):
                i = best_idx[b].item()
                j = flat_idx[b, i].item()
                # 平均合并
                x[b, i] = (x[b, i] + x[b, j]) / 2
                # 删除第j个token
                x = torch.cat([x[:, :j, :], x[:, j+1:, :]], dim=1)
                break  # 每次只合并一对
        
        return x


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 模拟图像patch token序列
    batch, N, d = 2, 196, 128  # 196 = 14x14 patches
    tokens = torch.randn(batch, N, d)
    
    print("=== Token压缩测试 ===")
    print(f"原始token序列: {tokens.shape}")
    
    # AvgPool压缩
    avgpool = AvgPoolProjector(d, target_tokens=36)  # 压缩到36=6x6
    compressed = avgpool(tokens)
    print(f"AvgPool压缩后: {compressed.shape}")
    print(f"压缩比: {N}/{compressed.shape[1]} = {N/compressed.shape[1]:.1f}x")
```

## 8. 手工代码实现

```python
"""从零实现Token压缩（不使用nn.AdaptiveAvgPool）"""
import torch
import torch.nn as nn
import math


class ManualAvgPoolCompressor(nn.Module):
    """手写平均池化token压缩
    
    不使用nn.AdaptiveAvgPool2d，用基础张量操作实现。
    """
    
    def __init__(self, d_model, target_tokens=36):
        super().__init__()
        self.target_tokens = target_tokens
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """x: (batch, N, d)"""
        batch, N, d = x.shape
        
        # 计算空间维度
        h = w = int(math.sqrt(N))
        target_h = target_w = int(math.sqrt(self.target_tokens))
        
        # Reshape: (batch, N, d) -> (batch, h, w, d)
        x = x[:, :h*w, :].reshape(batch, h, w, d)
        
        # 手工平均池化: 将(h,w)划分为(target_h, target_w)个网格
        # 每个网格内的token取平均
        kh = h // target_h  # 池化核高度
        kw = w // target_w  # 池化核宽度
        
        output = torch.zeros(batch, target_h, target_w, d)
        for i in range(target_h):
            for j in range(target_w):
                # 取对应区域的token并平均
                region = x[:, i*kh:(i+1)*kh, j*kw:(j+1)*kw, :]  # (batch, kh, kw, d)
                output[:, i, j, :] = region.mean(dim=(1, 2))
        
        # Reshape: (batch, target_h, target_w, d) -> (batch, target_tokens, d)
        output = output.reshape(batch, -1, d)
        
        return self.proj(output)


class ManualToMeCompressor(nn.Module):
    """手写Token Merging压缩
    
    使用二分匹配策略合并最相似的token对。
    """
    
    def __init__(self, merge_ratio=0.5):
        super().__init__()
        self.merge_ratio = merge_ratio  # 合并多少比例的token
    
    def _cosine_similarity_matrix(self, x):
        """计算余弦相似度矩阵"""
        # x: (batch, N, d)
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.bmm(x_norm, x_norm.transpose(1, 2))
    
    def forward(self, x):
        """x: (batch, N, d)"""
        batch, N, d = x.shape
        target_N = int(N * (1 - self.merge_ratio))
        
        x = x.clone()
        current_N = N
        
        while current_N > target_N:
            # 计算相似度矩阵
            sim = self._cosine_similarity_matrix(x)  # (batch, current_N, current_N)
            
            for b in range(batch):
                # 排除对角线（自身相似度=1）
                sim[b].fill_diagonal_(float('-inf'))
                
                # 找到最相似的token对
                flat_idx = sim[b].argmax()
                i = flat_idx // current_N
                j = flat_idx % current_N
                
                if i == j:
                    continue
                
                # 确保i < j
                if i > j:
                    i, j = j, i
                
                # 合并: 平均两个token
                x[b, i] = (x[b, i] + x[b, j]) / 2
                
                # 删除第j个token
                indices = list(range(current_N))
                indices.pop(j)
                x_b = x[b]
                x[b] = x_b[indices]
            
            current_N -= 1
        
        return x[:, :target_N, :]


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 模拟图像token: 100个token, 64维
    tokens = torch.randn(2, 100, 64)
    print("=== 手写Token压缩测试 ===")
    print(f"原始: {tokens.shape}")
    
    # AvgPool压缩
    compressor1 = ManualAvgPoolCompressor(64, target_tokens=36)
    out1 = compressor1(tokens)
    print(f"AvgPool压缩: {out1.shape} (100 -> 36, 压缩2.8x)")
    
    # ToMe压缩
    compressor2 = ManualToMeCompressor(merge_ratio=0.5)
    out2 = compressor2(tokens)
    print(f"ToMe压缩: {out2.shape} (100 -> 50, 压缩2.0x)")
    
    # 信息保留度量: 计算压缩前后token均值的相似度
    orig_mean = tokens.mean(dim=1)
    comp1_mean = out1.mean(dim=1)
    sim = torch.cosine_similarity(orig_mean, comp1_mean, dim=-1)
    print(f"\nAvgPool信息保留(cosine sim): {sim.mean():.4f}")
```

## 9. 可视化与结果理解

```python
"""Token压缩效果可视化"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import math

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 压缩前后token数量对比
methods = ['原始\n(196 tokens)', 'AvgPool\n(36 tokens)', 'ToMe\n(64 tokens)', 'Pixel-Shuffle\n(49 tokens)']
token_counts = [196, 36, 64, 49]
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

axes[0].bar(methods, token_counts, color=colors, edgecolor='black')
for i, v in enumerate(token_counts):
    axes[0].text(i, v + 3, str(v), ha='center', fontweight='bold')
axes[0].set_title('不同压缩方法的Token数量', fontsize=14)
axes[0].set_ylabel('Token数量')

# 图2: 压缩比对注意力计算量的影响
ratios = [1, 2, 4, 8, 16]
original_cost = 196 ** 2  # 原始注意力复杂度
compressed_costs = [(196/r)**2 for r in ratios]
savings = [(1 - c/original_cost)*100 for c in compressed_costs]

axes[1].plot(ratios, savings, 'o-', color='#e74c3c', linewidth=2, markersize=8)
axes[1].fill_between(ratios, 0, savings, alpha=0.2, color='#e74c3c')
axes[1].set_title('Token压缩带来的注意力计算量节省', fontsize=14)
axes[1].set_xlabel('压缩比')
axes[1].set_ylabel('计算量节省 (%)')
axes[1].grid(True, alpha=0.3)
for r, s in zip(ratios, savings):
    axes[1].annotate(f'{s:.1f}%', (r, s), textcoords="offset points", xytext=(0, 10))

# 图3: 压缩精度trade-off
compress_ratios = np.array([1, 2, 4, 8, 16])
# 模拟的精度数据
accuracy_retention = [100, 98.5, 96.2, 91.8, 84.3]

axes[2].plot(compress_ratios, accuracy_retention, 's-', color='#2ecc71', linewidth=2, markersize=8)
axes[2].axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95%精度阈值')
axes[2].set_title('压缩比与精度保留的Trade-off', fontsize=14)
axes[2].set_xlabel('压缩比')
axes[2].set_ylabel('精度保留率 (%)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('token_compression_viz.png', dpi=100)
plt.show()

print("图1解读: 不同压缩方法将原始196个token压缩到不同数量")
print("图2解读: 压缩4倍时注意力计算量减少约93.75% (1-(196/4)^2/(196^2))")
print("图3解读: 压缩比在4-8倍时通常能保持95%以上的精度")
```

## 10. 模型评估

```python
def evaluate_compression(original_tokens, compressed_tokens):
    """评估token压缩的信息保留质量"""
    # 计算token序列的中心点（全局特征）
    orig_center = original_tokens.mean(dim=1)  # (batch, d)
    comp_center = compressed_tokens.mean(dim=1)
    
    # 余弦相似度: 衡量全局信息保留
    cos_sim = torch.cosine_similarity(orig_center, comp_center, dim=-1).mean()
    
    # 方差保留率: 衡量token多样性保留
    orig_var = original_tokens.var(dim=1).mean()
    comp_var = compressed_tokens.var(dim=1).mean()
    var_retention = comp_var / (orig_var + 1e-8)
    
    print(f"=== Token压缩质量评估 ===")
    print(f"压缩比: {original_tokens.shape[1]}/{compressed_tokens.shape[1]} = "
          f"{original_tokens.shape[1]/compressed_tokens.shape[1]:.1f}x")
    print(f"全局信息保留(cosine sim): {cos_sim:.4f}")
    print(f"多样性保留(variance ratio): {var_retention:.4f}")
```

## 11. 常见问题与易错点

### 数据层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 非完全平方数的token数 | AvgPool reshape失败 | N不是完全平方数 | 截断或padding到最近的完全平方数 |
| 不同样本token数不同 | 批处理失败 | 变长序列无法batch | 预先统一到固定长度 |

### 模型层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 压缩过度 | 下游任务精度显著下降 | 信息丢失太多 | 控制压缩比在4-8倍以内 |
| 位置信息丢失 | 模型对空间关系不敏感 | AvgPool丢失位置 | 添加可学习的位置编码或用Pixel-Shuffle保留空间关系 |

### 调参层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 目标token数选择 | 太少精度差/太多加速不明显 | 需要平衡 | 从64开始实验，根据任务调整 |

## 12. 学习总结

Token压缩是多模态大模型的关键效率优化技术，核心是在信息保留和计算效率间取得平衡。主要方法：
- **AvgPool**：简单高效，适合均匀特征
- **ToMe**：信息保留最好，基于相似度合并
- **Pixel-Shuffle**：保留空间结构，可逆

关键公式：注意力复杂度从 $O(N^2)$ 降为 $O(M^2)$，$M \ll N$。

## 13. 练习题与思考题

### 基础题1：压缩比计算

原始图像经ViT产生14×14=196个token，需要压缩到目标序列长度64。压缩比是多少？注意力计算量节省多少？

**参考答案**：
- 压缩比 = 196/64 ≈ 3.06x
- 原始注意力复杂度 = 196² = 38,416
- 压缩后 = 64² = 4,096
- 节省 = (38416-4096)/38416 = 89.3%

### 基础题2：AvgPool维度计算

输入token序列形状为(batch=4, N=144, d=256)。用AvgPool压缩到64个token，中间需要怎样的reshape？

**参考答案**：
1. (4, 144, 256) → transpose → (4, 256, 144)
2. (4, 256, 144) → reshape → (4, 256, 12, 12)  (因为√144=12)
3. AvgPool(12,12) → (8,8) → (4, 256, 8, 8) (√64=8)
4. reshape → (4, 256, 64) → transpose → (4, 64, 256)

### 进阶题：设计压缩策略

对于视频理解任务，一段视频产生16帧×49个patch = 784个token。如何设计压缩策略将token降到64？

**参考答案**：
推荐两阶段压缩：
1. **帧内压缩**：每帧49个token → AvgPool → 16个token (7×7→4×4)。16帧×16 = 256个token。
2. **帧间压缩**：256个token → ToMe → 64个token。利用相邻帧的冗余性合并。

理由：先压缩空间（帧内），再压缩时间（帧间），分阶段压缩比一次性压缩更高效。

### 开放思考题

随着KV Cache优化（如MLA）的发展，是否还有必要在输入端做Token压缩？还是说应该在每一层都做动态压缩？

**参考思路**：
两种优化解决不同瓶颈：
- **输入端Token压缩**：减少进入Transformer的token数，降低所有层的计算量
- **KV Cache优化（MLA）**：减少推理时KV缓存的显存占用
- 它们可以同时使用：输入端压缩减少token数，MLA进一步压缩KV缓存
- 动态层间压缩（如ToMe per layer）可能是最优方案，但实现复杂度更高

## 14. 学习路径建议

### 前置知识
- Transformer架构和注意力复杂度
- 池化操作和降采样
- 多模态特征表示

### 平行学习
- KV Cache优化（MQA/GQA/MLA）
- 模型量化与剪枝

### 进阶方向
- 动态token剪枝（训练时自适应决定保留哪些token）
- 稀疏注意力（只计算部分token对的注意力）
- 视频理解中的时空token压缩

### 推荐资源
1. **论文**：ToMe: Token Merging: Your ViT But Faster (Bolya et al., 2023)
2. **论文**：EfficientSAM: Leveraged Segment Anything (ICLR 2024)
3. **博客**：多模态大模型中视觉token压缩方案总结（知乎/技术博客）
