# PEFT 参数高效微调 学习文档

> 来源线索：本节内容根据原书中关于"PEFT参数高效微调"（第8章 8.2节）的相关章节整理、扩展与教学化改写。

> 不动原模型，只调小部分——PEFT让普通GPU也能微调百亿参数大模型。

## 1. 算法基础认知

**一句话定义**：PEFT通过只训练模型中极少量的额外参数来实现对预训练大模型的高效微调。

**直觉类比**：想象一座已经建好的大厦（预训练模型）。全量微调相当于对大厦进行全面改造（成本极高）。PEFT方法则像是只在几个房间添加家具（Adapter）、在大门贴上指引牌（Prompt Tuning）或安装几块智能面板（LoRA），用极少的改动就能适应新用途。

**历史背景**：随着预训练模型规模从亿级增长到千亿级，全量微调变得不现实（GPU显存不够、训练时间长）。PEFT方法在2020-2023年间快速发展，包括Adapter（2019）、Prefix-Tuning（2021）、Prompt Tuning（2021）、LoRA（2021）等方法。

**算法定位**：深度学习 / 模型微调 / 参数高效方法。是一类方法的统称。

**前置知识**：
- 预训练-微调范式
- Transformer架构
- 反向传播和梯度更新

## 2. 核心原理

### 核心思想

PEFT的核心思想是**冻结预训练模型的大部分参数，只训练少量新增参数**：

- 新增参数可以是：适配器层、前缀向量、提示token、低秩矩阵等
- 训练参数通常仅占总参数的0.01%-3%
- 效果通常能达到全量微调的90%-99%

### 主要PEFT方法

**1. Adapter Tuning（适配器微调）**
在Transformer层中插入小型Adapter模块（两层FFN + 残差连接），只训练Adapter参数。

**2. Prefix-Tuning（前缀微调）**
在输入序列前添加可学习的"虚拟前缀"token，只训练这些前缀参数。

**3. Prompt Tuning（提示微调）**
Prefix-Tuning的简化版，只在输入层添加可学习的提示嵌入。

**4. LoRA（低秩适配）**
在权重矩阵旁添加低秩分解矩阵（已单独成文）。

**5. P-Tuning v2**
Prefix-Tuning的改进，在每层都添加可学习的前缀。

### 工作流程

1. 加载预训练模型并冻结所有参数
2. 添加PEFT模块（Adapter/Prefix/LoRA等）
3. 只优化PEFT模块的参数
4. 推理时可选择合并或保持分离

## 3. 数学公式与推导

### 符号约定

| 符号 | 含义 |
|------|------|
| $\theta$ | 预训练参数（冻结） |
| $\phi$ | PEFT可训练参数 |
| $L$ | Transformer层数 |
| $r$ | 降维维度（Adapter）或秩（LoRA） |

### Adapter

$$h' = h + f_{down}(\text{ReLU}(f_{up}(h)))$$

其中 $f_{down}: d \rightarrow r$，$f_{up}: r \rightarrow d$，$r \ll d$。

### Prefix-Tuning

$$\text{Prefix} = [P_1, P_2, ..., P_k] \in \mathbb{R}^{k \times d}$$

$$h = [\text{Prefix}; \text{Original\_Input}]$$

前缀token参与注意力计算但不对应真实输入。

### 参数效率分析

| 方法 | 新增参数量 | 相对全量微调 |
|------|-----------|-------------|
| Adapter | $2 \times L \times d \times r$ | ~2% |
| Prefix-Tuning | $L \times k \times d$ | ~0.1% |
| Prompt Tuning | $k \times d$ | ~0.01% |
| LoRA | $2 \times M \times r \times d$ | ~0.5% |

## 4. 训练过程讲解

### 数据预处理

与标准微调相同。根据任务准备数据。

### 超参数表

| 超参数 | 作用 | 推荐范围 | 默认建议 |
|--------|------|----------|----------|
| PEFT方法类型 | 选择方法 | LoRA/Adapter/Prefix | LoRA |
| 秩/瓶颈维度 | 控制表达力 | 4-64 | 8-16 |
| 学习率 | PEFT参数的学习率 | 1e-4 到 1e-3 | 3e-4 |
| 前缀长度 | Prefix的token数 | 10-200 | 20-50 |

## 5. 应用场景

1. **大语言模型指令微调**：对LLaMA、DeepSeek等模型进行领域适配或指令跟随训练。

2. **多任务学习**：为每个任务训练独立的PEFT模块，共享基础模型。

3. **资源受限环境**：在单张GPU上微调数十亿参数的模型。

4. **多模态模型适配**：对CLIP等模型进行下游任务适配。

## 6. 优缺点分析

| 优点 | 缺点 |
|------|------|
| 显著减少训练资源需求 | 某些复杂任务效果不如全量微调 |
| 存储效率高（只保存PEFT权重） | 方法选择需要经验 |
| 适合多任务切换 | 不同PEFT方法之间的兼容性问题 |
| 减少灾难性遗忘风险 | 增加了实现复杂度 |

## 7. 调库实现

```python
"""使用 PyTorch 实现常见PEFT方法"""
import torch
import torch.nn as nn


class AdapterLayer(nn.Module):
    """Adapter微调模块"""
    
    def __init__(self, d_model, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # 零初始化up_proj，初始时Adapter不影响输出
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        return x + self.dropout(self.up_proj(self.act(self.down_proj(x))))


class PrefixEncoder(nn.Module):
    """Prefix-Tuning编码器"""
    
    def __init__(self, d_model, prefix_length=20, num_layers=6):
        super().__init__()
        self.prefix_length = prefix_length
        # 每层都有独立的前缀参数
        self.prefix_embeddings = nn.ParameterDict({
            f'layer_{l}': nn.Parameter(torch.randn(2, prefix_length, d_model) * 0.02)
            for l in range(num_layers)
        })
        # 2 = 一份给Key, 一份给Value
    
    def forward(self, layer_idx):
        """返回指定层的前缀参数"""
        params = self.prefix_embeddings[f'layer_{layer_idx}']
        prefix_k = params[0]  # (prefix_length, d_model)
        prefix_v = params[1]  # (prefix_length, d_model)
        return prefix_k, prefix_v


class PromptTuning(nn.Module):
    """Prompt Tuning模块"""
    
    def __init__(self, d_model, num_tokens=20):
        super().__init__()
        self.prompt_embeddings = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        self.num_tokens = num_tokens
    
    def forward(self, input_embeds):
        """在输入前添加可学习的prompt token"""
        batch = input_embeds.shape[0]
        prompts = self.prompt_embeddings.unsqueeze(0).expand(batch, -1, -1)
        return torch.cat([prompts, input_embeds], dim=1)


class TransformerWithAdapter(nn.Module):
    """带Adapter的Transformer层"""
    
    def __init__(self, d_model, num_heads, d_ff, bottleneck_dim=64):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Adapter层
        self.attn_adapter = AdapterLayer(d_model, bottleneck_dim)
        self.ffn_adapter = AdapterLayer(d_model, bottleneck_dim)
    
    def forward(self, x, mask=None):
        # 注意力 + Adapter
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.attn_adapter(attn_out)
        
        # FFN + Adapter
        x = x + self.ffn_adapter(self.ffn(self.norm2(x)))
        return x


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    d_model = 256
    
    print("=== PEFT方法测试 ===")
    
    # Adapter测试
    adapter = AdapterLayer(d_model, bottleneck_dim=32)
    x = torch.randn(2, 10, d_model)
    out = adapter(x)
    print(f"Adapter: {x.shape} -> {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in adapter.parameters()):,}")
    
    # Prefix-Tuning测试
    prefix = PrefixEncoder(d_model, prefix_length=10, num_layers=4)
    pk, pv = prefix(0)
    print(f"\nPrefix-Tuning: K={pk.shape}, V={pv.shape}")
    print(f"  参数量: {sum(p.numel() for p in prefix.parameters()):,}")
    
    # Prompt Tuning测试
    prompt = PromptTuning(d_model, num_tokens=5)
    input_emb = torch.randn(2, 8, d_model)
    combined = prompt(input_emb)
    print(f"\nPrompt Tuning: {input_emb.shape} -> {combined.shape}")
    print(f"  参数量: {sum(p.numel() for p in prompt.parameters()):,}")
    
    # 完整Adapter Transformer测试
    block = TransformerWithAdapter(d_model, 4, d_model*4, bottleneck_dim=32)
    total_params = sum(p.numel() for p in block.parameters())
    adapter_params = sum(p.numel() for n, p in block.named_parameters() if 'adapter' in n)
    print(f"\nAdapter Transformer:")
    print(f"  总参数: {total_params:,}")
    print(f"  Adapter参数: {adapter_params:,} ({adapter_params/total_params*100:.2f}%)")
```

## 8. 手工代码实现

```python
"""从零实现PEFT训练流程"""
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """简单的模型用于演示PEFT"""
    
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        return self.encoder(x)


def apply_peft_adapter(model, bottleneck_dim=32):
    """为模型的每个Linear层添加Adapter"""
    adapters = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            d_in = module.in_features
            d_out = module.out_features
            
            adapter = nn.Sequential(
                nn.Linear(d_out, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, d_out)
            )
            # 零初始化最后一层
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
            
            module.adapter = adapter
            adapters.append(adapter)
            
            # 冻结原始权重
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            
            # 修改forward
            original_forward = module.forward
            
            def make_forward(orig, adapt):
                def new_forward(x):
                    return orig(x) + adapt(orig(x))
                return new_forward
            
            module.forward = make_forward(original_forward, adapter)
    
    # 收集可训练参数
    trainable = []
    for adapter in adapters:
        trainable.extend(adapter.parameters())
    
    return trainable


# ====== 测试 ======
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 模拟"预训练"模型
    model = SimpleModel(128, 5)
    
    total_before = sum(p.numel() for p in model.parameters())
    print(f"原始参数量: {total_before:,}")
    
    # 应用PEFT
    trainable = apply_peft_adapter(model, bottleneck_dim=16)
    
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"冻结参数: {frozen_count:,}")
    print(f"可训练参数: {trainable_count:,} ({trainable_count/total_before*100:.2f}%)")
    
    # 训练测试
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    x = torch.randn(8, 128)
    labels = torch.randint(0, 5, (8,))
    
    for step in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, labels)
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}: loss = {loss.item():.4f}")
```

## 9. 可视化与结果理解

```python
"""PEFT方法可视化对比"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 各PEFT方法的可训练参数占比
methods = ['全量微调', 'Adapter', 'Prefix\nTuning', 'Prompt\nTuning', 'LoRA']
params_pct = [100, 2.0, 0.1, 0.01, 0.5]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

axes[0].bar(methods, params_pct, color=colors, edgecolor='black')
for i, v in enumerate(params_pct):
    axes[0].text(i, v + 2, f'{v}%', ha='center', fontweight='bold')
axes[0].set_title('可训练参数占比', fontsize=13)
axes[0].set_ylabel('参数占比 (%)')
axes[0].set_yscale('log')
axes[0].set_ylim(0.005, 200)

# 图2: 各方法的性能保留率
perf_retention = [100, 98, 95, 90, 99]
axes[1].barh(methods, perf_retention, color=colors, edgecolor='black')
for i, v in enumerate(perf_retention):
    axes[1].text(v + 0.5, i, f'{v}%', va='center')
axes[1].set_title('性能保留率（相对全量微调）', fontsize=13)
axes[1].set_xlabel('性能保留率 (%)')
axes[1].set_xlim(85, 105)

# 图3: 推理开销对比
inference_overhead = [0, 5, 10, 0, 0]  # %
axes[2].bar(methods, inference_overhead, color=colors, edgecolor='black')
axes[2].set_title('额外推理开销', fontsize=13)
axes[2].set_ylabel('推理延迟增加 (%)')

plt.tight_layout()
plt.savefig('peft_viz.png', dpi=100)
plt.show()

print("图1解读: LoRA和Prompt Tuning参数最少, 全量微调参数最多")
print("图2解读: LoRA性能最接近全量微调, Prompt Tuning最弱")
print("图3解读: LoRA可合并无额外开销, Adapter和Prefix有推理开销")
```

## 10. 模型评估

PEFT微调的评估与标准微调相同，使用下游任务指标。额外关注：
- **训练效率**：GPU显存占用、训练时间
- **参数效率**：可训练参数占总参数的比例
- **性能保持**：与全量微调的性能差距

## 11. 常见问题与易错点

### 数据层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 微调数据与预训练分布差异大 | PEFT效果差 | 低参数量难以弥补大分布差异 | 增大PEFT模块容量或分阶段微调 |

### 模型层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| 忘记冻结原始权重 | 显存不够 | 原始参数也在更新 | 检查requires_grad=False |
| Adapter推理变慢 | 延迟增加 | 额外的Adapter层 | 合并Adapter权重或使用LoRA |

### 调参层面

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|----------|
| PEFT方法选择 | 不确定用哪种 | 每种方法适用场景不同 | 默认推荐LoRA |

## 12. 学习总结

PEFT是一类参数高效微调方法，核心思想是冻结预训练参数，只训练少量新增参数。主流方法包括：
- **LoRA**：参数少、无推理开销、效果最好（推荐首选）
- **Adapter**：实现简单、有推理开销
- **Prefix/Prompt Tuning**：参数最少、效果稍弱

关键公式：$h = f_{frozen}(x) + f_{trainable}(x)$

## 13. 练习题与思考题

### 基础题1：参数量对比

一个7B参数的LLM（70亿参数），使用LoRA（r=8）微调Q和V投影（共64层），每层d=4096。LoRA参数量是多少？

**参考答案**：
- 每层Q+V的LoRA参数 = 2 × 2 × r × d = 2 × 2 × 8 × 4096 = 131,072
- 64层总LoRA参数 = 131,072 × 64 = 8,388,608 ≈ 8.4M
- 占比 = 8.4M / 7B ≈ 0.12%

### 基础题2：Adapter参数量

一个Transformer层d_model=768，Adapter瓶颈维度=48。Adapter新增多少参数？

**参考答案**：
- down_proj: 768 × 48 = 36,864
- up_proj: 48 × 768 = 36,864
- 总计: 73,728 参数（如果包含偏置则更多）

### 进阶题：PEFT方法选择

给定以下场景，推荐最合适的PEFT方法：(1) 单任务分类，GPU显存有限；(2) 多任务切换，需要快速切换；(3) 生成任务，需要高精度。

**参考答案**：
1. **单任务分类，显存有限**：推荐Prompt Tuning（参数最少，分类任务效果够用）
2. **多任务切换**：推荐LoRA（每个任务独立LoRA权重，可快速切换且无推理开销）
3. **生成任务，高精度**：推荐LoRA（r=16-32）或Adapter（表达力更强）

### 开放思考题

PEFT是否会导致"灾难性遗忘"问题减轻？为什么？如果需要让模型学习全新能力（而非适配），PEFT是否还合适？

**参考思路**：
- PEFT确实减轻了灾难性遗忘，因为原始权重被冻结，知识被保留
- 但如果需要学习全新能力（如新语言、新模态），低秩参数可能不够
- 此时可以考虑：更大的r、更多层的PEFT、或混合策略（冻结大部分层，全量微调最后几层）

## 14. 学习路径建议

### 前置知识
- 预训练-微调范式
- Transformer架构
- 反向传播

### 平行学习
- LoRA详解（核心方法）
- 量化技术（QLoRA的前提）

### 进阶方向
- 多任务PEFT（Multi-task Adapter Fusion）
- PEFT与量化的结合（QLoRA、GPTQ-LoRA）
- 自动PEFT配置搜索

### 推荐资源
1. **论文**：Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning (He et al., 2022) — PEFT综述
2. **库**：Hugging Face PEFT库 — 包含所有主流PEFT方法
3. **论文**：Towards a Unified View of Parameter-Efficient Transfer Learning (He et al., 2022)
