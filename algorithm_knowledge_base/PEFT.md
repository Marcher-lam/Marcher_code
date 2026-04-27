# PEFT 学习文档

## 1. 算法基础认知

PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）是大规模语言模型时代的关键技术。当模型的参数量从数亿增长到数千亿甚至万亿时，传统的全参数微调（Full Fine-Tuning）变得不切实际：需要巨大的GPU显存、漫长的训练时间和计算资源。PEFT通过只更新少量参数，实现与全参数微调相当的性能，同时大幅降低成本。

### 1.1 为什么需要PEFT？

大规模语言模型的挑战：
- **显存消耗**：175B参数的模型需要数千GB GPU显存
- **训练时间**：全参数微调可能需要数周
- **存储成本**：每个任务需要存储完整模型
- **灾难性遗忘**：全参数微调会丧失预训练能力

PEFT的核心优势：
- **只更新少量参数**：通常1-10%的参数量
- **降低显存需求**：可以在消费级GPU上训练
- **快速部署**：多个任务可共用主干网络
- **保持预训练能力**：避免灾难性遗忘

### 1.2 PEFT的发展历程

| 年份 | 方法 | 介绍 |
|------|------|------|
| 2017 | Adapter | 首次提出Adapter |
| 2019 | Prefix Tuning | 前缀调优 |
| 2021 | Prompt Tuning | 提示调优 |
| 2021 | LoRA | 低秩适应 |
| 2022 | QLoRA | 量化+LoRA |
| 2023 | IA³ | 缩放适配器 |

### 1.3 PEFT的分类

PEFT方法可分为四大类：
1. **Additive方法**：添加额外参数
   - Prefix Tuning：添加连续前缀
   - Prompt Tuning：添加可学习提示
2. **Selective方法**：选择部分参数
   - BitFit：只更新偏置
   - Freeze：冻结大部分层
3. **Reparametrization方法**：低秩重构
   - LoRA：低秩矩阵分解
   - AdaLoRA：自适应低秩
4. **Hybrid方法**：混合多种方法

## 2. 核心原理

### 2.1 LoRA原理

LoRA（Low-Rank Adaptation）是目前最流行的PEFT方法，核心思想是：在预训练模型的权重矩阵旁边添加低秩矩阵，通过训练低秩矩阵来间接更新大模型。

**LoRA的核心假设**：预训练语言模型具有低秩特性，即模型参数的更新本质上是低秩的。

**数学公式**：
对于预训练权重 $W_0 \in \mathbb{R}^{d \times k}$，LoRA添加两个低秩矩阵：
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$

其中 $r \ll \min(d, k)$ 通常设为8, 16, 32, 64。

前向传播时：
$$h = W_0 x + B A x$$

其中 $W_0$ 冻结，只训练 $B$ 和 $A$。

训练完成后，可以将 $BA$ 合并回 $W_0$：
$$W_{merged} = W_0 + BA$$

### 2.2 Adapter原理

Adapter在Transformer层中添加小型网络模块：

```
Transformer Layer:
  → Attention → [Adapter] → LayerNorm → Residual
  → FFN → [Adapter] → LayerNorm → Residual
```

Adapter结构：
- Down-project: $d_{model} \to d_{adapter}$
- Non-linear: ReLU/GELU
- Up-project: $d_{adapter} \to d_{model}$
- LayerNorm后的残差连接

### 2.3 Prefix Tuning原理

Prefix Tuning在输入序列前添加可学习的连续向量：

对于自注意力机制：
$$A_i = \text{Attention}(Q W_i^Q, K W_i^K \oplus P, V W_i^V \oplus P)$$

其中 $P$ 是可学习的前缀，$\oplus$ 表示拼接。

Prefix Tuning有两种形式：
1. ** Continuous Prefix**：可学习的连续向量
2. **Discrete Prompt**：离散的提示词

### 2.4 Prompt Tuning原理

Prompt Tuning将提示词转换为可学习的embedding：

```python
class PromptEmbedding(nn.Module):
    def __init__(self, num_prompts, embed_dim):
        self.prompts = nn.Embedding(num_prompts, embed_dim)
    
    def forward(self, batch_size):
        return self.prompts.weight.unsqueeze(0).expand(batch_size, -1, -1)
```

## 3. 数学���式与推导

### 3.1 LoRA的梯度分析

假设损失函数为 $\mathcal{L}$，对输入 $x$ 的前向传播：

$$h = W_0 x + B A x = (W_0 + B A) x$$

对参数 $A$ 的梯度：
$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial h} \cdot x^T \cdot B^T$$

对参数 $B$ 的梯度：
$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial h} \cdot (A x)^T$$

### 3.2 LoRA的参数效率

设模型参数量为 $N$，LoRA rank为 $r$，可训练参数比例：

$$\text{trainable ratio} = \frac{2 N r}{N} = \frac{2r}{\text{dim}} \approx \frac{2r}{d_{model}}$$

对于 $d_{model}=4096, r=8$:
$$\approx \frac{2 \times 8}{4096} = 0.39\%$$

### 3.3 Adapter的参数效率

Adapter参数量：
$$N_{adapter} = d_{model} \cdot d_{adapter} + d_{adapter} + d_{adapter} \cdot d_{model}$$

若 $d_{adapter} = \frac{d_{model}}{16}$：
$$N_{adapter} = 2 \cdot \frac{d_{model}^2}{16} = \frac{d_{model}^2}{8}$$

每个Transformer层增加 $12.5\%$ 的参数。

### 3.4 量化分析

QLoRA使用4-bit量化：

$$x_{quant} = \text{round}\left(\frac{x}{2^b} \cdot 2^b\right)$$

量化误差：
$$\epsilon = |x - x_{quant}|$$

量化后的LoRA：
$$W_{quant} = Q(W_0) + BA$$

## 4. 训练过程讲解

### 4.1 LoRA训练流程

```
Step 1: 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained('gpt2')

Step 2: 添加LoRA层
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['attn.q', 'attn.v'],
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)

Step 3: 训练
    trainer = Trainer(model, train_dataset)
    trainer.train()

Step 4: 合并权重（可选）
    merged_model = model.merge_and_unload()
```

### 4.2 PEFT配置参数

```python
# LoRA配置
lora_config = LoraConfig(
    r=8,                     # LoRA rank
    lora_alpha=16,            # LoRA缩放系数
    target_modules=['q_proj', 'v_proj'],  # 目标模块
    lora_dropout=0.1,         # Dropout
    bias='none',              # 偏置训练策略
    task_type='CAUSAL_LM'     # 任务类型
)

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=3,
    fp16=True,
    save_strategy='epoch',
    save_total_limit=3,
)
```

### 4.3 多任务训练

```python
# 共享主干，多任务独立LoRA
class MultiTaskPEFT:
    def __init__(self, base_model, tasks, lora_config):
        self.base_model = base_model
        self.task_models = {}
        
        for task in tasks:
            task_model = get_peft_model(base_model, lora_config)
            self.task_models[task] = task_model
    
    def train_task(self, task, dataset):
        """训练特定任务"""
        model = self.task_models[task]
        # 训练...
    
    def infer_task(self, task, input_ids):
        """推理特定任务"""
        model = self.task_models[task]
        return model.generate(input_ids)
```

## 5. 应用场景

### 5.1 典型应用

**指令微调**：
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('gpt2')
lora_config = LoraConfig(
    r=8,
    target_modules=['attn.q', 'attn.v', 'fc1', 'fc2']
)
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())
# Output: trainable params: 0.15% || all params: 0.08%
```

**情感分类**：
```python
from peft import PromptTuningConfig, get_peft_model

model = AutoModelForSequenceClassification.from_pretrained('gpt2')
peft_config = PromptTuningConfig(
    task_type='SEQ_CLS',
    num_virtual_tokens=8
)
model = get_peft_model(model, peft_config)
```

**命名实体识别**：
```python
from peft import IA3Config, get_peft_model

config = IA3Config(
    target_modules=['q', 'v', 'fc1', 'fc2'],
    fan_in_fan_out=True
)
model = get_peft_model(model, config)
```

### 5.2 代码实现

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 加载模型
model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 配置LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    bias='none',
    task_type=TaskType.CAUSAL_LM
)

# 创建PEFT模型
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# 训练循环
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=3e-4)

for epoch in range(3):
    peft_model.train()
    for batch in train_loader:
        inputs = tokenizer(batch['text'], return_tensors='pt').to('cuda')
        
        outputs = peft_model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 5.3 推理和导出

```python
# 推理
peft_model.eval()
output = peft_model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7
)
print(tokenizer.decode(output[0]))

# 导出合并后的模型
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained('merged_model')

# 只保存LoRA权重
peft_model.save_pretrained('lora_weights')
```

### 5.4 QLoRA实现

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    'facebook/opt-13b',
    quantization_config=bnb_config,
    device_map='auto'
)

# 添加LoRA
lora_config = LoraConfig(r=8, lora_alpha=16)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

## 6. 优缺点分析

### 6.1 各种PEFT方法的对比

| 方法 | 可训练参数 | 效果 | 推理延迟 | 显存节省 |
|------|-----------|------|---------|---------|
| Full FT | 100% | 最好 | 无 | 1x |
| LoRA | 1-3% | 接近最好 | 轻微 | 2-3x |
| Adapter | 3-5% | 好 | 中等 | 2x |
| Prefix | 0.1% | 一般 | 很小 | 5x+ |
| Prompt | 0.01% | 一般 | 无 | 最大 |

### 6.2 LoRA的优点

1. **参数效率高**：1-3%参数量即可
2. **效果接近全参数微调**：在很多任务上达到95%+的性能
3. **推理开销小**：可以合并权重
4. **可扩展性好**：可叠加多个LoRA
5. **训练稳定**：不易出现梯度问题

### 6.3 LoRA的缺点

1. **需要选择目标模块**：不恰当的选择效果差
2. **Rank敏感性**：需要调参
3. **无法改变模型结构**：只适应权重
4. **组合任务**：多任务时需要管理多个LoRA

### 6.4 使用场景

**推荐LoRA**：
- 下游任务微调
- 垂直领域适应
- 角色扮演

**推荐Adapter**：
- 需要快速切换任务
- 需要模块化

**推荐Prefix**：
- 极度资源受限
- 探针分析

## 7. 调库实现（Python + PEFT库）

### 7.1 LoRA完整示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset

# 1. 加载数据
dataset = load_dataset(' glue', 'sst2')

# 2. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    'gpt2',
    torch_dtype=torch.float32
)

# 3. 配置LoRA
peft_config = LoraConfig(
    task_type='CAUSAL_LM',
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['c_attn', 'c_proj'],
    bias='none',
)

# 4. 创建PEFT模型
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 5. 训练
training_args = TrainingArguments(
    output_dir='./train_results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=3e-4,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
)

trainer.train()

# 6. 推理
model.eval()
input_ids = tokenizer("This movie is", return_tensors='pt').to(model.device)
output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

### 7.2 多LoRA管理

```python
from peft import PeftModel, PeftConfig

class LoRAManager:
    """管理多个LoRA模型"""
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.adapters = {}
    
    def add_adapter(self, name, config):
        """添加LoRA"""
        peft_model = get_peft_model(self.base_model, config)
        self.adapters[name] = peft_model
    
    def switch_adapter(self, name):
        """切换LoRA"""
        if name in self.adapters:
            return self.adapters[name]
        raise ValueError(f"Adapter {name} not found")
    
    def merge_adapter(self, name):
        """合并LoRA到基础模型"""
        model = self.adapters[name]
        return model.merge_and_unload()

# 使用
manager = LoRAManager(base_model)
manager.add_adapter('finance', lora_config_finance)
manager.add_adapter('medical', lora_config_medical)

# 切换使用
model = manager.switch_adapter('finance')
```

### 7.3 评估工具

```python
from datasets import load_metric
import evaluate

def evaluate_peft(model, tokenizer, dataset):
    """评估PEFT模型"""
    model.eval()
    predictions = []
    references = []
    
    for example in dataset:
        inputs = tokenizer(example['text'], return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(**inputs)
        pred = tokenizer.decode(outputs[0])
        predictions.append(pred)
        references.append(example['label'])
    
    # 计算指标
    metric = load_metric('accuracy')
    return metric.compute(
        predictions=predictions,
        references=references
    )
```

## 8. 手工代码实现

### 8.1 LoRA层实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    """LoRA层的实现
    
    LoRA核心公式: h = Wx + BAx
    """
    
    def __init__(
        self,
        in_features,
        out_features,
        rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        
        # LoRA参数
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(lora_dropout)
        
        # 缩放系数
        self.scaling = lora_alpha / rank
        
        # 冻结原始权重
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features),
            requires_grad=False
        )
        nn.init.zeros_(self.weight)
    
    def forward(self, x):
        """LoRA前向传播
        
        x: (batch, seq, in_features)
        """
        # 原始输出
        original = F.linear(x, self.weight)
        
        # LoRA输出
        lora_input = self.dropout(x)
        lora_output = F.linear(
            F.linear(lora_input, self.lora_A),
            self.lora_B
        )
        
        return original + lora_output * self.scaling


class LoRAInjector:
    """LoRA注入器"""
    
    @staticmethod
    def inject_lora(model, target_modules, rank=8):
        """为模型注入LoRA"""
        for name, module in model.named_modules():
            # 检查是否是目标模块
            for target in target_modules:
                if target in name:
                    # 替换为LoRA层
                    if isinstance(module, nn.Linear):
                        new_module = LoRALayer(
                            in_features=module.in_features,
                            out_features=module.out_features,
                            rank=rank
                        )
                        parent = model.get_submodule('.'.join(name.split('.')[:-1]))
                        child_name = name.split('.')[-1]
                        setattr(parent, child_name, new_module)
        
        return model
```

### 8.2 LoRA合并

```python
def merge_lora_weights(base_weight, lora_A, lora_B, scaling):
    """合并LoRA权重
    
    W_merged = W + BA * scaling
    """
    merged = base_weight + torch.mm(lora_B, lora_A) * scaling
    return merged


def merge_and_unload(model):
    """合并所有LoRA权重并卸载"""
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A'):
            # 获取基础权重
            base_weight = module.weight.data
            
            # 合并LoRA权重
            lora_weight = torch.mm(
                module.lora_B.data,
                module.lora_A.data
            ) * module.scaling
            
            merged = base_weight + lora_weight
            
            # 替换
            module.weight.data = merged
            module.weight.requires_grad = False
            
            # 删除LoRA参数
            del module.lora_A
            del module.lora_B
    
    return model
```

### 8.3 梯度检查

```python
def check_lora_gradients():
    """验证LoRA梯度计算"""
    torch.manual_seed(42)
    
    # 创建LoRA层
    lora = LoRALayer(10, 10, rank=4)
    
    # 输入
    x = torch.randn(2, 5, 10, requires_grad=True)
    
    # 前向传播
    output = lora(x)
    loss = output.sum()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"LoRA A grad shape: {lora.lora_A.grad.shape}")
    print(f"LoRA B grad shape: {lora.lora_B.grad.shape}")
    print(f"Base weight grad: {lora.weight.grad}")

check_lora_gradients()
```

### 8.4 Adapter实现

```python
class Adapter(nn.Module):
    """Adapter层实现
    
    结构: Linear(in->hidden) -> ReLU -> Linear(hidden->out)
    """
    
    def __init__(
        self,
        in_features,
        adapter_size=64,
        dropout=0.1,
    ):
        super().__init__()
        
        self.down = nn.Linear(in_features, adapter_size)
        self.activation = nn.ReLU()
        self.up = nn.Linear(adapter_size, in_features)
        self.dropout = nn.Dropout(dropout)
        
        # 缩放
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        """前向传播
        
        x: (batch, seq, features)
        residual: (batch, seq, features)
        """
        h = self.down(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up(h)
        
        return x + self.scale * h
```

## 9. 可视化与结果理解

### 9.1 LoRA注意力可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_lora_weights():
    """可视化LoRA权重矩阵"""
    
    # 模拟LoRA权重
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # LoRA A
    A = np.random.randn(8, 64)
    im1 = axes[0].imshow(A, aspect='auto', cmap='RdBu_r')
    axes[0].set_title('LoRA A (rank×dim)')
    axes[0].set_xlabel('Input Dim')
    plt.colorbar(im1, ax=axes[0])
    
    # LoRA B
    B = np.random.randn(64, 8)
    im2 = axes[1].imshow(B, aspect='auto', cmap='RdBu_r')
    axes[1].set_title('LoRA B (dim×rank)')
    plt.colorbar(im2, ax=axes[1])
    
    # BA矩阵
    BA = np.random.randn(64, 64)
    im3 = axes[2].imshow(BA, aspect='auto', cmap='RdBu_r')
    axes[2].set_title('BA Matrix (dim×dim)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('lora_weights.png', dpi=150)
    plt.close()

visualize_lora_weights()
```

### 9.2 参数分布可视化

```python
def plot_parameter_distribution():
    """可视化PEFT参数量分布"""
    
    methods = ['Full FT', 'LoRA', 'Adapter', 'Prefix', 'Prompt']
    params_pct = [100, 2, 5, 0.5, 0.01]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, params_pct, color=['red', 'green', 'blue', 'orange', 'purple'])
    plt.ylabel('Trainable Parameters (%)')
    plt.title('Parameter Efficiency Comparison')
    plt.yscale('log')
    
    # 添加数值标签
    for bar, pct in zip(bars, params_pct):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('peft_params.png', dpi=150)
```

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 | 测量方法 |
|------|------|----------|
| 可训练参数比例 | 参数量占比 | 计算参数���量 |
| 推理延迟 | 额外计算时间 | Benchmark |
| 显存节省 | GPU显存减少 | 峰值显存 |
| 任务性能 | 下游任务表现 | Accuracy/BLEU等 |

### 10.2 性能基准

```python
def benchmark_lora():
    """LoRA性能基准测试"""
    import time
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试参数配置
    for rank in [4, 8, 16, 32]:
        lora = LoRALayer(4096, 4096, rank=rank).to(device)
        x = torch.randn(8, 128, 4096).to(device)
        
        # 计时
        start = time.time()
        for _ in range(100):
            _ = lora(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        ms = (time.time() - start) / 100 * 1000
        print(f"Rank {rank}: {ms:.3f}ms")

benchmark_lora()
```

### 10.3 任务性能对比

```python
from datasets import load_dataset

def compare_tasks():
    """对比不同任务上的性能"""
    
    tasks = ['sst2', 'mrpc', 'cola', 'qqp']
    full_ft_results = [94, 85, 60, 72]
    lora_results = [93, 84, 58, 71]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, full_ft_results, width, label='Full FT')
    plt.bar(x + width/2, lora_results, width, label='LoRA')
    plt.xticks(x, tasks)
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('task_comparison.png', dpi=150)

compare_tasks()
```

## 11. 常见问题与易错点

### 11.1 LoRA目标模块选择

**错误**：目标模块选择不当
**正确**：
- CausalLM: q_proj, v_proj
- Seq2Seq: q_proj, v_proj, fc1, fc2

### 11.2 Rank选择

**错误**：Rank过大或过小
**建议**：从8开始尝试，效果差再调整

### 11.3 量化冲突

**错误**：QLoRA与某些模型不兼容
**解决**：检查模型是否支持bnb量化

### 11.4 推理模式

**错误**：训练后未切换到推理模式
**正确**：使用`model.eval()`再推理

## 12. 学习总结

### 核心要点

1. **LoRA公式**：$h = W_0 x + B A x$
2. **参数效率**：1-3%参数量
3. **目标模块**：attention相关层
4. **实现**：可合并到原模型

### 关键优势

- 近似全参数微调的效果
- 大幅降低显存和计算需求
- 可管理多个任务

### 实现要点

```python
# 核心配置
LoraConfig(
    r=8,              # Rank
    lora_alpha=16,       # 缩放
    target_modules=['q', 'v']
)
```

## 13. 练习题与思考题

### 练习题

**Q1**: LoRA的核心原理是什么？

**答案**：LoRA通过在预训练权重旁边添加两个低秩矩阵，通过训练低秩矩阵来间接更新大模型，实现参数高效微调。

**Q2**: 为什么LoRA可以保持预训练能力？

**答案**：LoRA只训练低秩矩阵，原始预训练权重被冻结，不会发生灾难性遗忘。

**Q3**: LoRA的Rank如何选择？

**答案**：从8开始尝试，在效果和参数量间平衡。通常8, 16, 32够用。

**Q4**: LoRA和Adapter的区别？

**答案**：LoRA是额外添加低秩矩阵，Adapter是添加小型网络模块。

### 思考题

**Q1**: PEFT适合哪些场景？

**答案**：下游任务微调、垂直领域适应、个性化定制、算力受限场景。

**Q2**: 如何选择PEFT方法？

**答案**：根据任务复杂度、算力限制、效果要求选择。通用推荐LoRA。

## 14. 学习路径建议

### 基础阶段

1. 理解微调的概念
2. 学习LoRA原理
3. 实现简单LoRA

### 进阶阶段

1. 配置多种PEFT方法
2. 性能对比分析
3. 部署优化

### 实践阶段

1. 生产项目应用
2. 多任务管理
3. 模型压缩

### 参考资源

- Paper: "LoRA: Low-Rank Adaptation of Large Language Models"
- GitHub: huggingface/peft
- 文档: PEFT官方文档