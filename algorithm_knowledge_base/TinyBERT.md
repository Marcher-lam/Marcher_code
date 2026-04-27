# TinyBERT 学习文档

## 1. 算法基础认知

### 1.1 发展背景

TinyBERT 由华为诺亚实验室于 2019 年在论文《TinyBERT: Distilling BERT for Natural Language Understanding》中提出，采用两阶段蒸馏：通用领域预训练蒸馏 + 下游任务微调蒸馏。TinyBERT的出现是为了解决BERT模型参数量大、推理速度慢的问题，使得大型语言模型可以在资源受限的设备上运行。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 算法类型 | 知识蒸馏 |
| 教师模型 | BERT-base (12层, 768维) |
| 学生模型 | TinyBERT (4层, 312维) |
| 参数减少 | 87% (110M → 14M) |
| 性能保留 | 96% |

### 1.3 历史背景与技术演进

知识蒸馏的发展历程：
- 2015: Hinton et al. 首次提出知识蒸馏（Distillation）
- 2019: BERT的提出引发模型压缩需求
- 2019: TinyBERT提出两阶段蒸馏方法
- 2020: MobileBERT、MiniLM等后续工作

TinyBERT与DistilBERT的对比：
| 模型 | 参数 | 层数 | 维度 | GLUE性能 |
|------|------|------|------|---------|
| BERT-base | 110M | 12 | 768 | 100% |
| DistilBERT | 66M | 6 | 768 | 97% |
| TinyBERT | 14M | 4 | 312 | 96% |
| MobileBERT | 25M | 12 | 512 | 97% |

### 1.4 前置知识

学习TinyBERT需要：
1. BERT模型结构（Transformer编码器）
2. 知识蒸馏基本概念
3. 注意力机制
4. 中间层表示

---

## 2. 核心原理

### 2.1 两阶段蒸馏框架

TinyBERT采用独特的两阶段蒸馏方法：

**第一阶段：通用蒸馏（General Domain Distillation）**
- 在大规模通用语料上进行预训练蒸馏
- 使用BookCorpus + Wikipedia数据
- 学习通用的语言表示
- 帮助学生模型建立初步的语言理解能力

**第二阶段：任务蒸馏（Task-Specific Distillation）**
- 在下游任务的标注数据上进行微调蒸馏
- 使用任务特定的训练数据
- 进一步优化任务性能

### 2.2 知识迁移机制

TinyBERT蒸馏三种类型的知识：

1. **输出层知识（Prediction Distillation）**
   - 蒸馏学生模型的logits与教师模型的logits
   - 使用交叉熵损失

2. **隐藏层知识（Hidden States Distillation）**
   - 蒸馏学生模型的隐藏状态与教师模型的隐藏状态
   - 使用MSE损失

3. **注意力知识（Attention Distillation）**
   - 蒸馏学生模型的注意力矩阵与教师模型的注意力矩阵
   - 使用MSE损失

### 2.3 层级映射策略

由于教师模型和学生模型层数不同，需要建立映射关系：

| 教师层 | 学生层 | 映射方法 |
|--------|--------|----------|
| 12层 | 4层 | 等间距映射 |
| 每3层 | 1层 | f(j) = floor(j * L_S / L_T) |

**隐藏层维度映射**：
- 教师：768维 → 投影 → 学生：312维
- 使用线性变换矩阵W进行维度适配

---

## 3. 数学公式与推导

### 3.1 注意力蒸馏损失

$$L_{attn} = \frac{1}{h} \sum_{i=1}^{h} MSE(A_i^S, A_i^T)$$

其中：
- $A_i^S$ 是学生模型第i个注意力头的注意力矩阵
- $A_i^T$ 是教师模型第i个注意力头的注意力矩阵
- $h$ 是注意力头的数量

### 3.2 隐藏层蒸馏损失

$$L_{hidn} = \sum_{j=1}^{L_S} MSE(W_j^S H_j^S, H_{\phi(j)}^T)$$

其中：
- $H_j^S$ 是学生模型第j层的隐藏状态
- $H_{\phi(j)}^T$ 是教师模型对应层的隐藏状态
- $W_j^S$ 是将学生隐藏状态映射到教师空间的变换矩阵
- $\phi(j)$ 是从学生层j到教师层的映射函数

### 3.3 输出层蒸馏损失

$$L_{pred} = CE(y, S_{soft}(x; \theta_S))$$

其中：
- $S_{soft}$ 是学生模型的softmax输出
- $y$ 是真实标签
- 使用教师模型的logits作为软标签

### 3.4 总损失函数

$$L = \alpha L_{attn} + \beta L_{hidn} + \gamma L_{pred}$$

超参数设置：
- $\alpha = 1.0$：注意力蒸馏权重
- $\beta = 1.0$：隐藏层蒸馏权重
- $\gamma = 0.5$：输出蒸馏权重

### 3.5 蒸馏温度

使用温度参数T控制软标签的熵：
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

温度T通常设置为2-10，越高的温度产生越平滑的概率分布。

---

## 4. 训练过程讲解

### 4.1 通用蒸馏配置

| 配置项 | 值 |
|--------|-----|
| 批量大小 | 512 |
| 学习率 | 1e-3 |
| 训练轮数 | 10 |
| 优化器 | AdamW |
| 预热比例 | 0.1 |
| 数据集 | BookCorpus + Wikipedia |

### 4.2 任务蒸馏配置

| 配置项 | 值 |
|--------|-----|
| 批量大小 | 64 |
| 学习率 | 5e-5 |
| 训练轮数 | 3-5 |
| 序列长度 | 128 |
| 数据集 | 下游任务数据 |

### 4.3 训练流程

```
Step 1: 初始化教师模型（预训练BERT）
Step 2: 初始化学生模型（4层Transformer）
Step 3: 通用蒸馏
   for epoch in range(10):
       for batch in dataloader:
           计算三种蒸馏损失
           反向传播更新学生模型
Step 4: 任务蒸馏
   for epoch in range(3):
       for batch in task_data:
           计算任务蒸馏损失
           微调学生模型
```

### 4.4 超参数推荐

| 超参数 | 推荐范围 | 默认值 |
|--------|----------|--------|
| 学生层数 | 4, 6 | 4 |
| 学生维度 | 312, 768 | 312 |
| 蒸馏温度 | 2-10 | 4 |
| 注意力头数 | 12 | 12 |
| 蒸馏epoch | 3-10 | 5 |

---

## 5. 应用场景

### 5.1 移动端部署

TinyBERT特别适合移动端部署场景：
- 手机APP中的文本分类
- 离线NLP任务处理
- 低延迟要求的交互系统

### 5.2 嵌入式设备

在资源受限的嵌入式设备上运行：
- IoT设备的自然语言处理
- 边缘计算的模型部署
- 实时语音助手

### 5.3 实时系统

需要低延迟的实时系统：
- 在线客服系统
- 实时翻译
- 文本审核

### 5.4 API服务

构建轻量级的API服务：
- 减少计算资源消耗
- 降低服务成本
- 提高吞吐量

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 | 量化 |
|------|------|------|
| 参数极少 | 14M参数，仅BERT的13% | 110M → 14M |
| 速度快 | 3倍加速 | 推理速度3x |
| 效果好 | 保留96%性能 | GLUE 96% |
| 可定制 | 可调整层数和维度 | 灵活配置 |

### 6.2 缺点

| 缺点 | 说明 | 缓解方法 |
|------|------|----------|
| 性能损失 | 仍有4%下降 | 权衡精度和速度 |
| 训练复杂 | 两阶段蒸馏 | 使用预训练模型 |
| 维度压缩 | 信息损失 | 调整维度 |

---

## 7. 调库实现（Python + 完整代码 + 注释）

### 7.1 使用HuggingFace预训练模型

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class TinyBERTModel:
    """TinyBERT 轻量模型"""
    
    def __init__(self, model_name='huawei-noah/TinyBERT_General_4L_312D'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def encode(self, text, max_length=128):
        """将文本编码为模型输入"""
        return self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
    
    def classify(self, text):
        """文本分类"""
        inputs = self.encode(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred = logits.argmax(-1)
        return pred.item()
    
    def predict_proba(self, text):
        """预测概率分布"""
        inputs = self.encode(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        return probs[0].tolist()


def demo():
    print("=== TinyBERT 演示 ===\n")
    
    model = TinyBERTModel()
    
    texts = [
        "This movie is amazing!",
        "This product is terrible.",
        "Great quality, recommend.",
    ]
    
    for text in texts:
        label = model.classify(text)
        probs = model.predict_proba(text)
        print(f"文本: {text}")
        print(f"预测: {'正面' if label == 1 else '负面'}")
        print(f"概率: {probs}")
        print()


if __name__ == "__main__":
    demo()
```

### 7.2 使用TinyBERT进行问答

```python
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

class TinyBERTQA:
    """TinyBERT 问答模型"""
    
    def __init__(self, model_name='huawei-noah/TinyBERT_General_4L_312D'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
    
    def answer(self, question, context):
        """从给定上下文中回答问题"""
        inputs = self.tokenizer(
            question,
            context,
            return_tensors='pt',
            max_length=384,
            truncation=True,
            stride=128,
            return_overflowing_tokens=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            start_idx = start_logits.argmax()
            end_idx = end_logits.argmax()
            
            answer = self.tokenizer.decode(
                inputs['input_ids'][0][start_idx:end_idx+1]
            )
        
        return answer.strip()


if __name__ == "__main__":
    qa = TinyBERTQA()
    
    question = "What is TinyBERT?"
    context = "TinyBERT is a distilled version of BERT with 4 layers and 312 hidden dimensions."
    
    answer = qa.answer(question, context)
    print(f"问题: {question}")
    print(f"答案: {answer}")
```

---

## 8. 手工代码实现（核心算法手写 + 注释）

### 8.1 学生模型实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyBERTStudent(nn.Module):
    """TinyBERT 学生模型（4层）"""
    
    def __init__(self, vocab_size=30522, hidden_dim=312, num_layers=4, 
                 num_heads=12, intermediate_size=1200):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        
        # 4 层 Transformer 编码器
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_size=intermediate_size
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, input_ids, attention_mask=None):
        # 嵌入
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        x = self.embedding_norm(x)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.output_norm(x)
        
        # 取[CLS] token的输出
        cls_output = x[:, 0]
        
        return self.classifier(cls_output)


class TransformerLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, hidden_dim, num_heads, intermediate_size):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, attention_mask=None):
        # 自注意力 + 残差连接
        attn_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.attention_norm(x + attn_output)
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x


class DistillationLoss(nn.Module):
    """蒸馏损失"""
    
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        # 软标签蒸馏损失
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        distill_loss = distill_loss * (self.temperature ** 2)
        
        # 硬标签交叉熵损失
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # 总损失
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * ce_loss
        
        return total_loss


def demo():
    print("=== TinyBERT 手工实现演示 ===\n")
    
    model = TinyBERTStudent(
        vocab_size=30522,
        hidden_dim=312,
        num_layers=4,
        num_heads=12,
        intermediate_size=1200
    )
    
    input_ids = torch.randint(0, 30522, (2, 20))
    attention_mask = torch.ones_like(input_ids)
    
    output = model(input_ids, attention_mask)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()


if __name__ == "__main__":
    demo()
```

### 8.2 中间层蒸馏实现

```python
import torch
import torch.nn as nn

class IntermediateDistillation(nn.Module):
    """中间层蒸馏模块"""
    
    def __init__(self, teacher_model, student_model, layer_mapping):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.layer_mapping = layer_mapping
        
        # 投影矩阵
        self.projection_layers = nn.ModuleList([
            nn.Linear(student_model.hidden_dim, teacher_model.hidden_dim)
            for _ in range(student_model.num_layers)
        ])
    
    def distill_hidden(self, hidden_student, hidden_teacher, student_idx):
        """隐藏层蒸馏"""
        mapped = self.projection_layers[student_idx](hidden_student)
        return F.mse_loss(mapped, hidden_teacher)
    
    def distill_attention(self, attn_student, attn_teacher):
        """注意力蒸馏"""
        return F.mse_loss(attn_student, attn_teacher)


def create_layer_mapping(teacher_layers=12, student_layers=4):
    """创建层映射"""
    mapping = {}
    for i in range(student_layers):
        teacher_idx = int(i * teacher_layers / student_layers)
        mapping[i] = teacher_idx
    return mapping
```

---

## 9. 可视化与结果理解

### 9.1 性能对比可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_performance_comparison():
    """绘制不同轻量BERT模型的性能对比"""
    models = ['BERT', 'DistilBERT', 'TinyBERT', 'MobileBERT', 'MiniLM']
    params = [110, 66, 14, 25, 22]  # 参数量(M)
    glue_scores = [100, 97, 96, 97, 95]  # GLUE相对分数
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 参数量条形图
    ax1.bar(models, params, alpha=0.7, label='Parameters (M)', color='steelblue')
    ax1.set_ylabel('Parameters (M)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    ax2 = ax1.twinx()
    ax2.plot(models, glue_scores, 'ro-', linewidth=2, markersize=8, 
             label='GLUE Score')
    ax2.set_ylabel('GLUE Score (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Lightweight BERT Models Comparison')
    plt.tight_layout()
    plt.savefig('bert_comparison.png', dpi=150)
    plt.show()


def plot_inference_speed():
    """绘制推理速度对比"""
    batch_sizes = [1, 8, 32, 128]
    bert_times = [100, 120, 180, 350]
    tinybert_times = [30, 40, 65, 120]
    
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, bert_times, 'o-', label='BERT-base', linewidth=2)
    plt.plot(batch_sizes, tinybert_times, 's-', label='TinyBERT', linewidth=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Speed Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('inference_speed.png', dpi=150)
    plt.show()


def plot_layer_impact():
    """绘制层数对性能的影响"""
    layers = [2, 3, 4, 6]
    glue_scores = [88, 93, 96, 97]
    speed = [5, 4, 3, 2]  # 相对速度
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(layers, glue_scores, 'o-', linewidth=2, markersize=8)
    ax1.set_ylabel('GLUE Score (%)')
    ax1.set_xlabel('Number of Layers')
    
    ax2 = ax1.twinx()
    ax2.bar(layers, speed, alpha=0.3)
    ax2.set_ylabel('Relative Speed')
    
    plt.title('Impact of Layer Count on TinyBERT')
    plt.tight_layout()
    plt.savefig('layer_impact.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    plot_performance_comparison()
    plot_inference_speed()
    plot_layer_impact()
```

### 9.2 训练曲线

```
Epoch: 0, Loss: 5.234, Val Acc: 0.450
Epoch: 1, Loss: 2.156, Val Acc: 0.680
Epoch: 2, Loss: 1.234, Val Acc: 0.780
Epoch: 3, Loss: 0.876, Val Acc: 0.840
Epoch: 4, Loss: 0.654, Val Acc: 0.880
Epoch: 5, Loss: 0.512, Val Acc: 0.900
```

---

## 10. 模型评估

### 10.1 GLUE基准测试

| 任务 | BERT-base | TinyBERT 4L | 下降 |
|------|----------|--------------|------|
| MNLI | 84.5% | 81.0% | -3.5% |
| SST-2 | 93.5% | 90.5% | -3.0% |
| MRPC | 88.0% | 85.2% | -2.8% |
| QQP | 71.2% | 68.5% | -2.7% |
| QNLI | 92.5% | 89.8% | -2.7% |
| CoLA | 58.0% | 52.0% | -6.0% |

### 10.2 推理速度评估

| 配置 | BERT-base | TinyBERT | 加速比 |
|------|----------|----------|--------|
| CPU推理 | 250ms | 85ms | 2.9x |
| GPU推理 | 15ms | 5ms | 3.0x |
| 移动端 | 2000ms | 650ms | 3.1x |

### 10.3 内存占用

| 模型 | 模型大小 | 运行时内存 |
|------|----------|------------|
| BERT-base | 420MB | 1.2GB |
| TinyBERT | 55MB | 180MB |

---

## 11. 常见问题与易错点

### Q1: 两阶段蒸馏的必要性

**问题**：为什么需要两阶段蒸馏，一步蒸馏不行吗？

**原因**：直接进行任务蒸馏可能导致学生模型泛化能力不足，因为任务数据量较少。

**解决方案**：先在大规模通用语料上进行通用蒸馏，帮助学生模型建立语言理解基础。

### Q2: 层映射策略选择

**问题**：如何选择学生层到教师层的映射？

**原因**：不同的映射策略会影响蒸馏效果。

**解决方案**：
- 等间距映射适合层数差异较大的情况
- 密集映射适合层数差异较小的情况

### Q3: 中间层维度不匹配

**问题**：学生维度312，教师维度768，如何处理？

**解决方案**：使用投影矩阵将学生表示映射到教师空间
$$h_{mapped} = W \cdot h_{student}$$

### Q4: 蒸馏温度设置

**问题**：蒸馏温度如何设置？

**原因**：温度影响软标签的分布。

**解决方案**：
- 较低的T（如2）会产生更尖锐的分布
- 较高的T（如10）会产生更平滑的分布
- 经验值：T=4

### Q5: 精度损失控制

**问题**：如何控制精度损失在可接受范围内？

**原因**：压缩会带来一定的精度损失。

**解决方案**：
- 调整学生模型配置（更多层或更大维度）
- 使用更多的蒸馏数据
- 结合量化技术

---

## 12. 学习总结

### 核心要点

1. **两阶段蒸馏**：通用蒸馏 + 任务蒸馏
2. **多层蒸馏**：注意力 + 隐藏层 + 输出
3. **极小模型**：4层312维，参数减少87%
4. **速度提升**：3倍加速

### 从TinyBERT到其他算法

TinyBERT → MobileBERT(结构重参数化) → MiniLM(跨架构蒸馏) → MiniLMv2(升级版)

---

## 13. 练习题与思考题（含答案）

### 练习题1：基础概念

**问题**：TinyBERT的核心创新是什么？

**答案**：两阶段蒸馏框架

**解析**：TinyBERT的创新在于先在大规模通用语料上进行预训练蒸馏，再在下游任务上进行微调蒸馏，解决了小样本蒸馏的问题。

### 练习题2：数学计算

**问题**：给定教师隐藏状态768维，学生隐藏状态312维，计算投影后的MSE损失

**答案**：

假设：
- 教师隐藏：h_T = [0.5, 0.3, ..., 0.1] (768维)
- 学生隐藏：h_S = [0.4, 0.2, 0.1] (312维)
- 投影矩阵：W (768 × 312)

**步骤1**：投影
h_S' = W @ h_S (768维)

**步骤2**：计算MSE
L = MSE(h_S', h_T) = ||h_S' - h_T||² / 768

### 练习题3：编程实现

**问题**：实现TinyBERT的学生模型

**答案**：参考第8节的代码实现

### 思考题：改进方案

**问题**：TinyBERT在某些任务上效果下降明显，如何改进？

**答案**：

1. **增加层数**：从4层增加到6层
2. **增加维度**：从312维增加到768维（全尺度）
3. **多阶段蒸馏**：增加中间蒸馏阶段
4. **数据增强**：使用数据增强提升泛化能力

---

## 14. 学习路径建议

### 初级阶段（1-2周）

1. 理解BERT模型结构
2. 学习知识蒸馏概念
3. 掌握TinyBERT框架
4. 实现简单蒸馏

### 中级阶段（2-3周）

1. 实现中间层蒸馏
2. 理解层映射策略
3. 调参与优化
4. 评估模型性能

### 高级阶段（3-4周）

1. 改进蒸馏方法
2. 探索跨架构蒸馏
3. 结合量化压缩
4. 实际应用部署

### 推荐资源

- **论文**：Wang et al. (2020). TinyBERT. arXiv:1909.10351
- **代码**：https://github.com/huawei-noah/TinyBERT
- **课程**：Stanford CS224N

---

**文档结束**