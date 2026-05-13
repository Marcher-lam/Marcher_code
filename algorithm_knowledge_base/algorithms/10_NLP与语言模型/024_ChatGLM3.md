# ChatGLM3 学习文档

## 1. 算法基础认知

### 1.1 定义与背景

ChatGLM3是清华大学和智谱AI联合开发的第三代对话生成语言模型（Chat Generative Language Model），是ChatGLM系列模型的最新版本。ChatGLM系列从ChatGLM-130B发展而来，经过ChatGLM2-6B、ChatGLM2-12B等多个版本的迭代，在模型架构、训练技术、部署效率等方面进行了全面升级。

**ChatGLM发展历程：**

| 版本 | 发布时间 | 参数量 | 主要改进 |
|------|---------|--------|---------|
| ChatGLM-130B | 2023.3 | 130B | 首个中文ChatGLM |
| ChatGLM2-6B | 2023.6 | 6B | GLM架构、RoPE |
| ChatGLM2-12B | 2023.7 | 12B | 更大规模 |
| ChatGLM3-6B | 2023.10 | 6B | 完整ChatGLM架构 |
| ChatGLM3-12B | 2024.1 | 12B | 最新版本 |

**ChatGLM3的核心特点：**

1. **完整预训练**：基于ChatGLM-Base进行有监督微调
2. **强化学习对齐**：采用PPO/DPO进行人类偏好对齐
3. **RoPE位置编码**：使用旋转位置编码，支持长序列
4. **高效推理**：支持INT4/INT8量化部署

### 1.2 应用场景

| 场景 | 说明 |
|------|------|
| 对话系统 | 智能对话、问答系统 |
| 内容生成 | 文本创作、代码生成 |
| 多模态 | 结合视觉理解能力 |
| 企业应用 | 客服、知识库 |

---

## 2. 核心原理

### 2.1 模型架构

ChatGLM3基于Transformer Decoder-only架构，核心组件包括：

1. **Embedding层**：词嵌入 + 位置编码
2. **GLM Blocks**：双向注意力 + 因果注意力混合
3. **RoPE**：旋转位置编码
4. **GLU FFN**：门控线性单元
5. **Output Head**：语言建模头

### 2.2 GLM Layer

**传统Transformer vs GLM：**

传统Transformer使用单向/双向注意力，GLM通过Toggle操作实现混合注意力：

- Attention Mask矩阵控制信息流动
- 位置编码支持任意位置查询

**GLM Block结构：**

```
Input → [Self-Attention (双向)] → Add & Norm → [GLU-FFN] → Add & Norm → Output
```

### 2.3 训练流程

**预训练阶段：**

$$\mathcal{L}_{PT} = -\sum_t \log P(x_t | x_{<t})$$

**有监督微调（SFT）：**

$$\mathcal{L}_{SFT} = -\sum_{(q,a) \in D} \log P(a | q)$$

**对齐训练（DPO）：**

$$\mathcal{L}_{DPO} = -\log \sigma(\log P(a^+ | q) - \log P(a^- | q))$$

---

## 3. PyTorch实现

### 3.1 基础模型结构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ChatGLMConfig:
    """ChatGLM3配置"""
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 4096,
        num_layers: int = 28,
        num_heads: int = 32,
        ffn_dim: int = 13696,
        max_position_embeddings: int = 8192,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        pad_token_id: int = 2,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码"""
    
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        return cos, sin
    
    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor, 
                   cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用旋转"""
        # q, k: [batch, num_heads, seq_len, head_dim]
        q_real, q_imag = q[..., :self.dim//2], q[..., self.dim//2:]
        k_real, k_imag = k[..., :self.dim//2], k[..., self.dim//2:]
        
        q_out_real = q_real * cos - q_imag * sin
        q_out_imag = q_real * sin + q_imag * cos
        k_out_real = k_real * cos - k_imag * sin
        k_out_imag = k_real * sin + k_imag * cos
        
        return torch.cat([q_out_real, q_out_imag], dim=-1), \
               torch.cat([k_out_real, k_out_imag], dim=-1)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU前馈网络"""
    
    def __init__(self, dim: int, ffn_dim: int = None):
        super().__init__()
        
        if ffn_dim is None:
            ffn_dim = int(dim * 8 / 3)
            ffn_dim = ((ffn_dim + 255) // 256) * 256
            
        self.dim = dim
        self.ffn_dim = ffn_dim
        
        # 一次线性变换得到gate和up
        self.w_input = nn.Linear(dim, ffn_dim * 2, bias=False)
        self.w_output = nn.Linear(ffn_dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hidden = self.w_input(x)
        
        gate, up = x_hidden.chunk(2, dim=-1)
        
        hidden = F.silu(gate) * up
        
        return self.w_output(hidden)
```

### 3.2 GLM Block实现

```python
class GLMBlock(nn.Module):
    """GLM Transformer Block"""
    
    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.config = config
        
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            dropout=config.attn_dropout,
            batch_first=True,
        )
        
        # SwiGLU FFN
        self.mlp = SwiGLUFeedForward(
            config.hidden_size,
            config.ffn_dim,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """前向传播"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # 残差连接
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # 自注意力
        attn_output, present_key_value = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            attention_mask=attention_mask,
            key_padding_mask=None,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )
        
        hidden_states = residual + attn_output
        
        # FFN残差
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(hidden_states)
        
        return hidden_states, present_key_value


class ChatGLMModel(nn.Module):
    """ChatGLM模型"""
    
    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # RoPE
        self.rotary_emb = RotaryPositionEmbedding(
            config.hidden_size // config.num_heads,
            max_seq_len=config.max_position_embeddings,
        )
        
        # Transformer层
        self.layers = nn.ModuleList([
            GLMBlock(config)
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        hidden_states = self.embedding(input_ids)
        
        # RoPE
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            
        cos, sin = self.rotary_emb(seq_len, input_ids.device)
        
        # Transformer层
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
```

### 3.3 完整ChatGLM3模型

```python
class ChatGLM3ForCausalLM(nn.Module):
    """ChatGLM3完整模型（包括语言建模头）"""
    
    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.config = config
        
        # 主模型
        self.model = ChatGLMModel(config)
        
        # LM Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 绑定权重（可选）
        self.lm_head.weight = self.model.embedding.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> dict:
        """前向传播"""
        # 获取hidden states
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
            }
        else:
            return (loss, logits) if loss is not None else logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """生成"""
        self.eval()
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            outputs = self.forward(generated, return_dict=True)
            next_token_logits = outputs['logits'][:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
            
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            # 检查是否生成结束符
            if (next_token == self.config.eos_token_id).all():
                break
                
        return generated
```

### 3.4 量化版本

```python
class Int4QuantizedChatGLM(nn.Module):
    """INT4量化版本"""
    
    def __init__(self, model: ChatGLM3ForCausalLM):
        super().__init__()
        
        # 量化权重
        self.model = model
        self.quantize()
        
    def quantize(self):
        """量化"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # INT4量化
                # 动态范围量化
                max_val = param.abs().max()
                scale = max_val / 7.0
                param.data = (param.data / scale).round().clamp(-8, 7) * scale
                
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
```

---

## 4. 代码示例

### 4.1 基础使用

```python
import torch

# 创建配置
config = ChatGLMConfig(
    vocab_size=50000,
    hidden_size=4096,
    num_layers=28,
    num_heads=32,
)

# 创建模型
model = ChatGLM3ForCausalLM(config)

# 前向传播
input_ids = torch.randint(0, config.vocab_size, (2, 32))
outputs = model(input_ids, return_dict=True)

print(f"Logits shape: {outputs['logits'].shape}")
print(f"Loss: {outputs['loss'].item() if outputs['loss'] else 'None'}")
```

### 4.2 文本生成

```python
# 生成示例
input_text = "今天天气真"
input_ids = torch.tensor([[1, 200, 300, 400, 500]])  # 简化的token

generated = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
)

print(f"Generated: {generated}")
```

### 4.3 多轮对话

```python
class ChatSession:
    """对话管理"""
    
    def __init__(self, model, max_history: int = 5):
        self.model = model
        self.max_history = max_history
        self.history = []
        
    def chat(self, user_input: str) -> str:
        # 添加到历史
        self.history.append(f"User: {user_input}")
        
        # 构建上下文
        context = "\n".join(self.history[-self.max_history:])
        
        # 简化：直接生成（实际需要tokenizer）
        input_ids = torch.tensor([[1] * 32])  # 简化
        
        output = self.model.generate(
            input_ids,
            max_new_tokens=100,
        )
        
        # 简化：返回模拟输出
        response = "这是模拟回复"
        
        self.history.append(f"Assistant: {response}")
        
        return response
    
    def clear(self):
        """清空历史"""
        self.history = []
```

---

## 5. 应用场景
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充ChatGLM3的应用场景相关内容]


---

## 6. 优缺点分析
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充ChatGLM3的优缺点分析相关内容]


---

## 7. 调库实现
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充ChatGLM3的调库实现相关内容]


---

## 8. 手工代码实现
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充ChatGLM3的手工代码实现相关内容]


---

## 9. 可视化与结果理解
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充ChatGLM3的可视化与结果理解相关内容]


---

## 10. 模型评估
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充ChatGLM3的模型评估相关内容]


---

## 11. 常见问题与易错点
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充ChatGLM3的常见问题与易错点相关内容]


---

## 12. 学习总结
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充ChatGLM3的学习总结相关内容]


---

## 13. 练习题与思考题
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充ChatGLM3的练习题与思考题相关内容]


---

## 14. 学习路径建议
[请参考《DeepSeek大模型高性能核心技术与多模态融合开发》补充ChatGLM3的学习路径建议相关内容]


---


## 3. 数学公式与推导

ChatGLM3的数学基础：

### 前向传播
$$h = \sigma(W_1 x + b_1), \quad \hat{y} = W_2 h + b_2$$

### 损失函数（交叉熵）
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}$$

### 反向传播（链式法则）
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W}$$


## 4. 训练过程讲解
### 训练步骤
1. **数据加载**：Dataset + DataLoader 批处理
2. **前向传播**：数据通过网络计算输出
3. **损失计算**：对比预测与标签
4. **反向传播**：计算梯度
5. **参数更新**：优化器更新权重
6. **循环迭代**：重复直至收敛
