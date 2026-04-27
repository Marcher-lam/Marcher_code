# Teacher Forcing 与 Beam Search 学习文档

> 序列生成训练与推理的经典技术。

## 1. 算法基础认知

### 一句话定义

Teacher Forcing是一种在训练时使用真实标签作为解码器输入的技巧，Beam Search是一种在推理时通过保留多条候选路径来提升生成质量的搜索算法。

### 直觉类比

- Teacher Forcing：就像老师教学生写作文时，学生每写一个词，老师就告诉他下一个词应该是什么，而不是让学生自己猜。这样学生学习更快。
- Beam Search：就像同时让几个学生各自写一段话，最后选写得最好的那一个。不是只盯着最好的一个，而是保留几个可能好的。

### 历史背景

- Teacher Forcing：1989年由Williams和Zipser提出
- Beam Search：广泛应用于统计机器翻译和语音识别

### 算法定位

都属于**序列生成技术**，分别用于训练和推理阶段。

---

## 2. 核心原理

### Teacher Forcing

训练时，每个时刻的输入使用上一时刻的真实标签，而不是模型预测：
- 优点：训练快、收敛稳定
- 缺点：推理时可能暴露问题（Exposure Bias）

### Beam Search

保留top-k条最优路径：
1. 每步保留概率最高的k个token
2. 继续扩展这k个路径
3. 选最终概率最高的路径

---

## 3. 调库实现

```python
import torch
import torch.nn.functional as F
import numpy as np

def beam_search_decode(model, encoder_output, beam_size=3, max_len=20):
    """Beam Search解码实现"""
    # 假设model是Seq2Seq模型
    device = encoder_output.device
    
    # 初始化
    beam_scores = torch.zeros(1, beam_size, device=device)
    beam_scores[:, 1:] = -1e9  # 第一个token固定
    
    # 假设decoder从encoder_output开始
    decoder_input = torch.full((1, 1), 2, dtype=torch.long, device=device)  # [PAD]
    
    # 简化的Beam Search（实际需要更复杂实现）
    finished = [False] * beam_size
    results = [[] for _ in range(beam_size)]
    
    for step in range(max_len):
        # 每次生成一个token的logits
        logits = model.decode_step(decoder_input, encoder_output)  # 简化
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取top-k
        log_probs, indices = log_probs.topk(beam_size, dim=-1)
        
        # 更新beam
        if step == 0:
            beam_scores = log_probs[0]
        else:
            beam_scores = beam_scores.unsqueeze(-1) + log_probs
            beam_scores = beam_scores.view(1, -1)
            beam_scores, top_idx = beam_scores.topk(beam_size, dim=-1)
        
        # 检查结束
        for i in range(beam_size):
            if indices[0, top_idx[0, i].item()].item() == 3:  # [EOS]
                finished[i] = True
        
        # 继续未完成的
        if all(finished):
            break
    
    return results

# Teacher Forcing示例
class TeacherForcingTrainer:
    """Teacher Forcing训练器"""
    def __init__(self, model):
        self.model = model
        
    def train_step(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: 源序列
        tgt: 目标序列 [B, T]
        teacher_forcing_ratio: 使用真实标签的概率
        """
        outputs = []
        dec_input = tgt[:, 0]  # [B], 起始符
        
        for t in range(1, tgt.size(1)):
            # 解码一步
            output = self.model(dec_input, src)  # [B, vocab_size]
            outputs.append(output)
            
            # 决定是否使用teacher forcing
            if np.random.random() < teacher_forcing_ratio:
                dec_input = tgt[:, t]  # 使用真实标签
            else:
                dec_input = output.argmax(-1)  # 使用预测
            
        return torch.stack(outputs, dim=1)

# 测试
if __name__ == "__main__":
    print("Teacher Forcing 和 Beam Search 实现已生成")
```

---

## 4. 优缺点

### Teacher Forcing

| 优点 | 缺点 |
|------|------|
| 训练稳定 | Exposure Bias |
| 收敛快 | 推理时分布偏移 |
| 梯度估计准确 | |

### Beam Search

| 优点 | 缺点 |
|------|------|
| 生成质量高 | 计算量大 |
| 避免贪婪解码 | 内存占用大 |
| 保留多样性 | 可能仍非最优 |

---

## 5. 学习路径

- 前置：RNN、Transformer、Seq2Seq
- 平行：Greedy Search、采样方法
- 进阶：Length Penalty、Coverage Penalty