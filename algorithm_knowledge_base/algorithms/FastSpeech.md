# FastSpeech 学习文档

> 非自回归语音合成，速度快、可控的 TTS 模型。

---

## 1. 算法基础认知

### 1.1 发展背景

FastSpeech 由微软亚研院于 2019 年在论文《FastSpeech: Fast, Robust and Controllable Text to Speech》中提出，是一种非自回归的语音合成模型，解决了自回归 TTS 的两大痛点：速度慢和可控性差。

### 1.2 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 非自回归 TTS |
| 速度 | 比自回归快 270 倍 |
| 可控性 | 语速、停顿、韵律 |
| 质量 | 与自回归相当 |

### 1.3 模型系列

| 模型 | 参数 |
|------|------|
| FastSpeech | 30M |
| FastSpeech 2 | 35M |

---

## 2. 核心原理

### 2.1 整体架构

```
文本 → Encoder → Duration Predictor → 长度对齐 → Decoder → 梅尔频谱
```

### 2.2 Duration Predictor

预测每个 phoneme 的持续时间：

```python
duration = DurationPredictor(encoder_output)
```

### 2.3 长度对齐

```python
# 重复 phoneme
expanded = expand_phone(phoneme, duration)
```

---

## 3. 数学公式与推导

### 3.1 Duration 预测

$$d_i = \text{Predictor}(enc_i)$$

损失函数：
$$L_{dur} = \frac{1}{N} \sum |d_i - \hat{d}_i|$$

### 3.2 Fastspeech 损失

$$L = L_{spec} + L_{dur} + L_{pitch} + L_{energy}$$

### 3.3 注意力掩码

位置编码 + 相对位置编码

---

## 4. 训练过程讲解

### 4.1 两阶段训练

1. **预训练**：使用自回归模型提取 duration
2. **微调**：联合训练

### 4.2 参数

| 参数 | 值 |
|------|-----|
| hop_length | 256 |
| win_length | 1024 |
| fft_size | 1024 |

---

## 5. 应用场景

### 5.1 典型应用

- **语音助手**：Siri、小爱同学
- **有声书**：自动朗读
- **多语言合成**：翻译配音

### 5.2 代码示例

```python
# 使用 TTS 库
from TTS.api import TTS

tts = TTS(model="fastspeech")
wav = tts.tts("Hello world!")
```

---

## 6. 调库实现

### 6.1 PyTorch 实现

```python
import torch
import torch.nn as nn

class DurationPredictor(nn.Module):
    """Duration 预测器"""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv(x)
        return out.transpose(1, 2).squeeze(-1)


class FastSpeech(nn.Module):
    """FastSpeech"""
    
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, embed_dim)
        self.duration = DurationPredictor(embed_dim)
        self.decoder = nn.Linear(embed_dim, 80)  # 梅尔频谱
        
    def forward(self, text):
        enc = self.encoder(text)
        dur = self.duration(enc)
        mel = self.decoder(enc)
        return mel


def demo():
    print("=== FastSpeech 演示 ===\n")
    model = FastSpeech(100, 256)
    text = torch.randint(0, 100, (10,))
    out = model(text)
    print(f"输出: {out.shape}")


if __name__ == "__main__":
    demo()
```

---

## 7. 优缺点分析

### 7.1 优点

1. **速度快**：270 倍加速
2. **可控**：语速、停顿
3. **稳定**：无暴露偏差

### 7.2 缺点

1. **需要 duration**：额外预测
2. **质量**：略低于自回归

### 7.3 改进

- FastSpeech 2：直接预测 pitch/energy

---

## 8. 可视化与结果理解

### 8.1 频谱可视化

```python
import librosa
import matplotlib.pyplot as plt

def visualize():
    mel = torch.rand(80, 100)
    plt.imshow(mel.numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('梅尔频谱')
    plt.savefig('fastspeech_mel.png')
```

---

## 9. 模型评估

### 9.1 MOS 分数

| 模型 | MOS |
|------|-----|
| Transformer TTS | 3.8 |
| FastSpeech | 3.7 |
| Tacotron | 3.5 |

---

## 10. 学习总结

**核心要点**：

1. **非自回归**：并行生成
2. **Duration Predictor**：时长预测
3. **可控制**：语速、停顿

**FastSpeech 核心优势**：
- 速度快 270 倍
- 可控性强

**学习建议**：

1. 理解 TTS 基础
2. 掌握 duration 预测
3. 实践语音合成

---

## 11. 练习题与思考题

### 11.1 基础练习

1. FastSpeech vs 自回归 TTS
2. Duration 预测原理

### 11.2 思考题

1. 质量改进方向

---

### 11.3 详细答案

**问题**：快速原因

**解答**：非自回归，并行生成

---

## 14. 学习路径建议

### 入门阶段

1. 语音合成基础
2. TTS 原理

### 进阶阶段

1. FastSpeech 实现
2. 实践 TTS

**推荐路线**：

```
TTS 基础 → Tacotron → FastSpeech → VALL-E
```

**FastSpeech 是实用的 TTS 模型，掌握它对语音合成很重要。**