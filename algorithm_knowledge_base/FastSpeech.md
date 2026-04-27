# FastSpeech 学习文档

> 非自回归语音合成，速度快、可控的TTS模型

---

## 1. 算法基础认知

### 1.1 一句话定义

FastSpeech是由微软亚研院于2019年提出的非自回归语音合成（TTS）模型，解决了自回归TTS的速度慢和可控性差问题，比自回归模型快270倍！

### 1.2 直觉类比

FastSpeech就像一个"高效的配音演员"。传统的自回归TTS像是在录音棚里一句一句地录制——每一句都要等上一句录完，效率很低。FastSpeech则像是一个"提前准备好的配音"：它一次性把整个剧本都记住，然后用不同的"腔调"快速朗读出来！

更形象地说：FastSpeech就像一个"语音打印机"——输入文本立即输出语音，而不是一句一句地往外"吐"！

### 1.3 发展背景

- 2019年，微软亚研院在论文《FastSpeech: Fast, Robust and Controllable Text to Speech》中提出
- 基于Transformer的编码器-解码器架构
- 2019年语音合成比赛冠军
- FastSpeech 2于2021年推出，支持更多控制

### 1.4 核心定位

| 特性 | 说明 |
|------|------|
| 类型 | 语音合成 → 非自回归TTS |
| 输出 | 语音波形 |
| 模型 | Encoder-Decoder |
| 速度 | 比自回归快270倍 |

---

## 2. 核心原理

### 2.1 为什么需要非自回归TTS？

**自回归TTS的问题**：
- 速度慢：每秒只能生成几个音素
- 不可控：无法控制语速、停顿
- 错误累积：前面错误会传播到后面

**FastSpeech的解决方案**：
- **一次性输出**：不是逐音素生成
- **可预测长度**：预先预测音素序列长度
- **显式对齐**：使用自回归模型提取对齐信息

### 2.2 vs其他TTS对比

| 模型 | 速度 | 可控性 | 质量 |
|------|------|--------|------|
| Tacotron 2 | 慢 | 差 | 好 |
| Transformer-TTS | 中 | 差 | 好 |
| **FastSpeech** | **快270x** | **好** | **相当** |
| WaveNet | 快 | 差 | 很好 |

### 2.3 核心架构

```
文本输入
    │
    ▼
编码器（Transformer Encoder）
    │
    ▼
长度调节器（Length Regulator）
    ╔══════════════════╗
    ║ 调节音素长度       ║
    ║ 控制语速           ║
    ╚══════════════════╝
    │
    ▼
解码器（Transformer Decoder）
    │
    ▼
梅尔频谱
    │
    ▼
声码器（WaveNet/Griffin-Lim）
    │
    ▼
语音输出
```

---

## 3. 数学公式与推导

### 3.1 编码器

$$H_{enc} = \text{Encoder}(text\_embedding)$$

输入：$X \in \mathbb{R}^{B \times L \times d_{model}}$
输出：$H_{enc} \in \mathbb{R}^{B \times L \times d_{model}}$

### 3.2 长度调节器

**功能**：将音素序列长度扩展到梅尔频谱长度

$$H_{expanded} = \text{Expand}(H_{enc}, duration)$$

其中 $duration$ 是每个音素的持续时间。

**扩展公式**：
$$H_{expanded}[i] = H_{enc}[i] \times duration[i]$$

### 3.3 持续时间预测

$$duration = \text{Pronunciation}(H_{enc})$$

使用一个额外的duration预测器预测每个音素的长度。

### 3.4 解码器

$$Mel = \text{Decoder}(H_{expanded})$$

输出：$Mel \in \mathbb{R}^{B \times T \times n_{mel}}$

---

## 4. 训练过程讲解

### 4.1 训练流程

```
Step 1: 编码器提取文本特征
Step 2: 对齐器提取音素-梅尔对齐
Step 3: 训练duration预测器
Step 4: 训练完整模型
```

### 4.2 对齐器使用

```python
# 对齐器（自回归TTS提供）
self.duration_extractor = get_duration_extractor(pretrained_tts)

# ��取对齐信息
duration = self.duration_extractor.extract(text, mel_spectrogram)
```

### 4.3 损失函数

```python
# 总损失
loss = loss_mel + loss_duration + loss_postnet

# 梅尔频谱损失
loss_mel = MSE(mel_pred, mel_target)

# 持续时间损失
loss_duration = MSE(duration_pred, duration_target)
```

### 4.4 模型配置

| 参数 | 值 |
|------|-----|
| encoder_layers | 6 |
| decoder_layers | 6 |
| head | 2 |
| hidden | 256 |
| attention_dropout | 0.1 |

---

## 5. 应用场景

### 5.1 语音助手

```python
# 语音助手应用
fastspeech = load_model("fastspeech")
text = "今天天气怎么样？"
audio = fastspeech.synthesize(text)
play(audio)
```

### 5.2 有声书

```python
# 有声书制作
chapters = read_book("novel.txt")
for paragraph in chapters:
    audio = fastspeech.synthesize(paragraph)
    save_audio(f"chapter_{i}.wav", audio)
```

### 5.3 可控语音生成

```python
# 控制语速
fastspeech.synthesize(text, speed=1.5)  # 1.5倍速

# 控制停顿
fastspeech.synthesize(text, pause_durations=[0.2, 0.5, 0.3])

# 控制情感
fastspeech.synthesize(text, emotion="happy")
```

### 5.4 对比其他TTS

| 场景 | 方法 |
|------|------|
| 实时语音 | FastSpeech |
| 高质量配音 | Tacotron 2 |
| 多语言 | MXT-FastSpeech |
| 云端服务 | FastSpeech |

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| **速度快** | 比自回归快270倍 |
| **可控制** | 语速、停顿、韵律 |
| **质量好** | 与自回归相当 |
| **稳定** | 无错误累积 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 需对齐器 | 需要额外TTS提供对齐 |
| 无声码器 | 需单独训练声码器 |
| 单语言 | 需单独训练多语言 |

### 6.3 注意事项

- 需要预训练的自回归TTS提取duration
- 声码器质量影响最终效果
- 中英文模型分开训练

---

## 7. 调库实现（Python）

### 7.1 安装

```bash
pip install fastspeech
```

### 7.2 基本用法

```python
import numpy as np
import soundfile as sf

# 加载模型（需额外下载）
import sys
sys.path.append("FastSpeech/")

from model import FastSpeech
from synthesizer import Synthesizer

# 加载
model = FastSpeech()
model.load("checkpoint.pth")

# 合成
text = "Hello, this is a test."
mel = model(text)

# 转换为音频
vocoder = load_vocoder()
audio = vocoder.inverse(mel)

# 保存
sf.write("output.wav", audio, 22050)
```

### 7.3 完整示例

```python
import torch

class FastSpeechDemo:
    def __init__(self, checkpoint_path):
        # 加载模型
        self.model = torch.load(checkpoint_path)
        self.model.eval()
    
    def synthesize(self, text, speed=1.0, pitch=0):
        # 编码文本
        text_encoded = self.text_to_ids(text)
        
        # 提取特征
        mel = self.model.predict(text_encoded)
        
        # 调节速度（通过调整duration）
        if speed != 1.0:
            mel = self.adjust_speed(mel, speed)
        
        # 调节音调
        if pitch != 0:
            mel = self.adjust_pitch(mel, pitch)
        
        # 声码
        audio = self.vocode(mel)
        
        return audio
    
    def adjust_speed(self, mel, speed):
        # 调整梅尔频谱长度
        if speed > 1.0:
            # 加速：减少帧
            target_len = int(len(mel) / speed)
            indices = np.linspace(0, len(mel)-1, target_len)
            mel = mel[indices]
        else:
            # 减速：重复帧
            target_len = int(len(mel) * speed)
            indices = np.arange(len(mel)).repeat(int(1/speed))[:target_len]
            mel = mel[indices]
        
        return torch.FloatTensor(mel)
    
    def adjust_pitch(self, mel, pitch):
        # 频移（简化）
        return mel * (2 ** (pitch / 12))


# 使用
demo = FastSpeechDemo("fastspeech.pt")
audio = demo.synthesize("Hello world", speed=1.2)
```

---

## 8. 手工代码实现（理解原理）

### 8.1 简化FastSpeech

```python
import torch
import torch.nn as nn
import numpy as np

class LengthRegulator(nn.Module):
    """长度调节器"""
    def __init__(self):
        super().__init__()
    
    def forward(self, encoder_output, duration):
        """
        encoder_output: [B, T, D]
        duration: [B, T]
        """
        # 扩展
        max_len = int(duration.sum().max().item())
        outputs = []
        
        for b in range(encoder_output.size(0)):
            expanded = []
            for t, d in enumerate(duration[b]):
                expanded.append(encoder_output[b, t].repeat(int(d.item()), 1))
            
            # 填充到最大长度
            expanded_cat = torch.cat([e.squeeze(0) for e in expanded])
            if expanded_cat.size(0) < max_len:
                padding = torch.zeros(max_len - expanded_cat.size(0), expanded_cat.size(1))
                expanded_cat = torch.cat([expanded_cat, padding])
            
            outputs.append(expanded_cat)
        
        return torch.stack(outputs)


class FastSpeech(nn.Module):
    """简化版FastSpeech"""
    def __init__(self, vocab_size=300, encoder_dim=256, decoder_dim=256, n_mels=80):
        super().__init__()
        
        # Encoder
        self.encoder_embedding = nn.Embedding(vocab_size, encoder_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(encoder_dim, 8),
            num_layers=6
        )
        
        # Length Regulator
        self.length_regulator = LengthRegulator()
        self.duration_predictor = nn.Linear(encoder_dim, 1)
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(decoder_dim, 8),
            num_layers=6
        )
        
        # Output
        self.mel_projection = nn.Linear(decoder_dim, n_mels)
    
    def forward(self, text, duration=None):
        # Embedding
        x = self.encoder_embedding(text)
        
        # Encode
        encoder_output = self.encoder(x)
        
        # Predict duration (if not provided)
        if duration is None:
            duration = torch.abs(self.duration_predictor(encoder_output).squeeze(-1))
        
        # Length regulation
        expanded = self.length_regulator(encoder_output, duration)
        
        # Decode
        decoder_output = self.decoder(expanded)
        
        # Output mel
        mel = self.mel_projection(decoder_output)
        
        return mel, duration


# 测试
if __name__ == "__main__":
    model = FastSpeech()
    
    # 输入
    text = torch.randint(0, 300, (1, 10))
    duration = torch.randint(1, 5, (1, 10))
    
    # 前向
    mel, duration_pred = model(text, duration)
    
    print(f"输入形状: {text.shape}")
    print(f"持续时间: {duration.shape}")
    print(f"输出梅尔: {mel.shape}")
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params/1e6:.1f}M")
```

---

## 9. 可视化与结果理解

### 9.1 梅尔频谱可视化

```python
import matplotlib.pyplot as plt

def visualize_mel(mel_spectrogram, save_path="mel.png"):
    """可视化梅尔频谱"""
    plt.figure(figsize=(12, 6))
    plt.imshow(mel_spectrogram.T, aspect='auto', origin='lower')
    plt.title("Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel")
    plt.colorbar()
    plt.savefig(save_path, dpi=100)
    plt.show()


# 使用
mel = np.random.randn(80, 200)
visualize_mel(mel)
```

### 9.2 持续时间可视化

```python
import matplotlib.pyplot as plt

def visualize_duration(duration, phonemes, save_path="duration.png"):
    """可视化音素持续时间"""
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(duration)), duration)
    plt.title("Phoneme Duration")
    plt.xlabel("Phoneme")
    plt.ylabel("Duration (frames)")
    plt.savefig(save_path, dpi=100)
    plt.show()


# 使用
duration = [2, 3, 1, 4, 2, 3]
phonemes = ["HH", "EH", "L", "OW", "W", "ER"]
visualize_duration(duration, phonemes)
```

---

## 10. 模型评估

### 10.1 评估指标

| 指标 | 说明 |
|------|------|
| MOS | 平均意见分 |
| 语速 | 字符/秒 |
| RMSE | 持续时间误差 |
| FFE | 快速傅里叶误差 |

### 10.2 对比

| 模型 | MOS | 速度/秒 | 参数量 |
|------|-----|----------|--------|
| Tacotron 2 | 4.2 | 1.5x | 30M |
| Transformer | 4.0 | 2.0x | 35M |
| **FastSpeech** | **4.1** | **1x** | **35M** |

---

## 11. 常见问题与易错点

### Q1: 如何获取duration？

**答案**：使用预训练的自回归TTS模型提取。

### Q2: 为什么需要声码器？

**答案**：FastSpeech只输出梅尔频谱，需要声码器转为音频。

### Q3: 语速控制原理？

**答案**：通过线性调整duration来实现。

### Q4: 支持哪些语言？

**答案**：需要单独训练多语言模型。

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 架构 | Transformer编码器-解码器 |
| 核心 | 长度调节器 |
| 优势 | 速度快、可控性强 |
| 应用 | 语音合成 |

### 12.2 公式汇总

编码器：
$$H_{enc} = \text{Encoder}(X)$$

持续时间：
$$D = \text{MLP}(H_{enc})$$

长度调节：
$$H_{expanded} = \text{Expand}(H_{enc}, D)$$

解码器：
$$Mel = \text{Decoder}(H_{expanded})$$

---

## 13. 练习题

### 13.1 选择题

1. FastSpeech的核心优势是：
   - A) 质量最高
   - B) 速度快+可控制
   - C) 最易训练

2. 长度调节器的作用：
   - A) 提取特征
   - B) 调节语速
   - C) 生成波形

### 13.2 简答题

1. 解释FastSpeech如何实现语速控制。
2. 比较FastSpeech和自回归TTS的区别。

### 13.3 编程题

1. 实现简化版FastSpeech。
2. 实现语速调节功能。

---

## 14. 学习路径建议

### 14.1 进阶路径

```
TTS基础
    ↓
Transformer-TTS
    ↓
FastSpeech原理
    ↓
实战部署
```

### 14.2 相关算法

| 算法 | 关系 |
|------|------|
| Tacotron | 自回归版 |
| Transformer-TTS | 对手 |
| FastSpeech 2 | 升级版 |

### 14.3 扩展阅读

- Ren et al. (2019). FastSpeech: Fast, Robust and Controllable Text to Speech

---

## 附录

### 参考

1. Ren et al. (2019). FastSpeech: Fast, Robust and Controllable Text to Speech
2. https://github.com/ming024/FastSpeech2

---

**文档结束**