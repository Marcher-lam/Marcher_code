# Tacotron 端到端语音合成学习文档

> 谷歌开发的端到端神经文本转语音系统，直接从字符序列生成语音波形

---

## 1. 算法基础认知

**一句话定义**：Tacotron是谷歌开发的端到端TTS（Text-to-Speech）系统，将文本字符序列直接转换为梅尔频谱图，后接声码器生成语音波形，无需传统TTS的复杂语言学特征工程。

**直觉类比**：Tacotron就像一个"声音翻译器"。想象婴儿学习说话——他们听到大量的语音后，自然而然学会了把文字转换成声音。Tacotron做的事情类似：给它大量"文本-语音"配对数据，它就能学会从文字直接生成语音，而不需要人工设计复杂的语言学规则。

**历史背景**：
- 2017年，谷歌Wang等人在论文"Tacotron: A Fully End-to-End Text-to-Speech Inference System"中首次提出
- 2017年底，谷歌推出Tacotron 2，使用Transformer编码器/解码器和WaveNet声码器，效果大幅提升
- 后续发展出Tacotron 3、FastTacotron等变体

**核心定位**：Tacotron标志着TTS从"拼接合成"和"参数合成"转向"端到端神经合成"的开端。

---

## 2. 核心原理

### 2.1 为什么需要端到端TTS？

传统TTS系统流程：
```
文本 → 语言学分析 → 韵律预测 → 声学模型 → 声码器 → 波形
    ↓             ↓          ↓          ↓
  分词           重音       基频       合成
  词性           节奏       时长       滤波
  语法           语调       共振峰     ...
```

这种流水线问题：
- 每个模块需要单独训练，错误会级联传播
- 需要大量语言学专业知识设计特征
- 模块间接口复杂，难以联合优化
- 无法捕捉跨模块的长距离依赖

Tacotron的创新：用一个神经网络直接完成"文本→语音"的映射。

### 2.2 序列到序列架构

Tacotron本质是一个**序列到序列（Seq2Seq）模型**，核心组件：

**（1）编码器（Encoder）**
- 输入：字符序列，如 "h e l l o ⎵" （⎵表示空格）
- 每个字符通过嵌入层转换为向量
- 使用CBHG模块（卷积+ highway + 双向GRU）提取文本特征

```python
# 编码器伪代码
char_ids = char_to_id("hello")  # [8, 5, 12, 12, 15]
char_embeddings = embedding(char_ids)  # [batch, seq_len, embed_dim]
encoder_output = CBHG(char_embeddings)  # [batch, seq_len, hidden_dim]
```

**（2）注意力机制（Attention）**
- 在解码每一步时，决定"当前应该关注文本的哪些部分"
- 使用位置敏感的注意力（Location-Sensitive Attention）
- 不仅看内容，还看之前的对齐历史

```
解码步1: 关注 "h" → 生成频谱帧1
解码步2: 关注 "he" → 生成频谱帧2
解码步3: 关注 "hell" → 生成频谱帧3
...
```

**（3）解码器（Decoder）**
- 自回归生成梅尔频谱图
- 每步预测多个帧（teacher forcing during training）
- 预测→后处理→stop token判断

**（4）后处理（Post-processing）**
- 将解码器输出转换为最终的梅尔频谱
- 学习预测.stop标记

### 2.3 声码器角色

Tacotron输出的是**梅尔频谱图**，不是波形。需要声码器将频谱转为波形：

| 声码器 | 特点 | 质量 |
|--------|------|------|
| Griffin-Lim | 快速但有 artifacts | 低 |
| WaveNet | 高质量但慢 | 很高 |
| WaveRNN | 实时可达到 | 高 |
| HiFi-GAN | 快速且高质量 | 高 |

---

## 3. 数学公式与推导

### 3.1 注意力分数计算

Tacotron使用**内容无关位置注意力**（Location-Sensitive Attention）：

**基本注意力**：
$$score(s_t, h_j) = v^T \tanh(W s_t + U h_j)$$

**加入位置信息**：
$$e_{t,j} = w^T \tanh(W_f f_{t,j} + V_h h_j)$$

其中：
- $s_t$：解码器第t步的状态
- $h_j$：编码器第j个隐藏状态
- $f_{t,j}$：累积注意力掩码（之前对齐的权重）

### 3.2 注意力权重

$$\alpha_{t,j} = \text{softmax}_j(e_{t,j})$$

### 3.3 上下文向量

$$c_t = \sum_j \alpha_{t,j} h_j$$

### 3.4 解码器输出

$$y_t = \text{Decoder}(s_{t-1}, c_t)$$

### 3.5 损失函数

训练时最小化：
$$\mathcal{L} = \sum_t \| \text{Mel}_{t}^{pred} - \text{Mel}_{t}^{target} \|_1 + \lambda \cdot \| \text{stop}_t^{pred} - \text{stop}_t^{target} \|^2$$

---

## 4. 训练过程讲解

### 4.1 训练流程

```
                 ┌─ 字符序列 ──┐
                 │  "hello"   │
                 └─────┬─────┘
                       ▼
            ┌───── 嵌入层 ─────┐
            │ [batch, seq]   │
            └──────┬────────┘
                   ▼
            ┌───── CBHG ──────┐
            │ [batch, 512]   │
            └──────┬────────┘
                   ▼
    ┌────────────┬───────────┐
    │           ▼           │
    │      ┌───────┐         │
    │      │注意力│─────────┼──→ 预测位置
    │      └───────┘         │
    │           │           │
    │           ▼           │
    │    ┌────────────┐     │
    │    │ 解码器GRU  │     │
    │    └────────────┘     │
    │           │           │
    │           ▼           │
    │    ┌────────────┐     │
    │    │ 线性变换  │     │
    │    └────────────┘     │
    │           │           │
    │    ┌─────┴─────┐      │
    │    ▼           ▼    │
    │  频谱       stop    │
    │  [batch, 80] 标量  │
    └────────────────────┘
```

### 4.2 Teacher Forcing

训练时使用teacher forcing：
- 正确的目标频谱作为下一时刻输入
- 同时预测stop token（当前帧是否为最后一个）

### 4.3 自回归推理

推理时：
- 上一帧预测作为下一帧输入
- stop token > 0.5 时停止

### 4.4 超参数

| 参数 | Tacotron 1 | Tacotron 2 |
|------|------------|-------------|
| 字符嵌入 | 256 | 512 |
| 编码器隐层 | 512 | 512 |
| 解码器隐层 | 512 | 512 |
| 注意力 | Location | Location |
| 输出维度 | 80 (mel) | 80 (mel) |
| 批大小 | 32 | 64 |

---

## 5. 应用场景

### 5.1 语音助手

Siri、Google Assistant等语音助手的语音生成：
- 查天气、设闹钟、导航等场景
- 要求：响应快、流畅、自然

### 5.2 有声书

长文本朗读：
- 需要保持一致的音色和韵律
- 段落间自然过渡

### 5.3 导航播报

TTS在车载导航中的应用：
- "前方500米左转"
- 清晰、准确、节奏稳定

### 5.4 语音无障碍

帮助视障人士"阅读"文字：
- 电子书转语音
- 界面朗读

### 5.5 游戏与虚拟角色

游戏NPC对话：
- 实时生成对话
- 多种音色/语言

---

## 6. 优缺点分析

### 6.1 优点

| 优点 | 说明 |
|------|------|
| 端到端 | 无需复杂的语言学特征工程 |
| 联合优化 | 所有模块一起训练，全局最优 |
| 灵活性 | 可处理任意文本输入 |
| 韵律自然 | 比传统参数TTS更自然 |

### 6.2 缺点

| 缺点 | 说明 |
|------|------|
| 计算重 | 推理慢，需要GPU |
| 训练难 | 需要大量"文本-语音"配对数据 |
| 韵律控制 | 难以精确控制语速/语调 |
| 重复 | 会出现词语重复或漏读 |

### 6.3 改进方向

- **FastTacotron**：加速推理
- **Tacotron + GAN**：提高音质
- **韵律控制**：加入额外输入控制语速/语调
- **多说话人**： Speaker embedding

---

## 7. 调库实现

### 7.1 使用TTS库（推荐）

```python
# 安装
# pip install TTS

from TTS.api import TTS

# Tacotron 2 (需要WaveNet声码器)
tts = TTS("tacotron2", gpu=False)

# 生成语音
tts.tts(
    text="Hello world, this is a test of text to speech.",
    file_path="output.wav"
)
```

### 7.2 使用TensorFlow TFS

```python
# 安装
# pip install tensorflow-tts

import tensorflow as tf
from tensorflow_tts.inference import tflite_inference

# 加载Tacotron 2模型
tacotron2 = tflite_inference.TFLiteTacotron2(
    model_path="tacotron2.tflite"
)
# 加载声码器
mb_melgan = tflite_inference.TFLiteMBMelGAN(
    model_path="mb_melgan.tflite"
)

# 推理
text = "Hello world"
mel = tacotron2.inference(text)
wav = mb_melgan.inference(mel)

# 保存
import soundfile as sf
sf.write("output.wav", wav, 24000)
```

### 7.3 完整pipeline示例

```python
import numpy as np
from TTS.api import TTS

class TTSPipeline:
    """完整的TTS pipeline"""
    
    def __init__(self, model_name="tacotron2"):
        self.tts = TTS(model_name, gpu=False)
        
    def synthesize(self, text, output_path="output.wav"):
        """
        合成语音
        
        Args:
            text: 输入文本
            output_path: 输出音频路径
        """
        # 生成
        self.tts.tts(text, file_path=output_path)
        
        print(f"Generated: {output_path}")
        return output_path
    
    def synthesize_batch(self, texts, output_dir="."):
        """批量合成"""
        for i, text in enumerate(texts):
            path = f"{output_dir}/sentence_{i}.wav"
            self.synthesize(text, path)

# 使用
if __name__ == "__main__":
    pipeline = TTSPipeline()
    
    # 单句
    pipeline.synthesize("Hello, this is a test.")
    
    # 批量
    texts = [
        "First sentence.",
        "Second sentence.",
        "Third sentence."
    ]
    # pipeline.synthesize_batch(texts)
```

---

## 8. 手工代码实现

### 8.1 简化版Seq2Seq TTS

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    """位置敏感的注意力机制"""
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim=128):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        # 投影
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, encoder_output, decoder_state, mask=None):
        """
        Args:
            encoder_output: [batch, encoder_len, encoder_dim]
            decoder_state: [batch, decoder_dim]
            mask: [batch, encoder_len]
        Returns:
            context: [batch, encoder_dim]
            attention_weights: [batch, encoder_len]
        """
        # 投影
        encoder_proj = self.encoder_proj(encoder_output)  # [B, L, A]
        decoder_proj = self.decoder_proj(decoder_state)  # [B, A]
        
        # 计算分数
        scores = encoder_proj + decoder_proj.unsqueeze(1)  # [B, L, A]
        scores = torch.tanh(scores)
        scores = self.v(scores).squeeze(-1)  # [B, L]
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 上下文
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_output)
        context = context.squeeze(1)  # [B, encoder_dim]
        
        return context, attention_weights


class CBHG(nn.Module):
    """CBHG模块：卷积+Highway+双向GRU"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 卷积 bank
        self.conv1d = nn.ModuleList([
            nn.Conv1d(input_dim, 128, kernel_size=k, padding=k//2)
            for k in range(1, 5)
        ])
        
        # 池化
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        # 投影
        self.proj = nn.Conv1d(128*4, input_dim, kernel_size=3)
        
        # Highway
        self.highway = nn.ModuleList([
            nn.Linear(input_dim, input_dim * 4)
            for _ in range(4)
        ])
        
        # 双向GRU
        self.gru = nn.GRU(
            input_dim, output_dim // 2,
            bidirectional=True,
            batch_first=True
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            output: [batch, seq_len, output_dim]
        """
        # Conv -> [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 多个卷积核
        conv_outputs = []
        for conv in self.conv1d:
            out = F.relu(conv(x))  # [batch, 128, seq_len]
            conv_outputs.append(out)
        
        # 拼接 -> [batch, 512, seq_len]
        conv_output = torch.cat(conv_outputs, dim=1)
        
        # 残差连接
        conv_output = self.proj(conv_output)
        x = x + conv_output
        
        # HighWay
        x = x.transpose(1, 2)  # [batch, seq_len, input_dim]
        for hw in self.highway:
            h = hw(x)
            h = torch.split(h, h.size(-1) // 4, dim=-1)
            h = F.relu(h[0])
            x = h[1] * (1 - torch.sigmoid(h[2])) + x * torch.sigmoid(h[2])
        
        # BiGRU
        output, _ = self.gru(x)
        
        return output


class Tacotron(nn.Module):
    """简化版Tacotron模型"""
    
    def __init__(self, vocab_size=50, embed_dim=256, encoder_dim=512,
                 decoder_dim=512, n_mels=80):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_mels = n_mels
        
        # 字符嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 编码器
        self.encoder = CBHG(embed_dim, encoder_dim)
        
        # 注意力
        self.attention = Attention(encoder_dim, decoder_dim)
        
        # 解码器
        self.decoder_gru = nn.GRU(encoder_dim + n_mels, decoder_dim,
                                  batch_first=True)
        
        # 输出层
        self.mel_linear = nn.Linear(decoder_dim, n_mels)
        self.stop_linear = nn.Linear(decoder_dim, 1)
        
    def forward(self, chars, max_decoder_steps=100):
        """
        Args:
            chars: [batch, seq_len]
            max_decoder_steps: 最大解码步数
        Returns:
            mel_outputs: [batch, steps, n_mels]
            stop_outputs: [batch, steps]
        """
        batch_size = chars.size(0)
        
        # 编码
        embedded = self.embedding(chars)  # [B, L, E]
        encoder_output = self.encoder(embedded)  # [B, L, D]
        
        # 初始化解码器状态
        decoder_state = torch.zeros(batch_size, 512, device=chars.device)
        
        # 初始化输入（空白帧）
        decoder_input = torch.zeros(batch_size, self.n_mels, device=chars.device)
        
        # 解码
        mel_outputs = []
        stop_outputs = []
        
        for t in range(max_decoder_steps):
            # 注意力
            context, attention_weights = self.attention(
                encoder_output, decoder_state
            )
            
            # 解码器
            decoder_input = torch.cat([context, decoder_input], dim=-1)
            decoder_output, decoder_state = self.decoder_gru(
                decoder_input.unsqueeze(1), decoder_state.unsqueeze(0)
            )
            decoder_output = decoder_output.squeeze(1)
            
            # 输出
            mel = self.mel_linear(decoder_output)
            stop = torch.sigmoid(self.stop_linear(decoder_output))
            
            mel_outputs.append(mel)
            stop_outputs.append(stop)
            
            # 下一帧输入
            decoder_input = mel
            
        # 堆叠
        mel_outputs = torch.stack(mel_outputs, dim=1)
        stop_outputs = torch.stack(stop_outputs, dim=1)
        
        return mel_outputs, stop_outputs
    
    def inference(self, text):
        """推理接口"""
        # 转换为 tensor
        char_ids = torch.tensor([list(text.encode())], dtype=torch.long)
        
        # 裁剪到 vocab
        char_ids = char_ids % self.vocab_size
        
        with torch.no_grad():
            mel, stop = self.forward(char_ids)
            
        return mel, stop


# 训练示例
def train_tacotron():
    """训练示例"""
    
    # 模型
    model = Tacotron(
        vocab_size=50,
        embed_dim=256,
        encoder_dim=512,
        decoder_dim=512,
        n_mels=80
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟数据
    batch_size = 4
    seq_len = 20
    n_mels = 80
    
    for epoch in range(10):
        # 输入
        chars = torch.randint(0, 50, (batch_size, seq_len))
        target_mels = torch.randn(batch_size, 100, n_mels)
        
        # 前向
        pred_mels, pred_stops = model(chars)
        
        # 损失
        loss_mel = F.l1_loss(pred_mels[:, :target_mels.size(1)], target_mels)
        
        optimizer.zero_grad()
        loss_mel.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss_mel.item():.4f}")


if __name__ == "__main__":
    # 测试模型
    model = Tacotron(
        vocab_size=128,
        embed_dim=256,
        encoder_dim=512,
        decoder_dim=512,
        n_mels=80
    )
    
    # 随机输入
    chars = torch.randint(0, 128, (2, 10))
    
    # 前向
    mel_outputs, stop_outputs = model(chars)
    
    print(f"Mel outputs shape: {mel_outputs.shape}")
    print(f"Stop outputs shape: {stop_outputs.shape}")
```

### 8.2 核心逻辑解析

关键组件说明：

| 组件 | 作用 | 实现 |
|------|------|------|
| 嵌入 | 字符→向量 | nn.Embedding |
| CBHG | 提取文本特征 | Conv1D + GRU |
| 注意力 | 对齐文本和频谱 | 位置敏感注意力 |
| 解码器 | 逐帧生成频谱 | Autoregressive GRU |
| 后处理 | 调整输出 | 线性层 |

---

## 9. 可视化与结果理解

### 9.1 梅尔频谱可视化

```python
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

def plot_mel_spectrogram(mel, sr=22050, hop_length=256, figsize=(10, 4)):
    """
    可视化梅尔频谱
    
    Args:
        mel: 梅尔频谱 [n_mels, time]
        sr: 采样率
        hop_length: 跳帧长度
    """
    plt.figure(figsize=figsize)
    librosa.display.specshow(
        mel, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()


# 示例
if __name__ == "__main__":
    # 生成或加载梅尔频谱
    # 这里用随机数据示例
    n_mels = 80
    time_steps = 100
    mel = np.random.randn(n_mels, time_steps)
    
    plot_mel_spectrogram(mel)
```

### 9.2 注意力对齐可视化

```python
def plot_attention(attention_weights, figsize=(10, 6)):
    """
    可视化注意力对齐
    
    Args:
        attention_weights: [decoder_steps, encoder_len]
    """
    plt.figure(figsize=figsize)
    plt.imshow(attention_weights, aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel('Encoder Position')
    plt.ylabel('Decoder Position')
    plt.title('Attention Alignment')
    plt.tight_layout()
    plt.show()
```

### 9.3 训练曲线

```python
def plot_training_curve(losses):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.grid(True)
    plt.show()
```

---

## 10. 模型评估

### 10.1 主观评估

| 指标 | 说明 | 评测方法 |
|------|------|----------|
| 自然度 | 听起来像人声 | MOS测试 |
| 清晰度 | 发音准确清晰 | Mean Opinion Score |
| 韵律 | 语调节奏自然 | CMOS对比 |

### 10.2 客观评估

| 指标 | 说明 | 计算 |
|------|------|------|
| MCD | 梅尔倒谱距离 | dtw(mel_pred, mel_gt) |
| F0 RMSE | 基频均方根误差 | rmse(f0_pred, f0_gt) |
| 字错误率 | ASR回译错误 | wer(asr(pred), text) |

### 10.3 计算MCD

```python
def calculate_mcd(mel1, mel2, sr=22050):
    """
    计算梅尔倒谱距离
    
    Args:
        mel1, mel2: 梅尔频谱 [n_mels, time]
    """
    # 转换为线性频谱
    spec1 = librosa.feature.inverse.mel_to_stft(mel1, sr=sr)
    spec2 = librosa.feature.inverse.mel_to_stft(mel2, sr=sr)
    
    # 计算MFCC
    mfcc1 = librosa.feature.mfcc(S=spec1, sr=sr)
    mfcc2 = librosa.feature.mfcc(S=spec2, sr=sr)
    
    # DTW对齐
    D, wp = librosa.sequence.dtw(mfcc1, mfcc2)
    
    # MCD
    mcd = np.mean(D[wp[:, 0], wp[:, 1]])
    
    return mcd
```

---

## 11. 常见问题与易错点

### 11.1 训练不稳定

**问题**：解码器输出崩溃，loss震荡

**原因**：
- 学习率过大
- teacher forcing比例不当

**解决**：
- 使用较小学习率 (lr=1e-3)
- 逐渐减少teacher forcing

### 11.2 注意力不对齐

**问题**：生成语音和文本对不上

**原因**：
- 注意力收敛太快或太慢

**解决**：
- 调整注意力温度
- 加入位置编码

### 11.3 重复生成

**问题**：同一个音节反复生成

**原因**：
- 解码器自回归

**解决**：
- 加入pre-net
- 使用dropout

### 11.4 推理慢

**问题**：生成速度慢

**解决**：
- 减少解码步长
- 使用并行解码
- 使用轻量声码器

---

## 12. 学习总结

### 12.1 核心要点

| 要点 | 内容 |
|------|------|
| 本质 | Seq2Seq模型：文本→梅尔频谱 |
| 核心 | 注意力机制对齐 |
| 输出 | 梅尔频谱 + 声码器 = 波形 |
| 训练 | Teacher forcing + 自回归 |
| 推理 | 自回归解码 |

### 12.2 学习路径

```
入门
├── 基础语音处理（ librosa）
├── 序列模型（RNN/LSTM）
└── 注意力机制

进阶
├── CBHG模块
├── 声码器
└── 端到端训练

实战
├── 数据准备
├── 模型调参
└── 部署优化
```

### 12.3 扩展阅读

- Wang et al., "Tacotron", 2017
- Shen et al., "Tacotron 2", 2018
- FastSpeech系列

---

## 13. 练习题与思考题

### 13.1 基础练习

**练习1**：为什么Tacotron输���梅���频谱而不是波形？

**答案**：梅尔频谱是对语音的对数能量谱，符合人耳听觉特性，数据量小，易于训练。波形需要更高采样率，信息冗余大。

**练习2**：注意力在TTS中的作用是什么？

**答案**：对齐文本字符和生成的频谱帧。告诉解码器"当前应该关注文本的哪个位置"。

**练习3**：CBHG模块的作用？

**答案**：提取丰富的文本特征。卷积捕获n-gram，GRU捕获序列依赖，Highway融合。

### 13.2 进阶思考

**思考1**：端到端TTS相比传统TTS的优势？

**提示**：从系统复杂度、联合优化、灵活性等角度思考。

**思考2**：如何控制生成的语速和语调？

**提示**：在模型中引入额外的控制输入，如语速因子、基频。

**思考3**：为什么推理时会出现重复？

**提示**：自回归模型的固有问题，解码策略的影响。

### 13.3 编程练习

**练习**：实现一个最小可用的TTS系统

```python
# 要求：
# 1. 实现字符嵌入
# 2. 实现简单注意力
# 3. 实现解码器
# 4. 输入文本，输出梅尔频谱

# 提示：
# - 使用PyTorch
# - 参考上面的Tacotron代码
# - 先在简单数据上测试
```

---

## 14. 学习路径建议

### 14.1 入门（1周）

| 天 | 内容 | 目标 |
|----|------|------|
| 1-2 | 语音基础 | 理解采样率/频谱/梅尔 |
| 3-4 | Seq2Seq | 理解编码器-解码器 |
| 5-6 | 注意力 | 理解attention机制 |
| 7 | TTS库 | 运行demo |

### 14.2 进阶（2周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | CBHG + 声码器 | 理解完整pipeline |
| 2 | 训练技巧 | 调参与优化 |

### 14.3 实战（3周）

| 周 | 内容 | 目标 |
|----|------|------|
| 1 | 数据处理 | 准备数据 |
| 2 | 模型训练 | 实际训练 |
| 3 | 部署应用 | 做成demo |

---

## 附录

### A. 重要参考

| 参考 | 链接 |
|------|------|
| Tacotron论文 | https://arxiv.org/abs/1703.10135 |
| Tacotron 2 | https://arxiv.org/abs/1712.05884 |
| TTS库 | https://github.com/coqui-ai/TTS |

### B. 数据集

| 数据集 | 描述 |
|------|------|
| LJ Speech | 13K短音频 |
| VCTK | 多说话人 |
| LibriTTS | 长文本 |

### C. 代码资源

```python
# 推荐资源
# 1. Coqui TTS
# 2. TensorFlow TTS
# 3. ESPnet
```

---

**文档结束**