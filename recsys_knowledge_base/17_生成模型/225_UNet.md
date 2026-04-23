# UNet 学习文档

> 编码器-解码器的经典架构——扩散模型的核心

**UNet** 最初用于医学图像分割，因其对称的编码器-解码器结构加跳跃连接被广泛应用于扩散模型。

## 核心结构

```
编码器(下采样):          解码器(上采样):
  64×64 → 32×32          32×32 → 64×64
  32×32 → 16×16          16×16 → 32×32
  16×16 → 8×8            8×8 → 16×16
       ↓       ↑(跳跃连接)
     瓶颈层(8×8)

跳跃连接: 将编码器的特征直接传给对应的解码器层
```

## 数学表示

编码器输出: $e_l$（第l层）
解码器输入: $d_l = \text{UpConv}(\text{concat}(e_l, d_{l+1}))$

## 调库实现

```python
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU()
        )
    def forward(self, x): return self.block(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        # 编码器
        self.enc1 = UNetBlock(in_ch, 64)
        self.enc2 = UNetBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        # 瓶颈
        self.bottleneck = UNetBlock(128, 256)
        # 解码器
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = UNetBlock(128, 64)
        self.final = nn.Conv2d(64, out_ch, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

if __name__ == "__main__":
    model = SimpleUNet()
    x = torch.randn(2, 1, 64, 64)
    print(f"UNet: {x.shape} → {model(x).shape}")
```

## 学习总结

1. UNet = 对称编码器-解码器 + 跳跃连接
2. 跳跃连接保留细节信息
3. 扩散模型中UNet用于噪声预测
