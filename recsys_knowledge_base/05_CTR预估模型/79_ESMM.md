# ESMM (Entire Space Multi-Task Model) 学习文档

## 1. 算法基础认知

### 1.1 什么是 ESMM？

ESMM（Entire Space Multi-Task Model）是阿里提出的多任务学习模型，专门解决 CVR 预估中的**样本选择偏差**和**数据稀疏**问题。

### 1.2 业务背景

```
电商场景转化漏斗:
曝光(Impression) → 点击(Click) → 转化(Conversion)

传统方法的问题:
- CVR 训练用点击样本，推理用曝光样本 → 样本选择偏差(SSB)
- 转化样本极少 → 数据稀疏
```

### 1.3 核心创新

```
pCVR = pCTCVR / pCTR

其中:
- pCTR = P(click=1 | impression)
- pCVR = P(conversion=1 | click=1)
- pCTCVR = P(conversion=1 | click=1, impression)

在全部曝光空间上训练，避免 SSB
```

## 2. 核心原理

### 2.1 数学关系

$$P(\text{click}, \text{conversion}) = P(\text{click}) \times P(\text{conversion} | \text{click})$$

即:
$$pCTCVR = pCTR \times pCVR$$

### 2.2 损失函数

$$L = L_{CTR} + L_{CTCVR}$$

$$= -\sum_{i} \left[ y_i \log pCTR(x_i) + (1-y_i) \log(1-pCTR(x_i)) \right]$$
$$- \sum_{i} \left[ z_i \log(pCTR(x_i) \times pCVR(x_i)) + (1-z_i) \log(1-pCTR(x_i) \times pCVR(x_i)) \right]$$

其中:
- $y_i$: 点击标签
- $z_i$: 转化标签
- $pCTR(x_i) \times pCVR(x_i) = pCTCVR(x_i)$

## 3. 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class ESMM(nn.Module):
    """
    Entire Space Multi-Task Model
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 mlp_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.2):
        """
        参数:
            field_dims: 各域特征数量
            embed_dim: 嵌入维度
            mlp_dims: MLP 隐藏层维度
            dropout: Dropout 比率
        """
        super().__init__()

        self.num_fields = len(field_dims)

        # 共享嵌入层
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.field_offsets = torch.tensor([0] + np.cumsum(field_dims)[:-1].tolist())

        # CTR Tower
        self.ctr_mlp = self._build_mlp(embed_dim * len(field_dims), mlp_dims, dropout)
        self.ctr_output = nn.Linear(mlp_dims[-1], 1)

        # CVR Tower
        self.cvr_mlp = self._build_mlp(embed_dim * len(field_dims), mlp_dims, dropout)
        self.cvr_output = nn.Linear(mlp_dims[-1], 1)

    def _build_mlp(self, input_dim: int, mlp_dims: List[int],
                   dropout: float) -> nn.Sequential:
        """构建 MLP"""
        layers = []
        prev_dim = input_dim

        for dim in mlp_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim

        return nn.Sequential(*layers)

    def _get_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """获取嵌入"""
        X_offset = X + self.field_offsets.to(X.device)
        embeds = self.embedding(X_offset)  # (batch, num_fields, embed_dim)
        return embeds.view(X.size(0), -1)  # (batch, num_fields * embed_dim)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数:
            X: (batch, num_fields) 特征索引

        返回:
            pCTR, pCVR, pCTCVR
        """
        # 嵌入
        embeds = self._get_embeddings(X)

        # CTR Tower
        ctr_hidden = self.ctr_mlp(embeds)
        p_ctr = torch.sigmoid(self.ctr_output(ctr_hidden))

        # CVR Tower
        cvr_hidden = self.cvr_mlp(embeds)
        p_cvr = torch.sigmoid(self.cvr_output(cvr_hidden))

        # CTCVR = CTR * CVR
        p_ctcvr = p_ctr * p_cvr

        return p_ctr.squeeze(-1), p_cvr.squeeze(-1), p_ctcvr.squeeze(-1)


class ESMMTrainer:
    """
    ESMM 训练器
    """

    def __init__(self, model: ESMM, learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.bce_loss = nn.BCELoss()

    def train_step(self, X: torch.Tensor, click_labels: torch.Tensor,
                   conversion_labels: torch.Tensor) -> Dict[str, float]:
        """
        训练一步

        参数:
            X: 特征
            click_labels: 点击标签 (0/1)
            conversion_labels: 转化标签 (0/1)

        返回:
            损失字典
        """
        self.model.train()
        self.optimizer.zero_grad()

        p_ctr, p_cvr, p_ctcvr = self.model(X)

        # CTR 损失（在全部曝光样本上）
        loss_ctr = self.bce_loss(p_ctr, click_labels.float())

        # CTCVR 损失（在全部曝光样本上）
        loss_ctcvr = self.bce_loss(p_ctcvr, conversion_labels.float())

        # 总损失
        loss = loss_ctr + loss_ctcvr

        loss.backward()
        self.optimizer.step()

        return {
            'total_loss': loss.item(),
            'ctr_loss': loss_ctr.item(),
            'ctcvr_loss': loss_ctcvr.item()
        }

    def predict(self, X: torch.Tensor) -> Dict[str, np.ndarray]:
        """预测"""
        self.model.eval()
        with torch.no_grad():
            p_ctr, p_cvr, p_ctcvr = self.model(X)

        return {
            'p_ctr': p_ctr.cpu().numpy(),
            'p_cvr': p_cvr.cpu().numpy(),
            'p_ctcvr': p_ctcvr.cpu().numpy()
        }


class MMOE_ESMM(nn.Module):
    """
    MMOE + ESMM 结合版本
    """

    def __init__(self, field_dims: List[int], embed_dim: int = 10,
                 n_experts: int = 4, expert_dim: int = 64,
                 tower_dims: List[int] = [64, 32]):
        super().__init__()

        self.num_fields = len(field_dims)

        # 嵌入层
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.field_offsets = torch.tensor([0] + np.cumsum(field_dims)[:-1].tolist())

        # Expert 层
        input_dim = embed_dim * len(field_dims)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, expert_dim)
            )
            for _ in range(n_experts)
        ])

        # Gate 层
        self.ctr_gate = nn.Linear(input_dim, n_experts)
        self.cvr_gate = nn.Linear(input_dim, n_experts)

        # Tower 层
        self.ctr_tower = self._build_tower(expert_dim, tower_dims)
        self.cvr_tower = self._build_tower(expert_dim, tower_dims)

    def _build_tower(self, input_dim: int, tower_dims: List[int]) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for dim in tower_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = X.size(0)

        # 嵌入
        X_offset = X + self.field_offsets.to(X.device)
        embeds = self.embedding(X_offset).view(batch_size, -1)

        # Expert 输出
        expert_outputs = torch.stack(
            [expert(embeds) for expert in self.experts],
            dim=1
        )  # (batch, n_experts, expert_dim)

        # CTR Gate
        ctr_gate = F.softmax(self.ctr_gate(embeds), dim=-1)  # (batch, n_experts)
        ctr_expert_out = torch.einsum('be,bef->bf', ctr_gate, expert_outputs)
        p_ctr = torch.sigmoid(self.ctr_tower(ctr_expert_out)).squeeze(-1)

        # CVR Gate
        cvr_gate = F.softmax(self.cvr_gate(embeds), dim=-1)
        cvr_expert_out = torch.einsum('be,bef->bf', cvr_gate, expert_outputs)
        p_cvr = torch.sigmoid(self.cvr_tower(cvr_expert_out)).squeeze(-1)

        # CTCVR
        p_ctcvr = p_ctr * p_cvr

        return p_ctr, p_cvr, p_ctcvr


def demo_esmm():
    """ESMM 示例"""
    # 模拟数据配置
    field_dims = [100, 50, 20, 10]  # 4个特征域
    n_samples = 1000

    # 创建模型
    model = ESMM(
        field_dims=field_dims,
        embed_dim=10,
        mlp_dims=[128, 64]
    )

    # 模拟数据
    X = torch.zeros(n_samples, len(field_dims), dtype=torch.long)
    for i, dim in enumerate(field_dims):
        X[:, i] = torch.randint(0, dim, (n_samples,))

    # 模拟标签（转化率较低）
    click_labels = (torch.rand(n_samples) > 0.8).long()  # 20% 点击率
    conversion_labels = (click_labels * (torch.rand(n_samples) > 0.9)).long()  # 点击中 10% 转化

    # 训练
    trainer = ESMMTrainer(model)

    for epoch in range(5):
        losses = trainer.train_step(X, click_labels, conversion_labels)
        print(f"Epoch {epoch+1}: {losses}")

    # 预测
    preds = trainer.predict(X[:10])
    print(f"\n预测示例:")
    print(f"pCTR: {preds['p_ctr'][:5]}")
    print(f"pCVR: {preds['p_cvr'][:5]}")
    print(f"pCTCVR: {preds['p_ctcvr'][:5]}")


if __name__ == "__main__":
    demo_esmm()
```

## 4. 与传统方法对比

### 4.1 传统 CVR 预估的问题

```
传统方法:
- 训练: 只用点击样本
- 推理: 对所有曝光样本预测

问题:
1. 样本选择偏差(SSB): 训练分布 ≠ 推理分布
2. 数据稀疏: 转化样本极少
```

### 4.2 ESMM 的优势

```
ESMM 方法:
- 训练: 使用全部曝光样本
- 通过 pCTR × pCVR = pCTCVR 间接学习 pCVR

优势:
1. 无 SSB: 训练和推理空间一致
2. 数据增强: 利用全部曝光数据
3. 知识迁移: CTR 任务帮助 CVR 学习
```

## 5. 实际应用

### 5.1 特征设计

```python
# CVR 相关特征
cvr_features = {
    # 用户特征
    'user_historical_cvr': '用户历史转化率',
    'user_purchase_power': '用户购买力',

    # 物品特征
    'item_historical_cvr': '物品历史转化率',
    'item_price_level': '价格档位',

    # 上下文特征
    'time_to_conversion': '点击到转化的时间'
}
```

### 5.2 业务应用

```python
class ESMMService:
    """
    ESMM 在线服务
    """

    def __init__(self, model_path: str):
        self.model = torch.load(model_path)
        self.model.eval()

    def predict(self, features: Dict) -> Dict[str, float]:
        """预测"""
        # 特征处理
        X = self._process_features(features)

        with torch.no_grad():
            p_ctr, p_cvr, p_ctcvr = self.model(X)

        return {
            'p_ctr': float(p_ctr),
            'p_cvr': float(p_cvr),
            'p_ctcvr': float(p_ctcvr),
            # 业务排序分数
            'rank_score': float(p_ctr * p_cvr * self._get_price_weight(features))
        }

    def _get_price_weight(self, features: Dict) -> float:
        """根据价格调整排序权重"""
        price = features.get('price', 0)
        return np.log1p(price)  # 价格越高，转化价值越大
```

## 6. 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| embed_dim | 10-20 | 嵌入维度 |
| mlp_dims | [256, 128] | Tower 隐藏层 |
| dropout | 0.1-0.3 | 防止过拟合 |
| learning_rate | 0.001 | 学习率 |

## 7. 学习总结

### 7.1 核心要点

1. **全空间训练**: 在曝光空间而非点击空间训练
2. **数学关系**: pCTCVR = pCTR × pCVR
3. **多任务学习**: CTR 和 CVR 共享嵌入

### 7.2 适用场景

- 电商转化预估
- 广告转化率预测
- 应用下载转化

## 8. 练习题

1. 实现带损失权重的 ESMM（CTR 和 CVR 不同权重）。

2. 比较 ESMM 和独立训练 CTR/CVR 的效果。

3. 分析 ESMM 在不同转化率场景下的表现。
