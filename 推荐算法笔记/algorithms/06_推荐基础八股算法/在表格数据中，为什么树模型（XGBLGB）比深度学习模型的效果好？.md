# 面试题：在表格数据中，为什么树模型（XGB\LGB）比深度学习模型的效果好？

面试题：在表格数据中，为什么树模型（XGB\LGB）比深度学习模型的效果好？

# 回答：

在表格数据集中，树模型（如 XGBoost、LightGBM）通常优于深度学习模型（如 MLP、ResNet），这一现象已被多项研究验证。以下是核心原因的分析：

# 1. 神经网络对非平滑函数的建模能力较弱

平滑性偏置问题：表格数据中的目标函数往往包含大量不规则、非平滑的模式（如阶跃式变化或局部突变）。

 神经网络（尤其是 MLP）倾向于学习平滑的决策边界（低频函数）。  
 树模型通过分段常数函数直接拟合这些不规则模式，无需平滑假设。

从数学角度看，ReLU 激活函数构成的神经网络是连续的分段线性函数，而树模型是分段常数函数。当目标函数存在剧烈跳变时，分段常数拟合天然更高效。研究表明，在合成数据集上，当目标函数包含大量阶跃型突变时，MLP 需要 10 倍以上的参数量才能达到树模型的拟合效果。

# 2. 树模型对噪音特征更具鲁棒性

特征选择机制：树模型在分裂节点时通过信息增益、基尼系数等指标自动筛选重要特征，天然忽略噪音特征（如随机噪声或无关特征列）。  
深度学习的敏感性：神经网络缺乏内置的特征选择机制，非信息特征会稀释模型的注意力。实验显示，当数据集中 $50 \%$ 的特征被随机替换后，ResNet 的准确率下降幅度远超 XGBoost。若主动移除噪音特征，神经网络与树模型的差距显著缩小。

在推荐系统的用户画像特征中，经常存在大量低信息量特征（如用户设备型号的哈希值）。树模型能自动降低这些特征的分裂优先级，而神经网络会为这些特征分配不必要的权重。

# 3. 表格数据的旋转非不变性

旋转不变性的矛盾：神经网络具有旋转不变性（即对特征进行线性变换不影响模型性能），但真实表格数据的特征通常具有方向性（如某一列代表年龄，另一列代表收入）。旋转操作会破坏原始特征的物理意义，导致树模型性能下降，而神经网络保持不变。

旋转不变性在表格数据中反而成为劣势，因为它忽视了特征本身的统计特性，而树模型通过特征方向性感知更贴合实际数据结构。

直觉理解：假设我们有一个"年龄"特征和一个"收入"特征，它们各自有明确的含义和分布。如果对这两个特征做旋转变换（如 PCA），新特征就失去了可解释性，但神经网络依然能同等处理。这种"无所谓"的态度意味着网络没有利用到特征本身的语义信息。

# 4. 样本不均匀与特征尺度敏感性

树模型基于特征的排序进行分裂，因此对特征的尺度（scale）不敏感，无需归一化。而神经网络对输入特征的尺度高度敏感，不同尺度的特征会导致梯度不平衡，影响训练稳定性。

在推荐系统的点击率预估任务中，特征可能包括用户年龄（0-100）、历史点击次数（0-10000）、物品价格（0-9999）等，量级差异巨大。树模型天然不受影响，而神经网络必须仔细进行特征归一化。

# 5. 表格数据集规模通常较小

深度学习的优势在于大规模数据上的扩展性。而表格数据集通常样本量有限（几千到几十万），在这个数据量级上，树模型凭借强大的归纳偏置，能更高效地从有限数据中学习。深度学习模型在数据量不足时更容易过拟合。

经验法则：当样本量小于 10 万时，XGBoost/LightGBM 几乎总是首选；当样本量超过百万且特征工程复杂时，深度学习才有机会发挥优势。

# 6. 深度学习何时能在表格数据上胜出？

尽管树模型总体占优，但以下场景深度学习可能更好：

- **超大规模数据**：当样本量达到千万级以上，深度学习的扩展性优势显现
- **特征间存在复杂交互**：如特征间存在高阶交叉关系，神经网络可以自动学习
- **多模态融合**：表格数据与文本、图像等非结构化数据混合时
- **在线学习场景**：需要快速增量更新模型时

# 7. 近年深度学习表格模型进展

为弥补深度学习在表格数据上的不足，近年出现了专门针对表格数据的深度学习架构：

- **TabNet (2020)**：Google 提出的基于注意力机制的表格数据模型，通过逐特征选择实现可解释性
- **FT-Transformer (2021)**：将表格特征嵌入为 token 后使用 Transformer 编码器处理
- **NODE (2019)**：Neural Oblivious Decision Ensembles，将可微决策树与深度学习结合
- **TabPFN (2022)**：基于元学习的表格数据模型，在小数据集上表现优异
- **GBDT-PL (2023)**：将 GBDT 的叶子节点编码作为深度网络的输入特征

# 8. 代码对比实验

下面通过代码实验验证树模型与深度学习在表格数据上的性能差异：

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

def load_tabular_data():
    data = fetch_openml(name='electricity', version=1, as_frame=True)
    df = data.frame.copy()
    cat_cols = df.select_dtypes(include=['category', 'object']).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    X = df.drop(columns=['class']).values.astype(np.float32)
    y = (df['class'].values == 1).astype(np.float32)
    return train_test_split(X, y, test_size=0.2, random_state=42)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_test, y_test, input_dim, epochs=30, lr=1e-3, batch_size=512):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train_s),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SimpleMLP(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test_s)
        proba = model(X_t).numpy().flatten()
        preds = (proba > 0.5).astype(int)
    return accuracy_score(y_test, preds), roc_auc_score(y_test, proba)

def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return accuracy_score(y_test, preds), roc_auc_score(y_test, proba)

def add_noise_features(X, noise_ratio=0.5):
    n_noise = int(X.shape[1] * noise_ratio)
    noise = np.random.randn(X.shape[0], n_noise).astype(np.float32)
    return np.hstack([X, noise]), n_noise

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_tabular_data()
    input_dim = X_train.shape[1]

    print("=" * 50)
    print("实验1: 原始数据对比")
    print("=" * 50)

    xgb_acc, xgb_auc = train_xgboost(X_train, y_train, X_test, y_test)
    mlp_acc, mlp_auc = train_mlp(X_train, y_train, X_test, y_test, input_dim)

    print(f"XGBoost  -> Accuracy: {xgb_acc:.4f}, AUC: {xgb_auc:.4f}")
    print(f"MLP      -> Accuracy: {mlp_acc:.4f}, AUC: {mlp_auc:.4f}")

    print("\n" + "=" * 50)
    print("实验2: 添加50%噪音特征后的对比")
    print("=" * 50)

    X_train_noisy, n_noise = add_noise_features(X_train, noise_ratio=0.5)
    X_test_noisy, _ = add_noise_features(X_test, noise_ratio=0.5)
    input_dim_noisy = X_train_noisy.shape[1]

    xgb_acc_n, xgb_auc_n = train_xgboost(X_train_noisy, y_train, X_test_noisy, y_test)
    mlp_acc_n, mlp_auc_n = train_mlp(X_train_noisy, y_train, X_test_noisy, input_dim_noisy)

    print(f"XGBoost  -> Accuracy: {xgb_acc_n:.4f}, AUC: {xgb_auc_n:.4f}")
    print(f"MLP      -> Accuracy: {mlp_acc_n:.4f}, AUC: {mlp_auc_n:.4f}")

    print("\n性能下降对比:")
    print(f"XGBoost AUC下降: {xgb_auc - xgb_auc_n:.4f}")
    print(f"MLP     AUC下降: {mlp_auc - mlp_auc_n:.4f}")
```

# 9. 常见误区与注意事项

- **误区1**："深度学习一定比树模型好"。在表格数据上这通常不成立，应优先尝试树模型。
- **误区2**："树模型不需要任何特征工程"。虽然树模型对特征尺度不敏感，但合理的特征工程（如目标编码、特征交叉）仍然能显著提升效果。
- **误区3**："在表格数据上深度学习毫无价值"。当数据量极大、特征交互复杂或需要端到端联合训练时，深度学习仍有其优势。
- **注意**：在推荐系统的排序阶段，经常使用"树模型特征 + 深度学习"的混合方案，如用 GBDT 的叶子节点编码作为 MLP 的输入特征（GBDT+LR 思路的延伸）。

# 10. 总结

| 维度 | 树模型 (XGBoost/LightGBM) | 深度学习 (MLP) |
|------|--------------------------|---------------|
| 表格数据性能 | 优 | 中 |
| 对噪音特征鲁棒性 | 强 | 弱 |
| 特征预处理要求 | 低 | 高（需归一化） |
| 大规模数据扩展性 | 中 | 优 |
| 训练速度 | 快 | 慢 |
| 可解释性 | 中（特征重要性） | 低 |
| 超参数敏感度 | 中 | 高 |

参考论文：Why do tree-based models still outperform deep learning on tabular data?

# 6.2 Transformer 面试题
