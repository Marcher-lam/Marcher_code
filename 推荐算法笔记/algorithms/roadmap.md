# 推荐算法学习路径

## 初级阶段：推荐系统基础

### 1. 系统认知
- [ ] 推荐系统核心链路
- [ ] 推荐系统与广告系统的区别

### 2. 特征工程
- [ ] 特征等距分桶和等频分桶
- [ ] 特征重要度评估方法
- [ ] 高基数类别特征 Embedding 维度确定
- [ ] 特征 Shuffle 重要度评估

### 3. 基础损失函数与指标
- [ ] KL 散度与交叉熵
- [ ] 交叉熵 vs MSE
- [ ] AUC 物理意义与计算
- [ ] NDCG@K / Recall@K / HitRate@K

### 4. 树模型基础
- [ ] XGBoost vs GBDT
- [ ] XGBoost vs LightGBM
- [ ] XGBoost 过拟合与缺失值处理
- [ ] 树模型 vs 深度学习（表格数据）

---

## 中级阶段：推荐核心算法

### 1. 召回与粗排
- [ ] 召回负采样方法（随机/硬负/混合/动态）
- [ ] ESANS 语义感知负采样
- [ ] CROLoss 召回优化损失
- [ ] 双塔模型 Layer Normalization

### 2. 精排特征交叉
- [ ] FFM 域感知因子分解机
- [ ] SENet 特征域注意力
- [ ] DCN / DCN-v2 特征交叉网络
- [ ] Wukong 缩放定律推荐模型
- [ ] RankMixer / OneTrans / TokenMixer-Large

### 3. 注意力机制
- [ ] DIN 深度兴趣网络（含时间衰减）
- [ ] GQA 分组查询注意力
- [ ] Gated Attention 门控注意力
- [ ] MLA / GQA / DSA 对比

### 4. 序列建模
- [ ] 超长行为序列建模（SIM/MIMN/TWIN/LONGER）
- [ ] SIM 长序列搜索方案
- [ ] TIN 时间兴趣网络
- [ ] KuaiFormer 序列建模
- [ ] HSTU vs Transformer

### 5. 多任务与多场景
- [ ] 多任务 Loss 权重平衡
- [ ] MMoE / PLE 多任务模型
- [ ] MMoE 极化现象与解决
- [ ] PPNet / EPNet / PEPNet 个性化网络

---

## 高级阶段：前沿技术与深度模型

### 1. 因果推断与 Uplift
- [ ] Uplift 模型总览
- [ ] DragonNet 深度 Uplift
- [ ] DESCN 全空间交叉网络
- [ ] AUUC 评估指标

### 2. CVR/LTV 预估
- [ ] CVR 样本稀疏问题
- [ ] CVR 样本选择偏差（ESMM / ESCM² / UKD）
- [ ] DFM 延迟反馈模型
- [ ] 电商大促 CVR 优化
- [ ] LTV 建模方案

### 3. 冷启动
- [ ] POSO 个性化冷启动
- [ ] 广告冷启动 vs 物品冷启动

### 4. Embedding 与特征进阶
- [ ] 多模态语义 ID 编码（RQ-VAE / RQ-Kmeans / COBRA）
- [ ] 多模态特征融合
- [ ] 预训练 User/Item Embedding 利用

---

## 专家阶段：大模型与强化学习

### 1. 生成式推荐
- [ ] 生成式推荐 vs 传统 DLRM
- [ ] 业界生成式推荐方案梳理
- [ ] RQ-VAE 语义 ID 编码与码本坍塌
- [ ] MTGR / OneRec / OneRec V2
- [ ] TIGER / SUM / HSTU
- [ ] UniDex / UniSearch

### 2. 大模型核心技术
- [ ] Transformer 参数量推导 / FFN / Mask 机制
- [ ] Pre-Norm vs Post-Norm
- [ ] Qwen 迭代改进
- [ ] NSA 稀疏注意力
- [ ] MTP 多 Token 预测
- [ ] MOE 负载均衡
- [ ] RoPE 旋转位置编码
- [ ] BPE / WordPiece 分词
- [ ] RAG 检索增强生成
- [ ] MLA 多头潜在注意力
- [ ] LoRA / AdaLoRA / QLoRA
- [ ] Decoder-only 架构
- [ ] DPO 算法

### 3. 强化学习
- [ ] 强化学习算法分类（价值/策略/Actor-Critic）
- [ ] 策略梯度数学推导
- [ ] RLHF（PPO / DPO / GRPO）
- [ ] Q 函数 / 价值函数 / 优势函数
- [ ] MDP 马尔科夫决策过程
- [ ] DQN / Double DQN / Dueling DQN
- [ ] Decision Transformer

### 4. 基础八股补充
- [ ] Self Attention 除以 √dk
- [ ] Adam / AdamW 优化器
- [ ] Dropout 训练预测一致性
- [ ] One Epoch 现象
- [ ] BatchNorm vs LayerNorm vs RMSNorm
- [ ] L1 vs L2 正则化
- [ ] Attention 层 vs 全连接层
- [ ] 离线 AUC 在线 AB 不一致
- [ ] 模型融合 / 参数初始化 / 过拟合 / NaN
- [ ] 假设检验（AB 测试）
