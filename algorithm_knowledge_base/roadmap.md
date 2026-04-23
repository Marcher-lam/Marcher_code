# 算法学习路径指南

> 从零开始的系统化学习路线

## 学习路径总览

```
初级阶段 ──────────────────> 中级阶段 ──────────────────> 高级阶段
(1-2个月)                    (2-3个月)                    (持续学习)
    │                            │                            │
    ▼                            ▼                            ▼
基础回归/分类              树模型/聚类/降维              深度学习/前沿
```

---

## 第一阶段：初级入门（1-2个月）

### Week 1-2：数学基础与回归

**学习目标**：理解回归问题本质，掌握优化思想

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [线性回归](./algorithms/线性回归.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 2 | [多项式线性回归](./algorithms/多项式线性回归.md) | ⭐⭐⭐ | 1天 |
| 3 | [岭回归](./algorithms/岭回归.md) | ⭐⭐⭐⭐ | 1天 |
| 4 | [LASSO回归](./algorithms/LASSO回归.md) | ⭐⭐⭐⭐ | 1天 |

**配套阅读**：
- [optimization.md](./utils/optimization.md) - 梯度下降与正则化
- [metrics.md](./utils/metrics.md) - MSE/RMSE/R²

### Week 3-4：基础分类

**学习目标**：理解分类问题，掌握概率模型

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [感知机](./algorithms/感知机.md) | ⭐⭐⭐⭐ | 1天 |
| 2 | [逻辑回归](./algorithms/逻辑回归.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 3 | [二项逻辑回归](./algorithms/二项逻辑回归.md) | ⭐⭐⭐ | 0.5天 |
| 4 | [多项式逻辑回归](./algorithms/多项式逻辑回归.md) | ⭐⭐⭐ | 0.5天 |
| 5 | [最大熵模型](./algorithms/最大熵模型.md) | ⭐⭐⭐ | 1天 |

**配套阅读**：
- [metrics.md](./utils/metrics.md) - Precision/Recall/F1/AUC

### Week 5-6：概率与距离方法

**学习目标**：掌握概率模型和距离度量方法

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [朴素贝叶斯](./algorithms/朴素贝叶斯.md) | ⭐⭐⭐⭐ | 2天 |
| 2 | [KNN](./algorithms/KNN.md) | ⭐⭐⭐⭐ | 1天 |
| 3 | [k-D tree](./algorithms/k-D_tree.md) | ⭐⭐⭐ | 1天 |
| 4 | [支持向量机](./algorithms/支持向量机.md) | ⭐⭐⭐⭐⭐ | 3天 |

### Week 7-8：神经网络基础

**学习目标**：理解神经网络基本原理

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [多层感知机](./algorithms/多层感知机.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 2 | [前馈神经网络](./algorithms/前馈神经网络.md) | ⭐⭐⭐⭐ | 1天 |
| 3 | [反向传播算法](./algorithms/反向传播算法.md) | ⭐⭐⭐⭐⭐ | 3天 |

---

## 第二阶段：中级进阶（2-3个月）

### Month 2：树模型系列

**学习目标**：深入理解决策树与集成方法

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [决策树](./algorithms/决策树.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 2 | [ID3](./algorithms/ID3.md) | ⭐⭐⭐⭐ | 1天 |
| 3 | [C4.5](./algorithms/C4.5.md) | ⭐⭐⭐⭐ | 1天 |
| 4 | [CART](./algorithms/CART.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 5 | [AdaBoost](./algorithms/AdaBoost.md) | ⭐⭐⭐⭐ | 2天 |
| 6 | [GBDT](./algorithms/GBDT.md) | ⭐⭐⭐⭐⭐ | 3天 |

### Month 3：无监督学习

**学习目标**：掌握聚类与降维方法

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [K-Means](./algorithms/K-Means.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 2 | [PCA](./algorithms/PCA.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 3 | [LDA](./algorithms/LDA.md) | ⭐⭐⭐⭐ | 2天 |
| 4 | [奇异值分解](./algorithms/奇异值分解.md) | ⭐⭐⭐⭐ | 2天 |
| 5 | [NMF](./algorithms/NMF.md) | ⭐⭐⭐ | 1天 |

### Month 3-4：概率图模型

**学习目标**：理解序列建模与概率图

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [EM](./algorithms/EM.md) | ⭐⭐⭐⭐ | 2天 |
| 2 | [变分EM](./algorithms/变分EM.md) | ⭐⭐⭐ | 1天 |
| 3 | [高斯混合EM](./algorithms/高斯混合EM.md) | ⭐⭐⭐⭐ | 2天 |
| 4 | [隐马尔可夫](./algorithms/隐马尔可夫.md) | ⭐⭐⭐⭐ | 3天 |
| 5 | [条件随机场](./algorithms/条件随机场.md) | ⭐⭐⭐⭐ | 3天 |
| 6 | [马尔可夫链蒙特卡洛](./algorithms/马尔可夫链蒙特卡洛.md) | ⭐⭐⭐ | 2天 |

### Month 4：主题模型

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [LSA](./algorithms/LSA.md) | ⭐⭐⭐ | 1天 |
| 2 | [PLSA](./algorithms/PLSA.md) | ⭐⭐⭐ | 1天 |

---

## 第三阶段：高级深入（持续学习）

### 深度学习 - CNN系列

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [卷积神经网络](./algorithms/卷积神经网络.md) | ⭐⭐⭐⭐⭐ | 3天 |
| 2 | [残差神经网络](./algorithms/残差神经网络.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 3 | [Unet](./algorithms/Unet.md) | ⭐⭐⭐⭐ | 2天 |

### 深度学习 - RNN系列

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [RNN](./algorithms/RNN.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 2 | [LSTM](./algorithms/LSTM.md) | ⭐⭐⭐⭐⭐ | 3天 |
| 3 | [GRU](./algorithms/GRU.md) | ⭐⭐⭐⭐ | 2天 |
| 4 | [DRNN](./algorithms/DRNN.md) | ⭐⭐⭐ | 1天 |

### 深度学习 - Attention与Transformer

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [Attention机制](./algorithms/Attention机制.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 2 | [Encoder-Decoder](./algorithms/Encoder-Decoder.md) | ⭐⭐⭐⭐ | 2天 |
| 3 | [MHA](./algorithms/MHA.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 4 | [Transformer](./algorithms/Transformer.md) | ⭐⭐⭐⭐⭐ | 4天 |
| 5 | [RNN-Search](./algorithms/RNN-Search.md) | ⭐⭐⭐ | 1天 |

### NLP词表示与预训练

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [one hot](./algorithms/one_hot.md) | ⭐⭐⭐ | 0.5天 |
| 2 | [TF-IDF](./algorithms/TF-IDF.md) | ⭐⭐⭐⭐ | 1天 |
| 3 | [word2vec](./algorithms/word2vec.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 4 | [char2vec](./algorithms/char2vec.md) | ⭐⭐⭐ | 1天 |
| 5 | [glove](./algorithms/glove.md) | ⭐⭐⭐⭐ | 2天 |
| 6 | [GPT](./algorithms/GPT.md) | ⭐⭐⭐⭐⭐ | 3天 |
| 7 | [Bert](./algorithms/Bert.md) | ⭐⭐⭐⭐⭐ | 3天 |

### 生成模型

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [AE](./algorithms/AE.md) | ⭐⭐⭐⭐ | 2天 |
| 2 | [VAE](./algorithms/VAE.md) | ⭐⭐⭐⭐⭐ | 3天 |
| 3 | [DAE](./algorithms/DAE.md) | ⭐⭐⭐ | 1天 |
| 4 | [GAN](./algorithms/GAN.md) | ⭐⭐⭐⭐⭐ | 3天 |
| 5 | [DCGAN](./algorithms/DCGAN.md) | ⭐⭐⭐⭐ | 2天 |
| 6 | [DDPM](./algorithms/DDPM.md) | ⭐⭐⭐⭐⭐ | 3天 |
| 7 | [DM](./algorithms/DM.md) | ⭐⭐⭐⭐ | 2天 |
| 8 | [SMLD](./algorithms/SMLD.md) | ⭐⭐⭐ | 2天 |

---

## 第四阶段：强化学习（选修）

### 基础概念

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [MDP](./algorithms/MDP.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 2 | [multi-armed bandits](./algorithms/multi-armed_bandits.md) | ⭐⭐⭐⭐ | 2天 |
| 3 | [UCB](./algorithms/UCB.md) | ⭐⭐⭐⭐ | 1天 |
| 4 | [Thompson Sampling](./algorithms/Thompson_Sampling.md) | ⭐⭐⭐⭐ | 1天 |

### 值函数方法

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [蒙特卡洛预测](./algorithms/蒙特卡洛预测.md) | ⭐⭐⭐⭐ | 2天 |
| 2 | [TD](./algorithms/TD.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 3 | [SARSA](./algorithms/SARSA.md) | ⭐⭐⭐⭐ | 2天 |
| 4 | [Q-learning](./algorithms/Q-learning.md) | ⭐⭐⭐⭐⭐ | 2天 |
| 5 | [DQN](./algorithms/DQN.md) | ⭐⭐⭐⭐⭐ | 3天 |

### 策略梯度

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [REINFORCE](./algorithms/REINFORCE.md) | ⭐⭐⭐⭐ | 2天 |
| 2 | [PPO](./algorithms/PPO.md) | ⭐⭐⭐⭐⭐ | 3天 |
| 3 | [A2C](./algorithms/A2C.md) | ⭐⭐⭐⭐ | 2天 |
| 4 | [ACER](./algorithms/ACER.md) | ⭐⭐⭐ | 2天 |

### Actor-Critic

| 顺序 | 算法 | 重要性 | 预计时间 |
|------|------|--------|----------|
| 1 | [DDPG](./algorithms/DDPG.md) | ⭐⭐⭐⭐ | 2天 |
| 2 | [SAC](./algorithms/SAC.md) | ⭐⭐⭐⭐⭐ | 3天 |
| 3 | [TD3](./algorithms/TD3.md) | ⭐⭐⭐⭐ | 2天 |

---

## 学习建议

### 时间安排

- **全日制学习**：按上述路径，约3-4个月完成初中级
- **业余学习**：每天2小时，约6-8个月完成初中级
- **面试准备**：优先学习⭐⭐⭐⭐⭐的算法

### 学习方法

1. **理解优先**：先理解原理，再看代码
2. **动手实践**：每个算法都要亲自实现
3. **对比学习**：相似算法对比理解
4. **项目驱动**：用实际项目巩固知识

### 推荐学习组合

| 应用场景 | 推荐学习组合 |
|----------|--------------|
| 回归问题 | 线性回归 → 岭回归 → LASSO |
| 二分类 | 逻辑回归 → SVM → 决策树 |
| 多分类 | 朴素贝叶斯 → 逻辑回归(多项式) |
| 聚类 | K-Means → 层次聚类 |
| 降维 | PCA → t-SNE |
| 文本分类 | TF-IDF → 朴素贝叶斯 → BERT |
| 图像分类 | CNN → ResNet |
| 序列建模 | RNN → LSTM → Transformer |
| 生成任务 | GAN → VAE → DDPM |

---

## 检查清单

### 初级阶段完成标志
- [ ] 能解释梯度下降原理
- [ ] 能区分L1和L2正则化
- [ ] 能手写线性回归
- [ ] 能手写逻辑回归
- [ ] 理解SVM的对偶问题
- [ ] 理解反向传播

### 中级阶段完成标志
- [ ] 能解释信息增益
- [ ] 理解Bagging和Boosting区别
- [ ] 能手写决策树
- [ ] 理解EM算法
- [ ] 理解HMM三个问题

### 高级阶段完成标志
- [ ] 能实现CNN
- [ ] 理解LSTM门控机制
- [ ] 能解释Self-Attention
- [ ] 理解Transformer架构
- [ ] 能实现简单GAN

---

**祝你学习顺利！**
