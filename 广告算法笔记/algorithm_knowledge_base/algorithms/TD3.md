# TD3（Twin Delayed DDPG）学习文档

## 1. 算法基础认知

TD3 是 DDPG 的改进版本，通过三个关键技巧解决 DDPG 中的 Q 值过估计问题：

1. **双 Q 网络（Clipped Double-Q）**：取两个 Q 网络中的较小值作为目标
2. **延迟策略更新（Delayed Policy Updates）**：策略更新频率低于 Q 网络
3. **目标策略平滑（Target Policy Smoothing）**：在目标动作上添加噪声

## 2. 核心原理

### Clipped Double-Q 目标

$$
y = r + \gamma \min_{i=1,2} Q'_{\theta_i'}(s', \mu'(s') + \tilde{\epsilon})
$$

其中 $\tilde{\epsilon} \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$

### 延迟更新

每 d 次 Critic 更新后才更新一次 Actor（通常 d=2）。

## 3. 在广告中的应用

- 连续动作空间的出价调整（DDPG 的升级版）
- 更稳定的训练过程

## 4. 学习总结

TD3 是 DDPG 的改进版，通过双 Q 网络和延迟更新提升稳定性。在广告出价中作为 DDPG 的替代方案。
