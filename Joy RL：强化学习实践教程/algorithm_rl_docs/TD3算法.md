# TD3（Twin Delayed DDPG）学习文档

> 解决DDPG过估计问题的双Q网络算法

---

## 1. 算法基础认知

**一句话定义**：TD3（Twin Delayed DDPG）通过使用两个Critic网络取较小的Q值来减少过估计，并引入延迟更新和噪声正则化来提高稳定性，是DDPG的改进版。

**直觉类比**：DDPG容易"骄傲自满"（过估计Q值），TD3的做法是"兼听则明"——让两个Critic分别评估，取较保守的那个，同时给动作加一点噪声让学习更稳健。

**历史背景**：由Fujita等人在2018年提出，是连续控制的表现最佳算法之一。

**算法定位**：
- 类型：深度强化学习 → 连续控制
- 输出：确定性动作
- 模型类型：双Critic + 延迟更新

---

## 2. 核心原理

### 2.1 三大改进

1. **双Q网络**：用两个Critic，取较小的Q值作为目标
2. **延迟更新**：Actor每2步更新一次
3. **目标噪声**：给目标动作加噪声，防止过拟合

### 2.2 公式

**双Q目标**：
$$y = r + \gamma \cdot \min_{i=1,2} Q_i(s', \mu(s') + \epsilon)$$

**目标噪声**：
$$\epsilon \sim N(0, 0.2)$$

---

## 3. 应用场景

- 机器人控制
- MuJoCo环境
- 高难度连续控制任务

---

## 4. 调库实现

```python
import torch
import torch.nn as nn
import numpy as np

class TD3Agent:
    def __init__(self, state_dim, action_dim):
        # 双Critic
        self.critic1 = self._build_critic(state_dim, action_dim)
        self.critic2 = self._build_critic(state_dim, action_dim)
        self.target_critic1 = self._build_critic(state_dim, action_dim)
        self.target_critic2 = self._build_critic(state_dim, action_dim)
        
        # Actor
        self.actor = self._build_actor(state_dim, action_dim)
        self.target_actor = self._build_actor(state_dim, action_dim)
        
        # 加载参数
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=3e-4
        )
        
        self.step_count = 0
        self.actor_update_freq = 2
    
    def _build_critic(self, s_dim, a_dim):
        return nn.Sequential(
            nn.Linear(s_dim + a_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def _build_actor(self, s_dim, a_dim):
        return nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, a_dim),
            nn.Tanh()
        )
    
    def select_action(self, state):
        with torch.no_grad():
            return self.actor(state).numpy()
    
    def update(self, batch):
        self.step_count += 1
        
        # Critic更新（每步）
        with torch.no_grad():
            target_a = self.target_actor(batch.s2)
            target_a += torch.randn_like(target_a) * 0.2
            target_a = target_a.clamp(-1, 1)
            
            target_Q1 = self.target_critic1(batch.s2, target_a)
            target_Q2 = self.target_critic2(batch.s2, target_a)
            target_Q = batch.r + 0.99 * torch.min(target_Q1, target_Q2)
        
        Q1 = self.critic1(batch.s, batch.a)
        Q2 = self.critic2(batch.s, batch.a)
        
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Actor延迟更新
        if self.step_count % self.actor_update_freq == 0:
            actor_loss = -self.critic1(batch.s, self.actor(batch.s)).mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            # 软更新目标网络
            self._update_target()
    
    def _update_target(self):
        for target, main in [(self.target_critic1, self.critic1),
                              (self.target_critic2, self.critic2),
                              (self.target_actor, self.actor)]:
            for tp, mp in zip(target.parameters(), main.parameters()):
                tp.data.copy_(0.005 * mp.data + 0.995 * tp.data)

if __name__ == "__main__":
    import gymnasium as gym
    
    env = gym.make('Pendulum-v1')
    agent = TD3Agent(3, 1)
    
    print("=" * 50)
    print("TD3测试")
    print("=" * 50)
    
    for ep in range(200):
        s, _ = env.reset()
        total_r = 0
        
        for _ in range(200):
            a = agent.select_action(torch.FloatTensor(s).unsqueeze(0))
            s2, r, d, t, _ = env.step(a)
            total_r += r
            d = d or t
        
        if ep % 50 == 0:
            print(f"回合{ep}: 奖励={total_r:.1f}")
```

---

## 5. 算法对比

| 特性 | DDPG | TD3 |
|------|------|-----|
| Critic数 | 1 | 2 |
| 过估计 | 严重 | 缓解 |
| 稳定性 | 一般 | 好 |
| 动作噪声 | 无 | 有 |

---

## 6. 总结

✓ TD3 = DDPG + 3大改进
✓ 减少Q值过估计
✓ 更稳定的连续控制