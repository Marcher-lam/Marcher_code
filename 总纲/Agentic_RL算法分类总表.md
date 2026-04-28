# Agentic RL 算法分类总表

> 主题：目前主流与前沿的 **Agentic Reinforcement Learning（Agentic RL，智能体强化学习 / 大模型智能体强化学习）** 算法、方法、训练范式、奖励设计与工程框架分类。  
> 说明：Agentic RL 不是单一算法，而是一组把大语言模型、多模态模型或智能体系统放入可交互环境中，通过奖励信号优化其“规划、行动、工具使用、记忆、协作、反思、执行”的方法体系。  
> 适用对象：LLM Agent、Web Agent、Code Agent、Tool Agent、RAG Agent、GUI Agent、机器人 Agent、多智能体系统、自动化工作流 Agent。

---

## 1. 总体分类框架

Agentic RL 可以从 9 个维度划分：

1. **按训练目标分类**
   - 对齐类 Agentic RL
   - 推理类 Agentic RL
   - 工具使用类 Agentic RL
   - 任务执行类 Agentic RL
   - 多步规划类 Agentic RL
   - 多智能体协作类 Agentic RL
   - 机器人/具身智能类 Agentic RL

2. **按奖励来源分类**
   - 人类反馈奖励
   - AI 反馈奖励
   - 偏好奖励
   - 可验证奖励
   - 环境返回奖励
   - 工具调用奖励
   - 过程奖励
   - 结果奖励
   - 混合奖励

3. **按优化算法分类**
   - PPO 系列
   - GRPO 系列
   - REINFORCE 系列
   - Actor-Critic 系列
   - DPO/IPO/KTO/ORPO/SimPO 等直接偏好优化系列
   - Q-learning / Value-based 系列
   - Offline RL / Batch RL 系列
   - Imitation + RL 混合系列

4. **按智能体能力分类**
   - Planning Agent
   - Tool-use Agent
   - Browser/Web Agent
   - Code Agent
   - RAG Agent
   - GUI/Computer-use Agent
   - Embodied Agent
   - Multi-agent Agent
   - Self-improving Agent

5. **按交互环境分类**
   - 文本环境
   - 网页环境
   - 代码环境
   - 数据库环境
   - 游戏环境
   - 模拟器环境
   - 机器人环境
   - 企业工作流环境
   - 多模态 GUI 环境

6. **按轨迹粒度分类**
   - 单轮响应级 RL
   - 多轮对话级 RL
   - 多步工具调用级 RL
   - 任务轨迹级 RL
   - 工作流级 RL
   - 长程生命周期级 RL

7. **按训练数据分类**
   - 人类示范轨迹
   - 模型自生成轨迹
   - 环境交互轨迹
   - 偏好比较数据
   - 失败案例数据
   - 合成任务数据
   - 自动课程学习数据

8. **按策略结构分类**
   - 单策略 Agent
   - Planner-Executor Agent
   - Actor-Critic Agent
   - Hierarchical Agent
   - Multi-agent Debate/Collaboration
   - Tool-router Agent
   - Memory-augmented Agent

9. **按落地场景分类**
   - 数学推理
   - 编程
   - 搜索与浏览
   - 数据分析
   - Text-to-SQL
   - 法律/金融/医疗专业 Agent
   - 科研 Agent
   - 自动办公 Agent
   - 机器人控制 Agent

---

# 2. 基础后训练类 Agentic RL

这类算法主要来源于 LLM 后训练体系，目标是让模型更符合人类偏好、更安全、更有帮助。

## 2.1 RLHF：Reinforcement Learning from Human Feedback

### 核心思想
使用人类偏好数据训练奖励模型，再用强化学习优化语言模型策略。

### 典型流程
1. 监督微调 SFT
2. 收集人类偏好比较数据
3. 训练 Reward Model
4. 使用 PPO 或类似算法优化策略模型

### 代表算法
- RLHF
- PPO-RLHF
- Reward Model + PPO
- SFT + RM + RL
- InstructGPT-style RLHF
- ChatGPT-style RLHF

### Agentic RL 中的用途
- 提升 Agent 回答有用性
- 减少幻觉
- 提升任务遵循能力
- 让工具调用更符合用户意图
- 改善多轮对话稳定性

---

## 2.2 RLAIF：Reinforcement Learning from AI Feedback

### 核心思想
用 AI 模型代替或辅助人类提供反馈。

### 代表方法
- Constitutional AI
- AI Feedback Reward Model
- LLM-as-a-Judge Reward
- Self-Critique Reward
- Debate-based AI Feedback
- Multi-agent Judging Reward

### Agentic RL 中的用途
- 大规模自动评价 Agent 轨迹
- 自动判断工具调用是否合理
- 自动打分长任务完成质量
- 低成本生成偏好数据
- 用强模型指导弱模型 Agent

---

## 2.3 RLCF：Reinforcement Learning from Critic Feedback

### 核心思想
用 Critic 模型或评判器对 Agent 行为进行打分，再进行策略优化。

### 代表方法
- Critic-guided RL
- Self-Critic RL
- Process Critic RL
- Outcome Critic RL
- Tool-call Critic RL
- Reflection Critic RL

### 应用
- 数学推理
- 代码生成
- 工具调用
- Web Agent
- 多步任务执行

---

## 2.4 RLHF-V：面向多模态/视觉反馈的 RLHF

### 代表方法
- Vision-Language RLHF
- Multimodal Reward Model
- Image Preference RLHF
- GUI Action Feedback RLHF
- Embodied Feedback RLHF

### 应用
- 多模态 Agent
- 视觉问答 Agent
- GUI Agent
- 机器人 Agent
- Computer-use Agent

---

# 3. 直接偏好优化类 Agentic RL

这类方法不一定显式训练奖励模型，而是直接利用偏好数据优化模型。

## 3.1 DPO：Direct Preference Optimization

### 核心思想
直接利用 preferred / rejected 样本对优化语言模型，使模型更偏向被偏好的回答。

### 特点
- 不需要显式 Reward Model
- 不需要在线 RL rollouts
- 训练稳定
- 工程实现简单

### Agentic RL 用途
- 优化 Agent 的回答偏好
- 优化多轮对话风格
- 优化工具调用轨迹偏好
- 优化代码 Agent 的方案选择

---

## 3.2 IPO：Identity Preference Optimization

### 核心思想
改进 DPO 目标，缓解过度优化偏好差异的问题。

### 用途
- 稳定偏好学习
- 减少过拟合
- 用于 Agent 回答风格对齐

---

## 3.3 KTO：Kahneman-Tversky Optimization

### 核心思想
不依赖成对偏好，而使用“好/坏”单样本反馈进行优化。

### 用途
- 数据更容易收集
- 适合 Agent 任务成功/失败反馈
- 适合自动化环境中的二值奖励

---

## 3.4 ORPO：Odds Ratio Preference Optimization

### 核心思想
把 SFT 与偏好优化结合在一个目标中。

### 用途
- 简化训练流程
- 提高对齐效率
- 适合 Agent 初始能力对齐

---

## 3.5 SimPO：Simple Preference Optimization

### 核心思想
去掉参考模型依赖，使用更简单的偏好优化目标。

### 用途
- 轻量化 Agent 对齐
- 降低训练成本
- 适合中小模型 Agent

---

## 3.6 CPO：Contrastive Preference Optimization

### 核心思想
通过对比偏好样本优化模型。

### 用途
- 多候选轨迹排序
- 代码 Agent 方案选择
- Web Agent 路径选择

---

## 3.7 RRHF：Rank Responses to align Human Feedback

### 核心思想
使用排序式偏好数据对多个候选回答排序学习。

### 用途
- 多轨迹 Agent 选择
- 多方案规划排序
- 生成多个解决路径后选择最优路径

---

## 3.8 SLiC-HF

### 核心思想
用序列似然校准人类反馈。

### 用途
- 对话 Agent 对齐
- 文本生成 Agent 对齐

---

## 3.9 BCO：Binary Classifier Optimization

### 核心思想
把偏好优化转化为二分类判别问题。

### 用途
- 任务成功/失败轨迹学习
- Web Agent 轨迹选择
- 工具调用成功率优化

---

# 4. PPO / Policy Gradient 类 Agentic RL

这类是最经典的强化学习优化方法，常用于在线交互式 Agent 训练。

## 4.1 PPO：Proximal Policy Optimization

### 核心思想
限制策略更新幅度，稳定地提升策略收益。

### Agentic RL 用途
- RLHF 后训练
- Web Agent 任务训练
- Tool-use Agent 训练
- 多轮对话 Agent 训练
- 游戏/文本环境 Agent 训练

### 优点
- 稳定
- 工程成熟
- 支持复杂奖励

### 缺点
- 训练成本高
- 需要 Reward Model 或环境奖励
- 对超参数敏感

---

## 4.2 PPO-Clip

### 核心思想
使用 clipped objective 限制策略更新。

### 用途
- 标准 RLHF
- Agent 多步轨迹优化
- 长任务执行优化

---

## 4.3 PPO-KL

### 核心思想
使用 KL 惩罚约束新策略不要偏离参考模型太远。

### 用途
- 防止大模型语言能力退化
- 防止 Agent 过度奖励黑客
- 保持生成质量和安全性

---

## 4.4 REINFORCE

### 核心思想
最基础的蒙特卡洛策略梯度算法。

### 用途
- 可验证奖励训练
- 数学/代码任务训练
- 简化版 RLHF
- 长轨迹稀疏奖励任务

---

## 4.5 REINFORCE++

### 核心思想
对 REINFORCE 进行工程稳定化改进，如 baseline、归一化、KL 控制等。

### 用途
- 大模型推理 RL
- Agent 轨迹级奖励优化
- 低资源 RL 后训练

---

## 4.6 RLOO：REINFORCE Leave-One-Out

### 核心思想
用多个采样回答互作 baseline，减少方差。

### 用途
- 多样本响应优化
- 数学推理
- 代码生成
- 可验证奖励任务

---

## 4.7 A2C / A3C

### 核心思想
Actor-Critic 框架，Actor 负责行动，Critic 估计价值。

### 用途
- 交互环境 Agent
- 游戏 Agent
- 机器人 Agent
- 多智能体 RL

---

## 4.8 TRPO

### 核心思想
通过信赖域约束策略更新。

### 用途
- 理论稳定性更强的策略优化
- 机器人控制
- 早期深度 RL Agent

---

# 5. GRPO 与组相对优化类 Agentic RL

## 5.1 GRPO：Group Relative Policy Optimization

### 核心思想
对同一问题采样一组输出，用组内相对奖励替代显式价值函数。

### 特点
- 不一定需要单独 Critic
- 适合可验证奖励
- 适合推理模型训练
- 适合多样本比较

### 应用
- 数学推理
- 代码推理
- 长链推理
- Tool-use Agent
- Web Agent 多轨迹优化

---

## 5.2 Dr.GRPO

### 核心思想
对 GRPO 的长度偏差、奖励归一化、稳定性进行改进。

### 用途
- 减少过长输出偏好
- 提升长链推理稳定性
- Agent 多步轨迹优化

---

## 5.3 DAPO：Decoupled Clip and Dynamic sAmpling Policy Optimization

### 核心思想
改进 GRPO/PPO 类策略优化中的采样与裁剪机制。

### 用途
- 大规模推理模型 RL
- Agent 多样本轨迹优化
- 长上下文推理 Agent

---

## 5.4 GSPO：Group Sequence Policy Optimization

### 核心思想
以序列级别而非 token 级别进行组相对策略优化。

### 用途
- 长输出任务
- 代码生成
- 多步 Agent 轨迹优化
- 复杂推理任务

---

## 5.5 BNPO / BAPO

### 核心思想
通过批归一化、优势归一化等方式提升组策略优化稳定性。

### 用途
- 多样本 RL
- 稀疏奖励环境
- 可验证任务训练

---

## 5.6 M-GRPO：Multi-turn GRPO

### 核心思想
将 GRPO 扩展到多轮交互、多步工具调用和 Web Agent 环境。

### 用途
- Web Agent
- Browser Agent
- GUI Agent
- Multi-turn Tool Agent
- 长程任务 Agent

---

# 6. 可验证奖励类 Agentic RL

可验证奖励是当前 Agentic RL 最重要的方向之一，尤其适合数学、代码、数据库、工具调用等任务。

## 6.1 RLVR：Reinforcement Learning with Verifiable Rewards

### 核心思想
使用可以自动验证的答案或行为作为奖励。

### 奖励来源
- 数学答案是否正确
- 单元测试是否通过
- SQL 查询是否正确
- 工具调用结果是否满足条件
- 网页任务是否完成
- 文件操作是否正确
- API 调用是否成功

### 代表算法
- PPO + Verifiable Reward
- GRPO + Verifiable Reward
- REINFORCE + Verifiable Reward
- RLOO + Verifiable Reward
- Outcome Reward RL
- Unit-test Reward RL
- Execution Reward RL

---

## 6.2 Outcome Reward RL

### 核心思想
只根据最终结果是否正确给奖励。

### 优点
- 简单
- 容易自动化
- 适合答案可验证任务

### 缺点
- 奖励稀疏
- 过程不可控
- 容易奖励黑客

### 应用
- 数学题
- 代码题
- Text-to-SQL
- 检索问答
- 网页任务

---

## 6.3 Process Reward RL

### 核心思想
对中间推理步骤、工具调用步骤、规划步骤进行奖励。

### 代表方法
- Process Reward Model, PRM
- Step-level Reward
- Chain-of-Thought Reward
- Tool-step Reward
- Plan-step Reward
- Subgoal Reward

### 应用
- 数学推理
- 多步规划
- Web Agent
- 机器人任务
- 科研 Agent

---

## 6.4 Dense Reward Agent RL

### 核心思想
在任务执行过程中给多个中间奖励，缓解稀疏奖励问题。

### 奖励设计
- 完成子任务 +1
- 正确调用工具 +1
- 避免无效动作 +1
- 找到关键信息 +1
- 最终成功 +N
- 错误操作惩罚

### 应用
- 浏览器任务
- 游戏任务
- GUI 操作
- RAG Agent
- 工作流自动化

---

## 6.5 Unit-Test RL

### 核心思想
用单元测试通过率作为奖励。

### 应用
- Code Agent
- 编程竞赛
- 函数补全
- 自动修复 Bug
- 软件工程 Agent

### 常见奖励
- 编译成功
- 单元测试通过数量
- 隐藏测试通过率
- 运行时间限制
- 内存限制
- 代码风格评分

---

## 6.6 SQL Execution RL

### 核心思想
用 SQL 执行结果是否正确作为奖励。

### 应用
- Text-to-SQL Agent
- 数据库查询 Agent
- BI 分析 Agent
- 企业数据问答 Agent

### 代表奖励
- 查询语法正确
- 执行成功
- 返回结果匹配
- 查询代价较低
- 无危险操作

---

## 6.7 Tool-call Verification RL

### 核心思想
对工具调用参数、调用顺序、调用结果进行验证奖励。

### 应用
- API Agent
- Function Calling Agent
- RAG Agent
- 数据分析 Agent
- 自动办公 Agent

### 奖励维度
- 工具选择是否正确
- 参数是否正确
- 调用次数是否合理
- 是否成功利用返回结果
- 是否避免无效调用

---

# 7. 工具使用类 Agentic RL

## 7.1 Tool-use RL

### 核心思想
通过 RL 训练 Agent 学会何时调用工具、调用哪个工具、如何使用工具结果。

### 代表方向
- Function Calling RL
- Toolformer-style Tool Learning
- API-use RL
- Calculator-use RL
- Search-use RL
- Code Interpreter-use RL

### 应用
- 搜索问答
- 数据分析
- 代码执行
- 数学计算
- 自动办公
- 企业 API 工作流

---

## 7.2 ReAct + RL

### 核心思想
将 Reasoning + Acting 的轨迹作为强化学习对象。

### 训练对象
- Thought
- Action
- Observation
- Final Answer

### 可优化内容
- 思考质量
- 动作选择
- 工具调用顺序
- 对观察结果的利用
- 最终答案质量

---

## 7.3 Toolformer 类自监督工具学习

### 核心思想
模型通过自监督数据学习插入工具调用。

### Agentic RL 扩展
- 对工具调用结果给奖励
- 对错误调用给惩罚
- 对工具节省 token 给予奖励
- 对工具增强准确率给予奖励

---

## 7.4 API-call RL

### 核心思想
训练模型调用结构化 API 完成任务。

### 应用
- 订票
- 购物
- 企业系统
- CRM
- ERP
- 数据库
- 日程管理
- 邮件处理

---

## 7.5 Search Agent RL

### 核心思想
训练 Agent 何时搜索、搜索什么、如何整合搜索结果。

### 奖励
- 找到相关证据
- 引用正确来源
- 减少无效搜索
- 回答准确
- 避免编造

### 应用
- 搜索增强问答
- Deep Research
- 科研综述
- 法律检索
- 医疗文献检索

---

## 7.6 RAG Agent RL

### 核心思想
用 RL 优化检索、阅读、重排、引用和答案生成全过程。

### 可训练模块
- Query Rewriter
- Retriever
- Reranker
- Reader
- Answer Generator
- Citation Selector
- Tool Router

### 奖励
- 检索命中率
- 引用准确率
- 答案正确性
- 忠实性
- 覆盖率
- 简洁性

---

# 8. Web / Browser / GUI Agentic RL

## 8.1 Web Agent RL

### 核心思想
训练 Agent 在网页环境中点击、输入、滚动、搜索、提交表单完成任务。

### 代表环境
- WebArena
- WebArena-Lite
- MiniWoB++
- Mind2Web
- WorkArena
- BrowserGym
- WebShop
- WebVoyager

### 代表方法
- WebRL
- WebAgent-R1
- M-GRPO for Web Agents
- Online Curriculum RL for Web Agents
- Sparse Binary Reward RL
- Browser Trajectory RL

---

## 8.2 GUI Agent RL

### 核心思想
训练 Agent 操作图形界面。

### 动作空间
- 点击
- 拖拽
- 输入
- 滚动
- 选择菜单
- 文件操作
- 窗口切换

### 代表方向
- Computer-use RL
- OS Agent RL
- Mobile Agent RL
- Desktop Automation RL
- GUI Grounding RL

### 奖励
- 任务是否完成
- 点击是否命中目标
- 操作是否安全
- 步数是否少
- 是否避免破坏性操作

---

## 8.3 Browser-use RL

### 核心思想
训练模型使用浏览器完成真实任务。

### 应用
- 购物比价
- 表单填写
- 旅行规划
- 资料收集
- 在线办公
- 网站测试
- 数据抓取

---

## 8.4 UI Grounding RL

### 核心思想
训练模型把自然语言意图定位到界面元素。

### 奖励
- 元素定位正确
- 坐标点击正确
- OCR 识别正确
- 多模态理解正确

---

## 8.5 End-to-End Multi-turn Web RL

### 核心思想
从任务开始到完成进行端到端多轮 RL，而不是单步行为克隆。

### 代表特点
- 稀疏奖励
- 长轨迹
- 多次工具调用
- 真实交互
- 自我纠错

---

# 9. 代码智能体 Agentic RL

## 9.1 Code RL

### 核心思想
用编译、执行、测试反馈训练代码生成模型。

### 代表奖励
- 编译成功
- 单测通过
- 隐藏测试通过
- 运行效率
- 代码简洁
- 安全性

---

## 9.2 Code Repair RL

### 核心思想
训练 Agent 自动定位并修复代码错误。

### 奖励
- Bug 修复成功
- 测试通过
- 修改最小化
- 无新错误
- 可读性提升

---

## 9.3 SWE-Agent RL

### 核心思想
训练 Agent 在真实软件仓库中阅读代码、修改文件、运行测试、提交补丁。

### 应用
- GitHub Issue 修复
- 代码重构
- 单测生成
- CI 错误修复
- 依赖升级

### 奖励
- Issue resolved
- 测试通过
- patch accepted
- 无破坏性改动

---

## 9.4 Program-of-Thought RL

### 核心思想
训练模型生成可执行程序作为推理过程。

### 应用
- 数学题
- 逻辑题
- 数据分析
- 科学计算

---

## 9.5 Execution-guided RL

### 核心思想
利用代码执行结果引导策略更新。

### 应用
- Python Agent
- SQL Agent
- Notebook Agent
- 数据科学 Agent

---

# 10. 推理类 Agentic RL

## 10.1 Reasoning RL

### 核心思想
通过 RL 提升模型长链推理能力。

### 代表方向
- Math RL
- Code RL
- Logic RL
- Scientific Reasoning RL
- Chain-of-Thought RL
- Long CoT RL

---

## 10.2 Chain-of-Thought RL

### 核心思想
对思维链生成过程或最终答案进行奖励优化。

### 奖励类型
- 最终答案正确
- 中间步骤正确
- 推理简洁
- 推理一致
- 避免自相矛盾

---

## 10.3 Process-supervised RL

### 核心思想
结合过程监督与强化学习优化每一步推理。

### 应用
- 数学证明
- 复杂问答
- 代码推理
- 科学推理

---

## 10.4 Self-Consistency RL

### 核心思想
采样多条推理路径，通过一致性与正确性信号优化模型。

### 用途
- 多路径推理
- 答案投票
- 降低随机错误

---

## 10.5 Reflection RL

### 核心思想
训练 Agent 反思自己的错误并修正行动。

### 代表机制
- Self-reflection
- Reflexion
- Verbal Reinforcement Learning
- Critique-Revise RL
- Error Memory RL

---

## 10.6 Planning RL

### 核心思想
强化训练模型制定、修正、执行计划。

### 训练目标
- 计划合理性
- 子目标分解
- 长程依赖
- 计划执行一致性
- 动态重规划

---

## 10.7 Tree-of-Thought RL

### 核心思想
对树状推理搜索过程进行奖励优化。

### 相关方法
- Tree-of-Thought
- Graph-of-Thought
- Monte Carlo Tree Search with LLM
- LLM-MCTS
- Search-guided RL

---

## 10.8 RLTR：Reinforcement Learning for Tool-use Reasoning / Trajectory Reasoning

### 核心思想
将规划、工具使用和答案生成解耦，对行动轨迹质量进行奖励。

### 用途
- 多步工具调用
- 长程任务规划
- Agent 动作序列优化

---

# 11. 规划与搜索类 Agentic RL

## 11.1 MCTS-Agent RL

### 核心思想
结合蒙特卡洛树搜索与语言模型策略。

### 应用
- 代码生成
- 数学推理
- 游戏 Agent
- Web 任务规划
- 多步决策

---

## 11.2 AlphaZero-style Agent RL

### 核心思想
结合自博弈、价值网络、策略网络与搜索。

### 应用
- 游戏
- 定理证明
- 程序合成
- 组合优化

---

## 11.3 Model-based Agentic RL

### 核心思想
训练或使用世界模型预测行动后果。

### 代表方法
- World Model Agent
- Latent Dynamics Agent
- Planning with Learned Model
- Model Predictive Control Agent
- Imagination Rollout Agent

---

## 11.4 Hierarchical Planning RL

### 核心思想
高层策略负责目标和计划，低层策略负责动作执行。

### 结构
- Manager-Worker
- Planner-Executor
- Goal-Conditioned Policy
- Options Framework
- Skills Library

---

## 11.5 Subgoal RL

### 核心思想
将长任务拆解为子目标，每个子目标设置奖励。

### 应用
- Web Agent
- 游戏 Agent
- 机器人 Agent
- 复杂办公任务
- 科研任务

---

# 12. 记忆增强类 Agentic RL

## 12.1 Memory-augmented RL

### 核心思想
训练 Agent 学会写入、检索和使用长期记忆。

### 记忆类型
- Episodic Memory
- Semantic Memory
- Skill Memory
- Tool Memory
- Error Memory
- User Preference Memory

---

## 12.2 Retrieval Memory RL

### 核心思想
用 RL 优化记忆检索策略。

### 奖励
- 检索是否相关
- 是否减少重复错误
- 是否提升任务成功率
- 是否降低 token 成本

---

## 12.3 Experience Replay for LLM Agents

### 核心思想
保存成功和失败轨迹，用于后续训练。

### 方法
- Successful Trajectory Replay
- Failure Replay
- Prioritized Agent Replay
- Self-improvement Replay

---

## 12.4 Lifelong Agent RL

### 核心思想
Agent 在长期使用中持续学习和改进。

### 挑战
- 灾难性遗忘
- 用户隐私
- 安全边界
- 稳定性
- 版本回滚

---

# 13. 多智能体 Agentic RL

## 13.1 Multi-Agent RL for LLM Agents

### 核心思想
多个 Agent 在共享环境中协作或竞争，通过 RL 提升整体表现。

### 代表场景
- 多 Agent 辩论
- 多 Agent 编程
- 多 Agent 搜索
- 多 Agent 任务分工
- 多 Agent 模拟社会

---

## 13.2 Cooperative Agent RL

### 核心思想
多个 Agent 共享团队奖励。

### 应用
- 软件工程团队 Agent
- 科研团队 Agent
- 企业流程自动化
- 多机器人系统

---

## 13.3 Competitive Agent RL

### 核心思想
Agent 在博弈环境中学习策略。

### 应用
- 谈判 Agent
- 安全攻防
- 游戏
- 市场模拟
- 广告竞价

---

## 13.4 Debate RL

### 核心思想
通过多个模型辩论产生更优答案，并用评判器给奖励。

### 用途
- 复杂问答
- 安全审查
- 法律分析
- 科研推理

---

## 13.5 Self-play Agent RL

### 核心思想
Agent 与自身或历史版本对抗训练。

### 应用
- 游戏
- 谈判
- 网络安全
- 市场策略
- 自动红队

---

## 13.6 Role-based Multi-Agent RL

### 核心思想
不同 Agent 扮演 Planner、Executor、Critic、Researcher、Coder、Tester 等角色。

### 应用
- 软件工程
- 数据分析
- 自动研究
- 多步骤决策

---

# 14. 具身智能与机器人 Agentic RL

## 14.1 Embodied Agent RL

### 核心思想
训练 Agent 在物理或模拟环境中感知、规划、行动。

### 应用
- 机器人控制
- 家庭机器人
- 移动机器人
- 自动驾驶
- 仿真环境任务

---

## 14.2 Vision-Language-Action RL

### 核心思想
结合视觉、语言和动作空间的强化学习。

### 代表方向
- VLA Model RL
- Robotic Instruction Following RL
- Multimodal Policy RL
- Language-conditioned Control RL

---

## 14.3 Imitation + RL for Agents

### 核心思想
先用人类/专家示范行为克隆，再用 RL 提升表现。

### 方法
- Behavior Cloning
- DAgger
- GAIL
- Offline Imitation + Online RL
- Demonstration-augmented RL

---

## 14.4 Skill Learning RL

### 核心思想
学习可复用技能，再组合完成复杂任务。

### 技能类型
- 抓取
- 导航
- 放置
- 点击
- 搜索
- 编码
- 查询
- 规划

---

# 15. 离线与批量 Agentic RL

## 15.1 Offline Agent RL

### 核心思想
只使用已有轨迹数据训练，不在线探索。

### 应用
- 企业日志
- 人类操作记录
- Web 操作历史
- 客服对话历史
- 代码修改历史

---

## 15.2 Batch RL for Agent Trajectories

### 核心思想
用批量轨迹学习策略。

### 方法
- Conservative Q-Learning
- Implicit Q-Learning
- Behavior Regularized RL
- Offline Actor-Critic
- Decision Transformer

---

## 15.3 Decision Transformer for Agents

### 核心思想
把 RL 轨迹建模成序列建模问题。

### 应用
- 多步任务轨迹
- 工具调用序列
- 机器人行为
- 游戏行为
- Web 操作行为

---

## 15.4 Trajectory Preference Learning

### 核心思想
对整条 Agent 轨迹做偏好学习。

### 用途
- Web Agent
- Tool Agent
- RAG Agent
- Multi-step Reasoning Agent

---

# 16. 层次强化学习类 Agentic RL

## 16.1 Hierarchical RL

### 核心思想
将复杂任务拆解为高层目标和低层动作。

### 结构
- High-level Policy
- Low-level Policy
- Options
- Skills
- Subgoals

---

## 16.2 Options Framework

### 核心思想
把可持续执行的一段策略封装为 option。

### Agent 应用
- 搜索 option
- 编码 option
- 调试 option
- 总结 option
- 调用工具 option

---

## 16.3 Manager-Worker Agent RL

### 核心思想
Manager 分配任务，Worker 执行动作。

### 应用
- 多工具 Agent
- 自动办公
- 软件工程 Agent
- 机器人系统

---

## 16.4 Skill Discovery RL

### 核心思想
自动发现可复用技能。

### 方法
- Diversity-driven Skill Discovery
- DIAYN
- Option Discovery
- Unsupervised Skill Learning

---

# 17. 自改进与自训练 Agentic RL

## 17.1 Self-Improvement RL

### 核心思想
Agent 通过自生成数据、自评估和再训练不断提升。

### 流程
1. 生成任务
2. 执行任务
3. 自动评价
4. 筛选成功轨迹
5. 训练模型
6. 迭代升级

---

## 17.2 Self-Play Fine-tuning

### 核心思想
通过自我对抗或自我合作产生训练信号。

### 应用
- 数学
- 编程
- 谈判
- 安全红队
- 游戏

---

## 17.3 Self-Critique RL

### 核心思想
模型先生成答案，再批判答案，再根据批判进行优化。

### 用途
- 长文生成
- 推理
- 代码
- 工具调用

---

## 17.4 Self-Evolving Curriculum RL

### 核心思想
Agent 自动生成由易到难的课程任务。

### 应用
- Web Agent
- 数学推理
- 代码 Agent
- 游戏 Agent
- 机器人 Agent

---

## 17.5 Reflexion-style Verbal RL

### 核心思想
用自然语言反思作为强化信号或记忆。

### 特点
- 不一定更新模型参数
- 可结合经验记忆
- 可用于在线 Agent 改进

---

# 18. 安全、约束与对齐类 Agentic RL

## 18.1 Safe Agentic RL

### 核心思想
在优化任务成功率的同时满足安全约束。

### 约束
- 不执行危险命令
- 不泄露隐私
- 不越权调用工具
- 不进行有害操作
- 不绕过系统规则

---

## 18.2 Constrained Policy Optimization

### 核心思想
在策略优化中加入约束。

### 方法
- CPO
- Lagrangian RL
- Constrained PPO
- Shielded RL
- Safe Exploration RL

---

## 18.3 Rule-based Reward + RL

### 核心思想
将规则检查器作为奖励或惩罚信号。

### 应用
- 安全工具调用
- 企业 Agent
- 合规问答
- 金融/医疗/法律 Agent

---

## 18.4 Red-Team RL

### 核心思想
用 RL 训练攻击者或防御者 Agent。

### 应用
- 模型安全测试
- Jailbreak 检测
- 网络安全
- 自动红队

---

## 18.5 Uncertainty-aware Agent RL

### 核心思想
让 Agent 在不确定时拒答、请求确认或调用工具。

### 奖励
- 正确拒答
- 正确请求用户确认
- 避免高风险猜测
- 降低幻觉

---

# 19. 课程学习与环境生成类 Agentic RL

## 19.1 Curriculum RL

### 核心思想
从简单任务逐步训练到复杂任务。

### 应用
- Web Agent
- 机器人
- 数学推理
- 编程
- 游戏

---

## 19.2 Automatic Curriculum Learning

### 核心思想
系统自动选择最适合当前模型能力的任务。

### 方法
- Self-paced Learning
- Teacher-Student Curriculum
- Difficulty-aware Sampling
- Success-rate Based Sampling

---

## 19.3 Environment Generation RL

### 核心思想
自动生成训练环境和任务。

### 应用
- Web 任务模拟
- 游戏地图生成
- 代码题生成
- 数学题生成
- GUI 任务生成

---

## 19.4 Adversarial Task Generation

### 核心思想
生成更难任务逼迫 Agent 提升。

### 应用
- 安全红队
- 鲁棒性训练
- 长程规划
- 多智能体博弈

---

# 20. Agentic RL 工程框架与系统方法

## 20.1 Agent Lightning / LightningRL

### 核心思想
将 Agent 执行和 RL 训练解耦，把任意 Agent 执行轨迹抽象为 MDP，并进行分层信用分配。

### 适用场景
- LangChain Agent
- OpenAI Agents SDK Agent
- AutoGen Agent
- 自定义 Agent
- 多 Agent 工作流
- Text-to-SQL
- RAG
- 工具调用

---

## 20.2 OpenRLHF

### 核心用途
大模型 RLHF/RLAIF/RLVR 工程训练框架。

### 支持方向
- PPO
- DPO
- Reward Modeling
- SFT
- RLHF
- 大规模分布式训练

---

## 20.3 Hugging Face TRL

### 核心用途
Transformer 后训练与强化学习库。

### 常见算法
- SFT
- PPO
- DPO
- GRPO
- Reward Modeling
- Online DPO
- ORPO
- KTO
- CPO

---

## 20.4 VERL

### 核心用途
面向大模型 RL 后训练的分布式框架。

### 应用
- PPO
- GRPO
- RLVR
- 推理模型训练
- 多节点训练

---

## 20.5 SkyRL

### 核心用途
大规模 RL 后训练与推理强化学习实验框架。

---

## 20.6 AgentGym / AgentBoard / AgentBench

### 核心用途
Agent 训练与评测环境集合。

### 覆盖任务
- Web
- Tool-use
- Embodied
- Game
- Database
- OS
- Code

---

## 20.7 BrowserGym / WebArena / WorkArena

### 核心用途
Web Agent 训练和评测环境。

---

## 20.8 SWE-bench / SWE-agent 环境

### 核心用途
软件工程 Agent 训练与评测。

---

## 20.9 TextArena / TextWorld / ALFWorld

### 核心用途
文本环境中的多步 Agent 强化学习。

---

# 21. 按应用场景分类

## 21.1 数学 Agentic RL

### 算法类别
- RLVR
- GRPO
- PPO
- REINFORCE
- Process Reward RL
- Chain-of-Thought RL
- Self-Consistency RL

### 奖励
- 答案正确
- 推理步骤正确
- 格式正确
- 证明有效

---

## 21.2 编程 Agentic RL

### 算法类别
- Unit-Test RL
- Execution-guided RL
- Code Repair RL
- SWE-Agent RL
- DPO for Code
- GRPO for Code
- PPO for Code

---

## 21.3 Web Agentic RL

### 算法类别
- WebRL
- WebAgent-R1
- Browser Trajectory RL
- M-GRPO
- Sparse Reward RL
- Curriculum Web RL

---

## 21.4 RAG Agentic RL

### 算法类别
- Retrieval RL
- Reranking RL
- Citation Reward RL
- Answer Faithfulness RL
- Tool-use RL
- Query Rewrite RL

---

## 21.5 数据分析 Agentic RL

### 算法类别
- SQL Execution RL
- Python Execution RL
- Notebook Agent RL
- Tool-call Verification RL
- BI Agent RL

---

## 21.6 GUI / Computer-use Agentic RL

### 算法类别
- GUI Grounding RL
- Computer-use RL
- Vision-Language-Action RL
- Browser-use RL
- UI Action RL

---

## 21.7 机器人 Agentic RL

### 算法类别
- Embodied RL
- Imitation + RL
- Skill Learning RL
- Hierarchical RL
- Model-based RL
- Vision-Language-Action RL

---

## 21.8 多智能体协作 Agentic RL

### 算法类别
- Cooperative MARL
- Debate RL
- Role-based Multi-Agent RL
- Self-play RL
- Team Reward RL
- Credit Assignment RL

---

## 21.9 企业工作流 Agentic RL

### 算法类别
- Tool-call RL
- API-use RL
- Workflow RL
- Safe RL
- Human-in-the-loop RL
- Constrained Agent RL

---

# 22. 按奖励设计分类

## 22.1 人类偏好奖励
- RLHF
- PPO-RLHF
- Human Preference Reward Model
- Ranking Reward Model

## 22.2 AI 偏好奖励
- RLAIF
- Constitutional AI
- LLM-as-a-Judge
- Critic Model Reward

## 22.3 可验证奖励
- RLVR
- Unit-Test Reward
- SQL Execution Reward
- Math Answer Reward
- Tool-call Verification Reward

## 22.4 过程奖励
- PRM
- Step Reward
- Chain-of-Thought Reward
- Plan Reward
- Tool-step Reward

## 22.5 结果奖励
- ORM
- Task Success Reward
- Final Answer Reward
- Binary Success Reward

## 22.6 环境奖励
- Web Environment Reward
- Game Reward
- Robot Simulator Reward
- Browser Task Reward

## 22.7 安全奖励
- Safety Reward
- Refusal Reward
- Constraint Satisfaction Reward
- Policy Compliance Reward

## 22.8 成本奖励
- Token Cost Reward
- Tool Cost Reward
- Latency Reward
- Step Efficiency Reward

---

# 23. 按算法族总结

## 23.1 Policy Gradient 家族
- REINFORCE
- RLOO
- PPO
- TRPO
- A2C
- A3C
- GRPO
- DAPO
- GSPO
- Dr.GRPO

## 23.2 Preference Optimization 家族
- DPO
- IPO
- KTO
- ORPO
- SimPO
- CPO
- RRHF
- SLiC-HF
- BCO

## 23.3 Actor-Critic 家族
- A2C
- A3C
- PPO
- SAC
- DDPG
- TD3
- IMPALA
- V-trace

## 23.4 Value-based 家族
- Q-learning
- DQN
- Double DQN
- Dueling DQN
- Distributional DQN
- Conservative Q-Learning
- Implicit Q-Learning

## 23.5 Model-based 家族
- World Models
- MPC
- MuZero-style Planning
- Dreamer-style RL
- Imagination Rollouts
- Learned Dynamics Planning

## 23.6 Offline RL 家族
- Decision Transformer
- CQL
- IQL
- BCQ
- BRAC
- AWAC

## 23.7 Multi-agent RL 家族
- MADDPG
- QMIX
- VDN
- MAPPO
- COMA
- Self-play
- Debate RL

## 23.8 Hierarchical RL 家族
- Options
- Feudal RL
- Manager-Worker
- Goal-conditioned RL
- Skill Discovery
- Subgoal RL

---

# 24. Agentic RL 与普通 LLM RL 的区别

| 维度 | 普通 LLM RL | Agentic RL |
|---|---|---|
| 优化对象 | 单次回答 | 多步行动轨迹 |
| 环境 | 静态 prompt | 动态交互环境 |
| 动作 | token 生成 | token + 工具 + 点击 + API + 代码 |
| 奖励 | 偏好或答案 | 任务成功、过程、工具、环境、安全 |
| 轨迹长度 | 短 | 长 |
| 训练难点 | 语言质量 | 长程信用分配、探索、工具可靠性 |
| 典型方法 | PPO、DPO、GRPO | PPO/GRPO + 工具奖励 + 环境奖励 + 轨迹学习 |
| 应用 | 聊天、问答 | 浏览器、代码、RAG、机器人、企业自动化 |

---

# 25. 最核心的 Agentic RL 算法/方法清单

## 25.1 当前最重要
- RLHF
- RLAIF
- RLVR
- PPO
- GRPO
- DPO
- KTO
- ORPO
- SimPO
- REINFORCE
- RLOO
- Process Reward Model
- Outcome Reward Model
- Tool-use RL
- Web Agent RL
- Code Execution RL
- Multi-turn Agent RL
- Agent Lightning / LightningRL

## 25.2 推理模型常用
- GRPO
- PPO
- REINFORCE
- RLVR
- Process Reward RL
- Outcome Reward RL
- Long-CoT RL
- Self-Consistency RL
- DAPO
- Dr.GRPO

## 25.3 Web / GUI Agent 常用
- WebRL
- WebAgent-R1
- Browser Trajectory RL
- M-GRPO
- Sparse Binary Reward RL
- Curriculum RL
- GUI Grounding RL

## 25.4 Code Agent 常用
- Unit-Test RL
- Execution-guided RL
- Code Repair RL
- SWE-Agent RL
- DPO for Code
- GRPO for Code
- PPO for Code

## 25.5 Tool Agent 常用
- Tool-use RL
- Function-calling RL
- API-call RL
- Tool-call Verification RL
- ReAct + RL
- RAG Agent RL

---

# 26. 学习路线建议

## 第一阶段：大模型后训练基础
1. SFT
2. Reward Model
3. PPO
4. DPO
5. GRPO
6. RLHF/RLAIF/RLVR

## 第二阶段：Agent 基础
1. ReAct
2. Tool Calling
3. Function Calling
4. RAG Agent
5. Planner-Executor
6. Memory Agent

## 第三阶段：Agentic RL
1. 轨迹建模
2. 任务奖励设计
3. 工具调用奖励
4. 多步信用分配
5. Web/Code/SQL 环境训练
6. 多轮 GRPO/PPO

## 第四阶段：高级方向
1. 多智能体 RL
2. 具身智能 RL
3. 自进化课程学习
4. 安全约束 RL
5. Agent 持续学习
6. 真实企业工作流 Agent RL

---

# 27. 推荐优先掌握的 30 个关键词

1. Agentic RL
2. RLHF
3. RLAIF
4. RLVR
5. PPO
6. GRPO
7. DPO
8. KTO
9. ORPO
10. SimPO
11. REINFORCE
12. RLOO
13. Reward Model
14. Process Reward Model
15. Outcome Reward Model
16. Tool-use RL
17. Function Calling RL
18. Web Agent RL
19. Browser Agent
20. GUI Agent
21. Code Agent RL
22. Unit-Test Reward
23. SQL Execution Reward
24. ReAct
25. Reflexion
26. Multi-turn RL
27. Hierarchical RL
28. Multi-Agent RL
29. Agent Lightning
30. Curriculum RL

---

# 28. 简明总表

| 大类 | 代表算法/方法 | 主要用途 |
|---|---|---|
| RLHF/RLAIF | PPO-RLHF、Constitutional AI | 人类/AI 偏好对齐 |
| 直接偏好优化 | DPO、KTO、ORPO、SimPO | 低成本偏好训练 |
| 可验证奖励 | RLVR、Unit-Test RL、SQL Reward | 数学、代码、数据库 |
| 组相对优化 | GRPO、DAPO、GSPO、Dr.GRPO | 推理模型、Agent 轨迹 |
| 工具使用 RL | Tool-use RL、Function-call RL | API、搜索、计算器、RAG |
| Web Agent RL | WebRL、WebAgent-R1、M-GRPO | 浏览器任务 |
| Code Agent RL | Unit-Test RL、SWE-Agent RL | 编程、修复 Bug |
| 推理 RL | CoT RL、PRM、ORM | 数学、逻辑、科学推理 |
| 规划 RL | MCTS-Agent、Planning RL | 长程任务规划 |
| 记忆 RL | Memory-augmented RL | 长期个性化、经验复用 |
| 多智能体 RL | Debate RL、Self-play、MAPPO | 协作、博弈、团队 Agent |
| 具身 RL | VLA RL、Imitation+RL | 机器人、自动驾驶 |
| 安全 RL | Constrained RL、Red-Team RL | 安全、合规、拒答 |
| 自改进 RL | Self-play、Self-critique、Curriculum RL | 持续提升 |

---

# 29. 备注

Agentic RL 仍处于快速发展阶段，很多方法并没有统一命名。实际工程中，通常不是单独使用某个算法，而是组合使用：

- SFT + DPO + GRPO
- SFT + Reward Model + PPO
- ReAct + Tool Reward + GRPO
- Web 环境 + Sparse Reward + Curriculum RL
- Code Agent + Unit Test Reward + PPO/GRPO
- RAG Agent + Citation Reward + DPO/RLVR
- Multi-Agent Debate + LLM-as-Judge + Preference Optimization
- Planner-Executor + Process Reward + Outcome Reward

真正落地 Agentic RL 时，最关键的不是只选择算法，而是同时设计：

1. 状态表示
2. 动作空间
3. 环境接口
4. 轨迹格式
5. 奖励函数
6. 失败回放
7. 安全约束
8. 评测基准
9. 训练稳定性
10. 部署监控

---

# 30. 参考方向

- Reinforcement Learning from Human Feedback, RLHF
- Reinforcement Learning from AI Feedback, RLAIF
- Reinforcement Learning with Verifiable Rewards, RLVR
- Proximal Policy Optimization, PPO
- Group Relative Policy Optimization, GRPO
- Direct Preference Optimization, DPO
- Transformer Reinforcement Learning, TRL
- Agent Lightning / LightningRL
- WebArena / BrowserGym / WorkArena
- SWE-bench / SWE-agent
- ReAct / Reflexion / Toolformer
- Multi-agent Reinforcement Learning
- Hierarchical Reinforcement Learning
- Safe Reinforcement Learning
