# 大模型 Agent / RAG / MCP / A2A / Skill / Skill Graph / 自动化优化 Skill / MAS 架构设计总表

> 文件用途：作为大模型智能体系统的架构分类、设计框架、技术路线和工程落地参考。  
> 覆盖范围：LLM Agent、Agentic RAG、GraphRAG、MCP、A2A、Skill、Skill Graph、自动化 Skill 优化、多智能体系统 MAS、AgentOps、评测与优化。

---

## 0. 总体认知：这些概念之间的关系

可以把一个现代大模型 Agent 系统理解为以下层级：

```text
用户目标
  ↓
Agent 编排层
  ↓
规划 / 推理 / 任务分解
  ↓
RAG / 记忆 / 知识图谱 / 工具 / Skill / 外部服务
  ↓
MCP / A2A / API / 浏览器 / 文件系统 / 数据库 / 代码执行器
  ↓
执行、反馈、评测、优化
```

各概念的定位：

| 概念 | 主要作用 | 更像什么 |
|---|---|---|
| LLM Agent | 让大模型具备目标驱动、规划、工具调用、执行反馈能力 | 智能体大脑 |
| RAG | 给大模型接入外部知识，减少幻觉 | 知识增强系统 |
| Agentic RAG | 让 Agent 自主规划检索、验证、汇总 | 会思考的检索系统 |
| MCP | 统一连接工具、数据源、上下文服务 | Agent 的 USB-C / 工具协议 |
| A2A | Agent 与 Agent 之间通信、发现、协作 | Agent 间通信协议 |
| Skill | 面向任务的可复用能力包 | 插件 / 作业指导书 / 小型专家模块 |
| Skill Graph | Skill 之间的依赖图、组合图、能力路由图 | 能力知识图谱 |
| 自动化优化 Skill | 自动发现、生成、测试、修复、压缩、组合 Skill | Skill AutoML / SkillOps |
| MAS | Multi-Agent System，多智能体协作系统 | Agent 组织 / 团队 / 公司 |
| AgentOps | 监控、评测、回放、权限、安全、成本、上线治理 | Agent 运维体系 |

---

# 1. 大模型 Agent 架构分类

## 1.1 单体 Agent 架构

### 1.1.1 简单工具调用 Agent

结构：

```text
User
  ↓
LLM
  ↓
Tool Calling
  ↓
Tool Result
  ↓
LLM Final Answer
```

核心组件：

- LLM
- System Prompt
- Tool Schema
- Function Calling
- 工具返回解析器
- 最终回答生成器

适用场景：

- 查询数据库
- 调用天气、日历、邮件
- 简单文件处理
- 一步或少量步骤的任务

优点：

- 架构简单
- 延迟低
- 易调试

缺点：

- 长任务能力弱
- 缺少长期记忆
- 复杂任务容易失败

---

### 1.1.2 ReAct Agent

ReAct = Reasoning + Acting。

结构：

```text
Thought → Action → Observation → Thought → Action → Observation → Final
```

核心思路：

- 模型先思考
- 再调用工具
- 根据工具返回继续思考
- 循环直到完成任务

适用：

- 搜索问答
- 代码调试
- 多步骤资料整理
- API 调用链

优势：

- 可解释性较好
- 工具调用自然
- 适合交互式问题求解

风险：

- 循环失控
- 工具误用
- 中间推理污染
- Prompt Injection

---

### 1.1.3 Plan-and-Execute Agent

先规划，再执行。

结构：

```text
User Goal
  ↓
Planner 生成任务计划
  ↓
Executor 执行每一步
  ↓
Verifier 检查结果
  ↓
Final Answer
```

模块：

- Planner
- Executor
- Tool Router
- Memory
- Verifier
- Replanner

适用：

- 长任务
- 报告生成
- 软件开发
- 数据分析
- 自动化流程

优点：

- 任务结构清晰
- 容易加入检查点
- 支持重规划

缺点：

- 规划可能过粗或过细
- 初始计划错误会传播
- 需要较强状态管理

---

### 1.1.4 Reflexion Agent

加入自我反思机制。

结构：

```text
执行 → 失败/反馈 → 反思 → 修正策略 → 再执行
```

组件：

- Actor
- Evaluator
- Self-Reflection Memory
- Retry Controller
- Error Analyzer

适用：

- 代码修复
- 数学推理
- Web 自动化
- 长链工具调用
- Agent 自我改进

---

### 1.1.5 Self-Ask / Decomposition Agent

把复杂问题拆成子问题。

结构：

```text
Main Question
  ↓
Sub-question 1
Sub-question 2
Sub-question 3
  ↓
Evidence
  ↓
Synthesis
```

适用：

- 复杂问答
- 研究型搜索
- 多文档分析
- 事实核查

---

### 1.1.6 Tree-of-Thought Agent

用树搜索探索多个推理路径。

结构：

```text
Root Question
 ├─ Thought Path A
 ├─ Thought Path B
 └─ Thought Path C
      ↓
Scoring / Voting / Search
      ↓
Best Path
```

搜索策略：

- BFS
- DFS
- Beam Search
- MCTS
- Self-consistency
- Best-first Search

适用：

- 数学题
- 规划问题
- 复杂推理
- 创意生成
- 代码方案搜索

---

### 1.1.7 Graph-of-Thought Agent

把思路组织成图，而不是线或树。

适用：

- 多约束推理
- 复杂系统设计
- 跨文档归纳
- 方案比较
- 多来源证据融合

核心节点：

- Problem Node
- Hypothesis Node
- Evidence Node
- Tool Result Node
- Decision Node
- Conflict Node
- Conclusion Node

---

### 1.1.8 Memory-Augmented Agent

带长期记忆和短期记忆。

记忆类型：

- 工作记忆
- 会话记忆
- 长期用户记忆
- 任务记忆
- 工具使用记忆
- 错误经验记忆
- 偏好记忆
- 语义记忆
- 情节记忆
- 程序性记忆

结构：

```text
User Input
  ↓
Memory Retrieval
  ↓
Context Builder
  ↓
LLM Planning
  ↓
Tool / Skill Execution
  ↓
Memory Update
```

---

### 1.1.9 Code Agent

专门用于软件开发。

核心能力：

- 读仓库
- 理解依赖
- 修改代码
- 运行测试
- 修复错误
- 生成提交
- 代码审查
- 重构
- 文档生成

典型架构：

```text
Repo Scanner
  ↓
Task Planner
  ↓
Code Editor
  ↓
Test Runner
  ↓
Error Analyzer
  ↓
Patch Generator
  ↓
Verifier
```

---

### 1.1.10 Browser / GUI Agent

操作浏览器或桌面 UI。

能力：

- 页面理解
- DOM 分析
- 截图理解
- 点击、输入、滚动
- 表单填写
- 文件下载上传
- 多页面流程控制

架构：

```text
Vision / DOM Parser
  ↓
State Representation
  ↓
Action Planner
  ↓
Browser Controller
  ↓
Observation
  ↓
Verifier
```

---

### 1.1.11 Data Agent

专门处理数据。

能力：

- SQL 查询
- 表格理解
- 数据清洗
- 特征工程
- 可视化
- 建模
- 报告生成

架构：

```text
Data Source Connector
  ↓
Schema Understanding
  ↓
Query Planner
  ↓
Python / SQL Executor
  ↓
Result Interpreter
  ↓
Report Generator
```

---

### 1.1.12 Research Agent

用于资料研究、综述、调研。

能力：

- 搜索
- 文献阅读
- 引用管理
- 证据抽取
- 观点对比
- 时间线构建
- 综述生成

架构：

```text
Query Planner
  ↓
Retriever / Web Search
  ↓
Source Reader
  ↓
Evidence Extractor
  ↓
Claim Verifier
  ↓
Synthesis Writer
```

---

# 2. Agent 架构核心模块

## 2.1 感知层

输入来源：

- 用户文本
- 语音
- 图像
- 视频
- 文件
- 网页
- 数据库
- API
- 环境状态
- 其他 Agent 消息

模块：

- Input Parser
- OCR
- ASR
- Vision Encoder
- Document Parser
- Schema Extractor
- State Encoder

---

## 2.2 状态层

状态包括：

- 用户目标
- 当前任务
- 已完成步骤
- 待完成步骤
- 工具结果
- 约束条件
- 错误信息
- 预算信息
- 权限信息
- 上下文窗口状态

状态表示：

```json
{
  "goal": "...",
  "plan": [],
  "current_step": "...",
  "memory": {},
  "tools_available": [],
  "constraints": [],
  "evidence": [],
  "errors": []
}
```

---

## 2.3 规划层

规划方式：

- 零规划：直接回答
- 短计划：列出 3~5 步
- 长计划：任务树
- 动态规划：边做边改
- 层级规划：目标 → 子目标 → 动作
- 约束规划：带权限、成本、时间限制
- 搜索式规划：ToT / MCTS
- 反事实规划：比较多方案

---

## 2.4 工具层

工具类型：

- 搜索工具
- 浏览器工具
- 文件工具
- 数据库工具
- 代码执行工具
- Shell 工具
- 邮件工具
- 日历工具
- 画图工具
- API 工具
- RPA 工具
- MCP 工具
- A2A 远程 Agent

---

## 2.5 记忆层

设计原则：

- 短期记忆负责当前任务
- 长期记忆负责稳定偏好
- 任务记忆负责可复用经验
- 工具记忆负责工具使用经验
- 错误记忆负责失败模式
- 检索记忆负责语义召回

记忆写入策略：

- 显式写入
- 自动摘要写入
- 重要性评分写入
- 频率触发写入
- 成功经验写入
- 失败经验写入
- 用户授权写入

---

## 2.6 执行层

执行方式：

- 单步执行
- 多步执行
- 并行执行
- 队列执行
- 事务执行
- 回滚执行
- 沙箱执行
- 人类确认后执行

---

## 2.7 验证层

验证类型：

- 格式验证
- 事实验证
- 引用验证
- 工具结果验证
- 代码测试
- 单元测试
- 安全验证
- 权限验证
- 目标完成度验证
- 成本验证

---

## 2.8 反思与重规划层

触发条件：

- 工具调用失败
- 输出格式错误
- 证据不足
- 用户反馈不满意
- 测试失败
- 目标变化
- 超预算
- 安全风险

处理动作：

- 重试
- 换工具
- 拆任务
- 回退上一步
- 请求人工确认
- 修改计划
- 降级方案

---

# 3. RAG 架构和思路分类

## 3.1 Naive RAG

基础流程：

```text
Document → Chunk → Embedding → Vector DB
User Query → Retrieve → Prompt → LLM → Answer
```

组件：

- 文档解析
- 切块
- 向量化
- 向量数据库
- Top-K 检索
- Prompt 拼接
- 答案生成

适用：

- 简单知识库问答
- FAQ
- 企业文档问答

缺点：

- 检索粒度粗
- 上下文容易断裂
- 不擅长复杂推理
- 多跳问题效果差

---

## 3.2 Advanced RAG

增强点：

- Query Rewrite
- Query Expansion
- Hybrid Search
- Rerank
- Context Compression
- Multi-query Retrieval
- Metadata Filtering
- Parent-Child Chunk
- Sliding Window Chunk
- Sentence Window Retrieval

架构：

```text
Query
  ↓
Query Rewriter
  ↓
Hybrid Retriever
  ↓
Reranker
  ↓
Context Compressor
  ↓
LLM
```

---

## 3.3 Modular RAG

把 RAG 拆成可组合模块。

模块：

- Indexing Module
- Query Understanding Module
- Retrieval Module
- Rerank Module
- Evidence Selection Module
- Answer Generation Module
- Verification Module

优点：

- 便于工程调优
- 模块可替换
- 适合复杂业务

---

## 3.4 Agentic RAG

Agent 自主决定如何检索、检索几次、用哪些工具。

流程：

```text
User Question
  ↓
Agent 判断问题类型
  ↓
选择检索策略
  ↓
多轮检索 / 工具调用
  ↓
证据验证
  ↓
答案生成
```

核心能力：

- 自主查询规划
- 多跳检索
- 工具选择
- 证据验证
- 反思重检索
- 多源融合

类型：

- Single-Agent RAG
- Multi-Agent RAG
- Planner-Retriever RAG
- Critic-RAG
- Self-RAG
- Corrective RAG
- Adaptive RAG
- Routing RAG

---

## 3.5 Self-RAG

模型自己判断是否需要检索、检索结果是否有用、答案是否被支持。

核心节点：

- Need Retrieval?
- Retrieve
- Is Relevant?
- Is Supported?
- Generate
- Critique

---

## 3.6 Corrective RAG（CRAG）

当检索质量差时，自动纠错。

机制：

- 检索评估器
- 文档过滤
- Web 补充搜索
- 知识重写
- 答案修正

---

## 3.7 Adaptive RAG

根据问题动态选择策略。

问题类型：

- 简单事实问答：一次检索
- 多跳推理：多轮检索
- 复杂研究：Agentic RAG
- 数值计算：工具调用
- 专业知识：领域索引
- 最新信息：Web Search

---

## 3.8 GraphRAG

使用知识图谱增强检索。

核心流程：

```text
Documents
  ↓
Entity / Relation Extraction
  ↓
Knowledge Graph
  ↓
Community Detection
  ↓
Graph Retrieval
  ↓
Evidence Synthesis
```

适合：

- 多文档关系分析
- 企业知识库
- 法律/金融/医疗
- 长文档总结
- 复杂实体关系问答

GraphRAG 类型：

- Entity Graph RAG
- Knowledge Graph RAG
- Community Graph RAG
- Hybrid Vector + Graph RAG
- Path-based Graph RAG
- Subgraph Retrieval RAG
- Personalized PageRank RAG
- Temporal Graph RAG

---

## 3.9 TreeRAG

使用树状文档结构。

结构：

```text
Document
 ├─ Chapter Summary
 │   ├─ Section Summary
 │   │   ├─ Chunk
```

适合：

- 书籍
- 法规
- 技术文档
- 长报告
- 层级知识库

---

## 3.10 Multi-Vector RAG

一个文档生成多个向量表示。

类型：

- Dense Vector
- Sparse Vector
- Summary Vector
- Question Vector
- Entity Vector
- Table Vector
- Image Vector
- Code Vector

---

## 3.11 Hybrid RAG

融合多种检索方式：

- 向量检索
- BM25
- 关键词检索
- 图检索
- SQL 检索
- API 检索
- Web 检索
- 文件检索

典型结构：

```text
Query Router
 ├─ Vector Search
 ├─ Keyword Search
 ├─ Graph Search
 ├─ SQL Search
 └─ Web Search
      ↓
Fusion / Rerank
      ↓
Answer
```

---

## 3.12 Multimodal RAG

处理文本、图片、表格、音频、视频。

模块：

- 文本解析
- 图像 embedding
- OCR
- 表格结构化
- 视频切帧
- 音频转写
- 多模态向量库
- 多模态 reranker

---

## 3.13 SQL-RAG / Structured RAG

对结构化数据查询。

流程：

```text
Natural Language
  ↓
Schema Linking
  ↓
Text-to-SQL
  ↓
SQL Execution
  ↓
Result Explanation
```

增强点：

- SQL 验证
- 权限控制
- 结果解释
- 多表 join 规划
- Query Cost 控制

---

## 3.14 RAG 评测指标

检索指标：

- Recall@K
- Precision@K
- MRR
- NDCG
- Hit Rate
- Context Relevance

生成指标：

- Faithfulness
- Answer Relevance
- Citation Accuracy
- Hallucination Rate
- Completeness
- Conciseness

系统指标：

- Latency
- Cost
- Cache Hit Rate
- Tool Success Rate
- User Satisfaction

---

# 4. MCP 架构和思路分类

## 4.1 MCP 的定位

MCP = Model Context Protocol。

它的核心作用：

- 把模型与外部工具、数据源、服务统一连接
- 提供标准化上下文交换协议
- 让不同应用以统一方式暴露能力给 Agent

可以理解为：

```text
LLM Host
  ↓
MCP Client
  ↓
MCP Server
  ↓
Tools / Resources / Prompts / Data Sources
```

---

## 4.2 MCP 基础架构

核心组件：

| 组件 | 作用 |
|---|---|
| Host | 承载大模型的应用，如 IDE、聊天应用、Agent 平台 |
| MCP Client | Host 内部的协议客户端 |
| MCP Server | 暴露工具、资源、提示词的服务 |
| Tool | 可调用动作 |
| Resource | 可读取上下文资源 |
| Prompt | 可复用提示模板 |
| Transport | 通信层，如 stdio、HTTP、SSE、Streamable HTTP |
| JSON-RPC | 消息协议基础 |

架构：

```text
Host Application
 ├─ MCP Client A → MCP Server A → File System
 ├─ MCP Client B → MCP Server B → Database
 └─ MCP Client C → MCP Server C → SaaS API
```

---

## 4.3 MCP Server 类型

### 4.3.1 本地工具型 MCP

- 文件系统 MCP
- Shell MCP
- Git MCP
- SQLite MCP
- 浏览器 MCP
- 本地代码执行 MCP

### 4.3.2 远程服务型 MCP

- GitHub MCP
- Slack MCP
- Google Drive MCP
- Notion MCP
- Jira MCP
- Linear MCP
- 数据库 MCP
- 云服务 MCP

### 4.3.3 企业系统型 MCP

- CRM MCP
- ERP MCP
- BI MCP
- 数据仓库 MCP
- 日志系统 MCP
- 内部知识库 MCP
- 权限系统 MCP

### 4.3.4 Agent 工具型 MCP

- RAG MCP
- Browser MCP
- Code Runner MCP
- Workflow MCP
- Scheduler MCP
- Memory MCP
- Evaluator MCP

---

## 4.4 MCP 设计模式

### 4.4.1 Tool Gateway Pattern

一个 MCP Server 聚合多个内部 API。

```text
Agent → MCP Gateway → Internal APIs
```

适用：

- 企业内部系统统一接入
- 权限统一管理
- 日志统一记录

---

### 4.4.2 Resource Provider Pattern

MCP Server 只暴露资源，不执行危险动作。

适用：

- 文档读取
- 数据库只读
- 知识库检索

---

### 4.4.3 Action Executor Pattern

MCP Server 执行动作。

风险较高，需要：

- 权限检查
- 参数校验
- 审计日志
- 人类确认
- 沙箱隔离

---

### 4.4.4 Prompt Provider Pattern

MCP Server 提供可复用 Prompt。

适用：

- 企业写作规范
- 代码审查规范
- 数据分析模板
- 客服话术

---

### 4.4.5 Secure MCP Broker Pattern

加入中间代理做安全控制。

```text
Host → MCP Broker → Policy Engine → MCP Servers
```

能力：

- 鉴权
- 授权
- 审计
- 限流
- 脱敏
- 工具白名单
- 风险动作审批

---

## 4.5 MCP 安全架构

风险：

- Prompt Injection
- Tool Injection
- Data Exfiltration
- Over-permission
- Command Injection
- Supply Chain Risk
- Malicious MCP Server
- Confused Deputy Problem

防护：

- 工具最小权限
- 参数 schema 校验
- 用户确认高风险动作
- MCP Server 签名
- 工具白名单
- 输出脱敏
- 沙箱执行
- 审计日志
- 速率限制
- 上下文隔离
- 权限分层
- 远程服务认证

---

# 5. A2A 架构和思路分类

## 5.1 A2A 的定位

A2A = Agent-to-Agent Protocol。

核心作用：

- 让不同平台、不同框架、不同厂商的 Agent 互相发现、通信、委托任务
- 解决 Agent 之间互操作问题

与 MCP 的区别：

| 协议 | 主要对象 | 作用 |
|---|---|---|
| MCP | Agent ↔ 工具/数据/上下文 | 接工具 |
| A2A | Agent ↔ Agent | 接智能体 |
| API | 程序 ↔ 服务 | 接功能 |
| RAG | Agent ↔ 知识 | 接知识 |

---

## 5.2 A2A 基础架构

```text
Agent A
  ↓ discovers
Agent Card
  ↓ sends task
Agent B
  ↓ returns result / stream / artifact
Agent A
```

核心组件：

- Agent Card
- Agent Endpoint
- Capability Description
- Task
- Message
- Artifact
- Streaming Update
- Authentication
- Authorization

---

## 5.3 Agent Card 设计

Agent Card 用于描述 Agent 能力。

字段示例：

```json
{
  "name": "CodeReviewAgent",
  "description": "Reviews Python and TypeScript repositories",
  "capabilities": ["code_review", "test_generation", "security_audit"],
  "endpoint": "https://example.com/a2a",
  "auth": "oauth2",
  "input_modes": ["text", "file"],
  "output_modes": ["text", "patch", "report"]
}
```

---

## 5.4 A2A 通信模式

### 5.4.1 请求-响应

```text
Agent A → Task → Agent B → Result
```

适用：

- 简单委托
- 一次性任务

---

### 5.4.2 流式任务

```text
Agent A → Task
Agent B → Progress 1
Agent B → Progress 2
Agent B → Final
```

适用：

- 长任务
- 编程任务
- 研究任务

---

### 5.4.3 多轮协作

```text
Agent A ↔ Agent B ↔ Agent C
```

适用：

- 需求澄清
- 联合分析
- 多专家讨论

---

### 5.4.4 任务委托链

```text
Manager Agent
 ├─ Research Agent
 ├─ Coding Agent
 └─ QA Agent
```

适用：

- MAS
- 企业流程
- 自动开发系统

---

### 5.4.5 市场式 Agent 网络

```text
Task → Capability Match → Bid / Select → Execute → Evaluate
```

适用：

- Agent Marketplace
- 动态能力发现
- 企业 Agent 生态

---

## 5.5 A2A 安全设计

关键问题：

- 谁可以调用 Agent？
- Agent 能看到什么数据？
- Agent 能否再委托给第三方？
- 任务结果是否可信？
- 是否允许跨组织通信？

安全机制：

- Agent 身份认证
- Agent Card 签名
- 权限范围声明
- 数据作用域限制
- 调用审计
- 结果溯源
- 沙箱执行
- 委托链追踪
- 任务级授权
- 人类审批

---

# 6. Skill 架构和思路分类

## 6.1 Skill 的定位

Skill 是面向特定任务的可复用能力包。

典型 Skill 包含：

```text
skill-name/
 ├─ SKILL.md
 ├─ scripts/
 ├─ templates/
 ├─ examples/
 ├─ assets/
 └─ tests/
```

Skill 的作用：

- 降低 Prompt 重复
- 固化流程规范
- 封装工具调用
- 提供领域知识
- 提供代码脚本
- 提供模板
- 提高任务稳定性

---

## 6.2 Skill 类型分类

### 6.2.1 指令型 Skill

只包含任务说明和流程。

例子：

- 写周报 Skill
- 代码审查 Skill
- PRD 分析 Skill
- 会议纪要 Skill

---

### 6.2.2 模板型 Skill

包含可复用模板。

例子：

- PPT 模板 Skill
- DOCX 模板 Skill
- 简历模板 Skill
- 项目计划模板 Skill
- API 文档模板 Skill

---

### 6.2.3 脚本型 Skill

包含可执行脚本。

例子：

- PDF 解析 Skill
- Excel 处理 Skill
- 图片压缩 Skill
- 代码扫描 Skill
- 数据清洗 Skill

---

### 6.2.4 工具封装型 Skill

封装外部工具或 API。

例子：

- GitHub Skill
- Jira Skill
- Notion Skill
- 数据库 Skill
- 浏览器 Skill
- 云服务 Skill

---

### 6.2.5 领域专家型 Skill

封装专业知识和判断标准。

例子：

- 法律合同审查 Skill
- 医疗文献分析 Skill
- 金融风控 Skill
- 机器学习建模 Skill
- 强化学习实验 Skill

---

### 6.2.6 工作流型 Skill

封装多步骤流程。

例子：

- 从需求到代码 Skill
- 从论文到综述 Skill
- 从日志到故障定位 Skill
- 从数据到报告 Skill
- 从产品想法到 PRD Skill

---

### 6.2.7 评测型 Skill

专门做验证和评分。

例子：

- 代码质量评分 Skill
- RAG 答案可信度评分 Skill
- Prompt 质量评分 Skill
- 安全审查 Skill
- 单元测试生成 Skill

---

## 6.3 Skill 设计标准

一个高质量 Skill 应包含：

| 部分 | 内容 |
|---|---|
| Name | 技能名 |
| Purpose | 解决什么问题 |
| When to use | 什么时候触发 |
| Inputs | 需要哪些输入 |
| Workflow | 执行步骤 |
| Tools | 需要哪些工具 |
| Constraints | 约束和边界 |
| Output format | 输出格式 |
| Examples | 示例 |
| Tests | 测试用例 |
| Failure handling | 失败处理 |
| Security | 权限与安全 |

---

## 6.4 Skill 执行流程

```text
User Task
  ↓
Skill Router
  ↓
Skill Selection
  ↓
Skill Context Load
  ↓
Tool / Script Execution
  ↓
Result Verification
  ↓
Output
```

---

## 6.5 Skill Router 设计

路由方式：

- 关键词匹配
- Embedding 相似度
- LLM 分类
- 规则匹配
- 工具可用性匹配
- 历史成功率匹配
- 成本/延迟匹配
- 多 Skill 组合搜索

---

# 7. Skill Graph 架构和思路分类

## 7.1 Skill Graph 的定位

Skill Graph 是把 Skill 组织成图结构：

```text
Skill A → Skill B → Skill C
```

节点是 Skill，边表示：

- 依赖关系
- 调用关系
- 前后置关系
- 数据流关系
- 替代关系
- 互补关系
- 冲突关系
- 版本关系

---

## 7.2 Skill Graph 类型

### 7.2.1 依赖图

```text
PDF Parser Skill → Text Extract Skill → Summary Skill
```

用于：

- 自动加载依赖
- 检查缺失能力
- 构建工作流

---

### 7.2.2 能力图

按能力组织。

```text
Document Processing
 ├─ PDF
 ├─ DOCX
 ├─ PPTX
 └─ Markdown
```

用于：

- 能力发现
- Skill 推荐
- Agent 规划

---

### 7.2.3 工作流图

表示任务执行流程。

```text
Requirement Analysis → Architecture Design → Coding → Testing → Report
```

用于：

- 自动流程编排
- 多 Agent 协作
- 企业 SOP 自动化

---

### 7.2.4 语义图

用 embedding 或知识图谱表示 Skill 相似性。

用于：

- 相似 Skill 查找
- Skill 聚类
- Skill 去重
- Skill 推荐

---

### 7.2.5 成功率图

边权由历史效果决定。

```text
Skill A --0.92--> Skill B
Skill A --0.55--> Skill C
```

用于：

- 自动选择最优组合
- 强化学习优化工作流
- A/B 测试

---

### 7.2.6 版本图

表示 Skill 的演化。

```text
Skill v1 → Skill v2 → Skill v3
```

用于：

- 回滚
- 灰度发布
- 效果对比

---

## 7.3 Skill Graph 数据结构

```json
{
  "nodes": [
    {
      "id": "pdf_extract",
      "name": "PDF Extract Skill",
      "capabilities": ["pdf", "ocr", "text_extraction"],
      "inputs": ["pdf"],
      "outputs": ["markdown"],
      "cost": 0.03,
      "latency": 5.2,
      "success_rate": 0.94
    }
  ],
  "edges": [
    {
      "from": "pdf_extract",
      "to": "summary",
      "type": "dataflow",
      "weight": 0.91
    }
  ]
}
```

---

## 7.4 Skill Graph 搜索算法

可用于选择 Skill 组合：

- BFS
- DFS
- Dijkstra
- A*
- Beam Search
- Monte Carlo Tree Search
- Topological Sort
- PageRank
- Personalized PageRank
- Graph Neural Network
- Reinforcement Learning
- Multi-objective Optimization

---

## 7.5 Skill Graph 优化目标

目标函数：

```text
maximize:
  task_success
  - cost
  - latency
  - risk
  + user_satisfaction
```

指标：

- 成功率
- 成本
- 延迟
- 工具失败率
- 用户满意度
- 可解释性
- 安全等级
- 可复用性
- 维护成本

---

# 8. 自动化优化 Skill 架构和思路

## 8.1 SkillOps 总体架构

```text
Skill Repository
  ↓
Skill Evaluator
  ↓
Failure Analyzer
  ↓
Skill Generator / Modifier
  ↓
Test Runner
  ↓
Benchmark
  ↓
Version Manager
  ↓
Deployment
```

---

## 8.2 自动生成 Skill

输入来源：

- 用户频繁任务
- 历史对话
- 工具调用日志
- 成功案例
- 失败案例
- 企业 SOP
- 文档规范
- 代码仓库

生成流程：

```text
Task Logs
  ↓
Pattern Mining
  ↓
Workflow Extraction
  ↓
Skill Draft
  ↓
Test Case Generation
  ↓
Human Review
  ↓
Publish
```

---

## 8.3 自动发现 Skill 需求

方法：

- 高频任务聚类
- 失败任务聚类
- 重复 Prompt 检测
- 工具调用链挖掘
- 用户反馈分析
- 延迟瓶颈分析
- 成本瓶颈分析

输出：

- 建议新增 Skill
- 建议拆分 Skill
- 建议合并 Skill
- 建议废弃 Skill
- 建议优化流程

---

## 8.4 自动评测 Skill

评测维度：

- 任务成功率
- 输出格式正确率
- 工具调用成功率
- 幻觉率
- 引用准确率
- 执行成本
- 执行时延
- 安全违规率
- 用户采纳率
- 回滚率

评测方式：

- 单元测试
- Golden Set
- LLM-as-Judge
- 人工评审
- 对抗测试
- 回放测试
- Shadow Mode
- A/B Test

---

## 8.5 自动修复 Skill

触发：

- 测试失败
- 工具接口变化
- 输出格式不稳定
- 用户差评
- 安全规则变化
- 成本异常

修复动作：

- 修改说明
- 增加约束
- 增加示例
- 修改脚本
- 替换工具
- 增加验证步骤
- 拆分 Skill
- 添加 fallback

---

## 8.6 自动压缩 Skill

目标：

- 降低上下文占用
- 提高加载速度
- 减少冗余说明
- 保留关键约束

方法：

- Prompt Distillation
- Instruction Compression
- Example Pruning
- Rule Deduplication
- Template Extraction
- Hierarchical Loading

---

## 8.7 自动组合 Skill

组合策略：

- 串行组合
- 并行组合
- 条件组合
- fallback 组合
- 投票组合
- planner 组合
- graph search 组合
- RL 优化组合

示例：

```text
需求分析 Skill
  → 架构设计 Skill
  → 代码生成 Skill
  → 测试 Skill
  → 文档 Skill
```

---

## 8.8 自动优化 Skill 的算法

可用算法：

- Prompt Optimization
- Bayesian Optimization
- Evolutionary Search
- Genetic Algorithm
- Monte Carlo Tree Search
- Reinforcement Learning
- Bandit Algorithm
- Program Synthesis
- Self-Refine
- Reflexion
- LLM-as-Optimizer
- DPO / Preference Optimization
- Failure-driven Optimization

---

# 9. MAS 多智能体系统架构

## 9.1 MAS 基础模式

MAS = Multi-Agent System。

核心问题：

- Agent 如何分工
- Agent 如何通信
- Agent 如何共享记忆
- Agent 如何冲突解决
- Agent 如何协作决策
- Agent 如何评测和治理

---

## 9.2 Manager-Worker 架构

```text
Manager Agent
 ├─ Worker Agent A
 ├─ Worker Agent B
 └─ Worker Agent C
```

适用：

- 软件开发
- 报告生成
- 企业流程
- 数据分析

优点：

- 简单清晰
- 易控
- 适合任务分解

缺点：

- Manager 成为瓶颈
- Worker 之间协作弱

---

## 9.3 Planner-Executor-Critic 架构

```text
Planner → Executor → Critic → Replanner
```

角色：

- Planner：制定计划
- Executor：执行任务
- Critic：检查错误
- Replanner：修正计划

适用：

- 高可靠任务
- 代码生成
- 自动研究
- 复杂分析

---

## 9.4 Debate / Discussion 架构

多个 Agent 辩论形成结论。

```text
Agent A opinion
Agent B opinion
Agent C opinion
  ↓
Judge Agent
  ↓
Final Decision
```

适用：

- 方案评审
- 风险分析
- 法律分析
- 战略决策
- 复杂推理

---

## 9.5 Blackboard 架构

所有 Agent 在共享黑板上读写。

```text
Shared Blackboard
 ├─ Research Agent writes evidence
 ├─ Analyst Agent writes insight
 ├─ Critic Agent writes issues
 └─ Writer Agent writes final
```

适用：

- 复杂协作
- 多源信息融合
- 科研系统
- 企业知识处理

---

## 9.6 Market / Auction 架构

任务发布后，Agent 竞标。

```text
Task → Bidding Agents → Selection → Execution → Evaluation
```

适用：

- 大规模 Agent 网络
- 动态任务分配
- 多供应商 Agent 生态

---

## 9.7 Swarm 架构

大量简单 Agent 群体协作。

特点：

- 无中心
- 局部规则
- 涌现行为
- 高鲁棒

适用：

- 搜索
- 优化
- 仿真
- 多机器人群体

---

## 9.8 Hierarchical MAS

层级化组织 Agent。

```text
Director Agent
 ├─ Product Manager Agent
 │   ├─ Requirement Agent
 │   └─ User Research Agent
 ├─ Engineer Manager Agent
 │   ├─ Backend Agent
 │   └─ Frontend Agent
 └─ QA Manager Agent
     ├─ Test Agent
     └─ Security Agent
```

适用：

- 企业级自动化
- 软件工程
- 大型项目管理
- 多部门流程

---

## 9.9 Role-Based MAS

按角色分配 Agent。

常见角色：

- Planner
- Researcher
- Coder
- Reviewer
- Tester
- Critic
- Writer
- Data Analyst
- Security Auditor
- Product Manager
- Project Manager
- Memory Manager
- Tool Manager

---

## 9.10 Graph-Based MAS

Agent 之间形成图结构。

```text
Agent A ↔ Agent B
   ↓        ↘
Agent C → Agent D
```

图边表示：

- 信息流
- 权限流
- 任务流
- 依赖关系
- 信任关系
- 评价关系

优化方法：

- 图路由
- GNN
- PageRank
- 社区发现
- 最短路径
- 最大流
- 强化学习路由

---

# 10. Agent / RAG / Skill / MAS 一体化架构

## 10.1 总体架构图

```text
User / External Event
  ↓
Intent & Goal Parser
  ↓
Agent Orchestrator
  ├─ Planner
  ├─ Memory Manager
  ├─ Tool Router
  ├─ Skill Router
  ├─ RAG Router
  ├─ MCP Client
  ├─ A2A Client
  └─ Safety Guard
        ↓
Execution Layer
  ├─ Tools via MCP
  ├─ Agents via A2A
  ├─ Skills
  ├─ RAG / GraphRAG
  ├─ Code Sandbox
  └─ Workflow Engine
        ↓
Verifier / Critic
        ↓
Memory Update / Skill Optimization
        ↓
Final Output
```

---

## 10.2 推荐工程分层

### L0：基础设施层

- 模型服务
- 向量数据库
- 图数据库
- 对象存储
- 关系数据库
- 消息队列
- 工作流引擎
- 日志系统
- 权限系统

### L1：协议连接层

- MCP Client / Server
- A2A Client / Server
- REST / GraphQL
- Webhook
- OAuth
- API Gateway

### L2：能力层

- Tools
- Skills
- RAG
- Memory
- Code Execution
- Browser Automation
- Data Analysis

### L3：Agent 层

- Planner Agent
- Executor Agent
- Research Agent
- Coding Agent
- Review Agent
- QA Agent
- Manager Agent

### L4：编排层

- Workflow Orchestrator
- Task Scheduler
- Event Bus
- Skill Graph Router
- Agent Router
- Retry Controller

### L5：治理层

- Safety
- Permission
- Audit
- Evaluation
- Cost Control
- Monitoring
- Human-in-the-loop

### L6：应用层

- 企业知识助手
- 自动开发助手
- 数据分析助手
- 客服助手
- 运营助手
- 科研助手
- 智能办公助手

---

# 11. 典型系统设计方案

## 11.1 企业知识 Agent

架构：

```text
User
  ↓
Intent Router
  ↓
RAG Router
  ├─ Vector RAG
  ├─ GraphRAG
  ├─ SQL-RAG
  └─ Web / API Search
  ↓
Evidence Verifier
  ↓
Answer Generator
  ↓
Citation Checker
```

关键设计：

- 权限继承原始数据源
- 文档级、段落级 ACL
- 引用强制
- 敏感信息脱敏
- 多源证据合并
- 回答可信度评分

---

## 11.2 自动代码开发 Agent

架构：

```text
Requirement
  ↓
Product Agent
  ↓
Architecture Agent
  ↓
Coding Agent
  ↓
Test Agent
  ↓
Review Agent
  ↓
Patch / PR
```

需要的 Skill：

- 仓库分析 Skill
- 需求拆解 Skill
- 架构设计 Skill
- 代码生成 Skill
- 单元测试 Skill
- Bug 修复 Skill
- 安全审查 Skill
- 文档生成 Skill

---

## 11.3 数据分析 Agent

架构：

```text
User Question
  ↓
Data Schema Agent
  ↓
SQL / Python Agent
  ↓
Chart Agent
  ↓
Insight Agent
  ↓
Report Agent
```

关键点：

- SQL 权限控制
- 查询成本限制
- 数据脱敏
- 结果校验
- 图表自动生成
- 分析过程可复现

---

## 11.4 智能客服 Agent

架构：

```text
User Message
  ↓
Intent Detection
  ↓
Policy / Knowledge RAG
  ↓
Tool Action
  ↓
Escalation Decision
  ↓
Response
```

关键点：

- 话术 Skill
- 工单 Skill
- 订单查询 MCP
- CRM MCP
- 人工转接
- 风险话题拦截

---

## 11.5 科研 Agent

架构：

```text
Research Question
  ↓
Literature Search Agent
  ↓
Paper Reader Agent
  ↓
Evidence Extractor
  ↓
Hypothesis Agent
  ↓
Experiment Planner
  ↓
Report Writer
```

关键点：

- 文献引用
- 实验复现
- 数据集跟踪
- 代码关联
- 结论置信度

---

# 12. 优化设计

## 12.1 Agent 优化

优化对象：

- Prompt
- Plan
- Tool Selection
- Memory Retrieval
- Skill Routing
- RAG Retrieval
- Agent Collaboration
- Verification
- Retry Strategy

方法：

- 日志回放
- 失败聚类
- LLM-as-Judge
- A/B Test
- Bandit Routing
- RL Routing
- Prompt Search
- Tool-use Fine-tuning
- Preference Optimization
- Cost-aware Planning

---

## 12.2 RAG 优化

优化方向：

- Chunk 策略
- Embedding 模型
- Hybrid Search
- Reranker
- Query Rewrite
- Metadata Filter
- Context Compression
- Graph Retrieval
- Citation Verification
- Evaluation Dataset

---

## 12.3 MCP 优化

优化方向：

- 工具粒度设计
- 参数 schema 精简
- 高风险工具隔离
- 只读/写入分离
- 工具缓存
- 批量调用
- 调用链审计
- 错误码规范
- 权限最小化

---

## 12.4 A2A 优化

优化方向：

- Agent Card 标准化
- 能力描述准确性
- 任务协议清晰
- 流式进度回传
- 委托链追踪
- 结果评分
- Agent Reputation
- Capability Registry
- 安全授权

---

## 12.5 Skill 优化

优化方向：

- Skill 拆分
- Skill 合并
- Skill 压缩
- 示例增强
- 工具封装
- 测试用例
- 自动评测
- 自动修复
- Skill Graph 路由
- 版本管理

---

## 12.6 MAS 优化

优化方向：

- 角色设计
- 通信拓扑
- 任务分配
- 冲突解决
- 共识机制
- 并行执行
- 成本控制
- 失败隔离
- 信任评分
- Agent 淘汰机制

---

# 13. 安全与治理框架

## 13.1 权限治理

- 用户权限
- Agent 权限
- Tool 权限
- Skill 权限
- 数据权限
- 环境权限
- 跨 Agent 委托权限

---

## 13.2 风险动作分类

低风险：

- 读取公开文档
- 总结文本
- 草拟内容

中风险：

- 查询内部数据
- 生成代码
- 修改草稿
- 创建任务

高风险：

- 发送邮件
- 删除文件
- 修改生产数据库
- 部署代码
- 付款
- 公开发布内容

高风险动作应要求：

- 人类确认
- 审计日志
- 回滚机制
- 权限校验
- 参数展示

---

## 13.3 Prompt Injection 防护

方法：

- 指令层级隔离
- 外部内容标记
- 工具输出不作为系统指令
- 检索内容过滤
- 高风险操作二次确认
- 数据外传检测
- 安全分类器
- 审计日志

---

## 13.4 数据安全

- ACL 继承
- 数据最小化
- 脱敏
- 加密
- 隔离向量库
- 多租户隔离
- 输出 DLP
- 审计追踪

---

# 14. 评测体系

## 14.1 Agent 评测

指标：

- 任务完成率
- 步骤成功率
- 工具调用成功率
- 平均轮数
- 平均成本
- 平均延迟
- 幻觉率
- 安全违规率
- 人类接管率
- 用户满意度

---

## 14.2 RAG 评测

指标：

- 检索召回率
- 上下文相关性
- 答案忠实度
- 引用准确率
- 覆盖率
- 多跳推理正确率
- 过期信息率

---

## 14.3 Skill 评测

指标：

- Skill 触发准确率
- Skill 成功率
- 输出格式正确率
- 平均成本
- 平均延迟
- 可复用率
- 失败恢复率
- 版本提升率

---

## 14.4 MAS 评测

指标：

- 协作效率
- 通信开销
- 冲突率
- 共识质量
- 角色贡献度
- 并行加速比
- 失败隔离能力
- 总体任务成功率

---

# 15. 设计模式总表

| 设计模式 | 核心思想 | 适合场景 |
|---|---|---|
| ReAct | 思考-行动-观察循环 | 工具调用 |
| Plan-Execute | 先规划后执行 | 长任务 |
| Reflexion | 失败后反思改进 | 代码/推理 |
| Self-RAG | 自主判断是否检索 | 知识问答 |
| Corrective RAG | 检索质量差时纠错 | 企业知识库 |
| GraphRAG | 实体关系图检索 | 复杂关系问答 |
| MCP Gateway | 统一工具入口 | 企业工具接入 |
| A2A Delegation | Agent 间任务委托 | 多 Agent 系统 |
| Skill Router | 自动选择技能 | 复杂工作流 |
| Skill Graph | 技能图编排 | 自动化流程 |
| Manager-Worker | 管理者分配任务 | 软件开发 |
| Debate | 多 Agent 讨论 | 方案评审 |
| Blackboard | 共享工作区 | 多源融合 |
| Market | Agent 竞标任务 | 动态能力网络 |
| Safe Tool Use | 高风险动作审批 | 企业生产系统 |

---

# 16. 推荐落地路线

## 16.1 第一阶段：单 Agent + 工具调用

目标：

- 做出可用 Agent
- 接入基础工具
- 完成简单任务

组件：

- LLM
- Tool Calling
- Basic Memory
- Basic RAG
- Logging

---

## 16.2 第二阶段：Agentic RAG + Skill

目标：

- 让 Agent 处理复杂知识任务
- 封装重复流程

组件：

- Hybrid RAG
- Reranker
- Skill Router
- Skill Repository
- Evaluation Set

---

## 16.3 第三阶段：MCP 化

目标：

- 标准化工具接入
- 降低系统耦合

组件：

- MCP Server
- MCP Client
- Tool Registry
- Permission Layer
- Audit Log

---

## 16.4 第四阶段：MAS + A2A

目标：

- 多 Agent 协同
- 跨系统调用 Agent

组件：

- A2A Server
- Agent Card Registry
- Manager Agent
- Worker Agents
- Message Bus
- Collaboration Protocol

---

## 16.5 第五阶段：Skill Graph + 自动优化

目标：

- 自动编排能力
- 自动修复和优化

组件：

- Skill Graph
- Skill Evaluator
- Failure Analyzer
- Auto Skill Generator
- A/B Testing
- Version Manager

---

## 16.6 第六阶段：AgentOps

目标：

- 生产级治理

组件：

- Observability
- Tracing
- Cost Control
- Security Guard
- Human Approval
- Replay Testing
- Benchmark
- Compliance

---

# 17. 最终推荐的统一架构

```text
                         ┌────────────────────┐
                         │      User / API     │
                         └─────────┬──────────┘
                                   ↓
                         ┌────────────────────┐
                         │  Intent / Goal Parser│
                         └─────────┬──────────┘
                                   ↓
                         ┌────────────────────┐
                         │  Agent Orchestrator │
                         └─────────┬──────────┘
             ┌─────────────────────┼─────────────────────┐
             ↓                     ↓                     ↓
      ┌────────────┐        ┌────────────┐        ┌────────────┐
      │   RAG Hub  │        │ Skill Graph│        │  MAS Layer │
      └─────┬──────┘        └─────┬──────┘        └─────┬──────┘
            ↓                     ↓                     ↓
 ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
 │ Vector/Graph/SQL │   │ Skill Repository │   │ A2A Agent Network│
 └──────────────────┘   └──────────────────┘   └──────────────────┘
            ↓                     ↓                     ↓
             ┌─────────────────────┼────────────────────┐
             ↓                     ↓                    ↓
       ┌──────────┐         ┌──────────┐          ┌──────────┐
       │ MCP Tools│         │ Workflows│          │ Executors│
       └────┬─────┘         └────┬─────┘          └────┬─────┘
            ↓                    ↓                     ↓
       ┌────────────────────────────────────────────────┐
       │        Verifier / Critic / Safety Guard         │
       └─────────────────────┬──────────────────────────┘
                             ↓
       ┌────────────────────────────────────────────────┐
       │ Memory Update / Skill Optimization / AgentOps   │
       └─────────────────────┬──────────────────────────┘
                             ↓
                         Final Output
```

---

# 18. 总结

可以把现代大模型 Agent 技术体系总结为：

1. **Agent 架构**解决“如何自主规划和执行任务”  
2. **RAG 架构**解决“如何接入外部知识并减少幻觉”  
3. **MCP 架构**解决“如何标准化连接工具和数据源”  
4. **A2A 架构**解决“Agent 与 Agent 如何互通协作”  
5. **Skill 架构**解决“如何沉淀可复用能力”  
6. **Skill Graph 架构**解决“如何组合、路由和优化能力”  
7. **自动化 Skill 优化**解决“如何让能力持续进化”  
8. **MAS 架构**解决“如何构建 Agent 团队和组织”  
9. **AgentOps**解决“如何上线、监控、安全、评测和治理”  

最终，生产级智能体系统不是单个 Prompt，而是：

```text
LLM + RAG + Tool + MCP + A2A + Skill + Skill Graph + MAS + Evaluation + Safety + Ops
```

也就是一个可观测、可扩展、可治理、可优化的智能体操作系统。

---

# 参考资料

- Model Context Protocol 官方架构文档：https://modelcontextprotocol.io/
- Model Context Protocol 2026 Roadmap：https://blog.modelcontextprotocol.io/
- A2A Protocol 官方文档：https://a2a-protocol.org/
- Google Developers Blog: Developer's Guide to AI Agent Protocols：https://developers.googleblog.com/
- Anthropic Claude Code Skills 文档：https://code.claude.com/docs/en/skills
- Anthropic Skills Repository：https://github.com/anthropics/skills
- Agentic RAG Survey：https://arxiv.org/abs/2501.09136
- GraphRAG / Agentic RAG / Modular RAG 相关综述
- Multi-Agent Systems, AgentOps, Tool-use Agent, ReAct, Reflexion, Tree-of-Thought, Graph-of-Thought, Self-RAG, Corrective RAG 等公开研究资料
