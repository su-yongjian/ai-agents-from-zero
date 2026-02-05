# 9-LangChain 框架原理与应用

---

## 1、LangChain 的理解

### 1.1 什么是 LangChain

**LangChain** 是 2022 年 10 月底由哈佛大学的 Harrison Chase（哈里森·蔡斯）发起的、基于开源大语言模型的 **AI 工程开发框架**。

顾名思义：

- **Lang**：指大语言模型（Large Language Model）
- **Chain**：即“链”，将大模型与数据源、工具、记忆等组件连接成链，借此构建完整的 AI 应用

LangChain 的发布比 ChatGPT 问世还要早约一个月，从启动时间可以看出创始团队的前瞻性；占得先机后，该框架迅速获得业界广泛关注与社区支持。

### 1.2 一句话总结

**LangChain 就是一套把大模型和外部世界连接起来的工具与代码框架。**  
开发者可以用它接入各种数据源、工具（搜索、数据库、API）和记忆组件，用统一的“链”式方式编排调用流程，从而快速搭建 RAG、Agent、对话等应用。

### 1.3 AB 法则（Before | After）

- **Before**：没有 LangChain 时，需要自己对接各家模型 API、设计提示模板、手写检索与调用逻辑，重复造轮子且难以复用。
- **After**：使用 LangChain 后，可以基于统一的抽象（Models、Chains、Agents、Memory 等）组装应用，大量集成与最佳实践开箱可用，开发效率显著提升。

---

## 2、LangChain 能做什么

### 2.1 大模型开发中的角色类比

可以把 LLM 在大模型应用中的角色，类比为传统 Web 开发中的分层：

| 传统分层     | 大模型应用中的对应   | 说明                        |
| ------------ | -------------------- | --------------------------- |
| Controller   | 入口与流程编排       | 接收用户输入，调度链/Agent  |
| Service      | LLM 调用与业务逻辑   | 调用模型、解析结果、决策    |
| DAO / Mapper | 检索、工具、外部数据 | 向量库、数据库、API、搜索等 |

LangChain 在上述每一层都提供了组件和集成，方便你快速搭建从“查数据 → 拼提示 → 调模型 → 再决策”的完整流水线。

### 2.2 大模型开发分类与岗位对应

- **应用开发**：基于现有大模型做 RAG、对话、客服、写作助手等，对应岗位中常见的“大模型应用开发”“RAG/Agent 开发”。
- **模型与算法**：预训练、微调、推理优化等，更偏算法与基础设施。
- **平台与工具**：LangChain、Dify、Coze 等框架与低代码平台的使用与二次开发。

Boss 直聘等招聘网站上与“大模型应用”“RAG”“LangChain”相关的岗位，多数属于**应用开发**这一层；掌握 LangChain 有助于快速对接企业中的检索、工具调用和流程编排需求。

### 2.3 LLM 大模型应用技术架构

典型架构包含：

- **数据与检索**：文档加载、切分、向量化、向量库检索（对应 RAG）。
- **模型层**：通过 LangChain 统一封装 OpenAI、Anthropic、本地模型等，便于切换与扩展。
- **编排层**：Chains（链）串联多步逻辑；Agents（智能体）在“思考—调用工具—再思考”之间循环。
- **记忆与状态**：Memory 管理对话历史或会话状态，支持多轮对话与上下文延续。
- **可观测性**：Callback 与 LangSmith 等用于日志、追踪与调试。

LangChain 在这些环节都提供了标准化组件与集成，构成一套相对完整的 **LLM 应用技术架构** 参考实现。

---

## 3、LangChain 与 LangChain4J

### 3.1 LangChain（Python 版）

- **定位**：LangChain For Python，当前生态最成熟、文档与示例最多的版本。
- **适用**：Python 技术栈下的 AI 应用开发、算法验证、RAG/Agent 快速搭建。

### 3.2 LangChain4J（Java 版）

随着 AI 应用在企业中的普及，仅支持 Python 的框架无法完全覆盖 Java/Spring 技术栈。**LangChain4J** 相当于 **LangChain For Java**，为 Java 开发者提供与 LangChain 理念一致的抽象与集成（模型、工具、记忆、链等），便于在 Spring 等现有系统中集成大模型与 RAG/Agent 能力。

任何希望在更大范围内推广的框架或云服务，通常都会考虑庞大的 Spring 与 Java 开发者群体，LangChain4J 正是面向这一群体的官方 Java 实现。

- **参考资源**：https://www.bilibili.com/video/BV1mX3NzrEu6/

---

## 4、资源与官方渠道

### 4.1 官网与文档

- 中文文档（社区）：https://docs.langchain.org.cn/oss/python/langchain/overview
- 英文官方文档：https://docs.langchain.com/oss/python/langchain/overview

### 4.2 GitHub 与 API 参考

- **GitHub**：https://github.com/langchain-ai
- **API 参考（Python）**：https://reference.langchain.com/python/langchain/
- **包结构与模块索引**：https://reference.langchain.com/python/langchain/langchain/

---

## 5、怎么学、怎么玩

### 5.1 为什么现在是学习 LangChain 的最佳时机

- **生态强**：常用工具（如 Google Search、Wikipedia、Notion、Gmail 等）和常用技术（RAG、ReAct、MapReduce 等）在 LangChain 中大多有现成组件或模板。
- **定位类似“AI 界的 Spring/React”**：虽然体量大、抽象多，但仍是目前上手最快、社区资源最多的选择之一。
- **学习建议**：不必试图学完所有 API；先搞懂**六大核心模块**（Models、Memory、Retrieval、Chains、Agents、Callback）的逻辑与组合方式，用到什么再查什么，把它当作**工具箱**而非教科书。

### 5.2 总体架构与版本演进

#### 5.2.1 早期版本（V0.1）

以单库为主，链条设计与基础组件集中在一个项目中，适合快速理解“链”的概念。

#### 5.2.2 V0.2 / V0.3 版本

LangChain 生态系统按三个层次划分：**架构（Architecture）**、**组件（Components）**、**部署（Deployment）**。

- **架构层**：LangChain（基础链与构建）+ LangGraph（图结构、复杂流程与循环），均为开源（OSS）。
- **组件层**：Integrations（与 API、数据库、第三方模型等的集成），支持灵活扩展。
- **部署层**：LangGraph Cloud、LangSmith 等，用于托管、监控与调试。

#### 5.2.3 LangChain 1.0：轻核心与模块化

- **langchain-core**：基础抽象与 **LCEL（LangChain 表达式语言）**，承载 Chains、Agents、Retrieval 等核心概念。
- **langchain-community**：社区与第三方集成；与各厂商的深度集成拆分为独立包，如 **langchain-openai**、**langchain-anthropic** 等。
- **LangGraph**：在 LangChain 之上提供图化编排能力，可协调多个 Chain、Agent、Tool，并支持循环与条件分支，适合复杂任务与多步推理。

### 5.3 六大核心模块（“六大金刚”）

LangChain 的六大组件**耦合松散**，没有固定调用顺序或强制的统一接口，开发者可以按需自由组合。

| 模块   | 英文      | 作用简述                                   |
| ------ | --------- | ------------------------------------------ |
| 模型   | Models    | 封装 LLM、Chat Model、Embedding 等统一接口 |
| 记忆   | Memory    | 管理对话历史、会话状态，支持多轮对话       |
| 检索   | Retrieval | 文档加载、切分、向量化与向量库检索（RAG）  |
| 链     | Chains    | 将多步调用串联成固定或条件式流水线         |
| 智能体 | Agents    | 根据推理结果选择工具、循环执行直至完成任务 |
| 回调   | Callback  | 埋点、日志、监控与调试支持                 |

- **ChatModel**：对“聊天式”大模型（如 GPT、Claude）的抽象，输入/输出多为消息列表，是构建对话与 Agent 的基础。
- **Agent**：在“规划 → 选工具 → 执行 → 再规划”的循环中完成任务，是复杂任务自动化的核心抽象之一。

---

## 6、LangChain 的不足与槽点

LangChain 虽是当前业界使用最广的 LLM 应用框架之一，但存在一些公认的问题：

1. **文档与代码不同步**：更新节奏快（几乎两三天一版），文档常滞后于代码，文档中的方法在最新版本中可能已被重命名或移除，对新手不友好。
2. **抽象层次多**：为兼容多种模型与数据源，封装层级较深，简单需求有时需要摸清很深的调用链；所谓“LangChain 很慢”往往指的是**理解与调试成本高**，而非单纯运行慢。
3. **版本兼容性**：升级后旧代码可能无法直接运行，依赖与 API 变更较多，需要关注版本与迁移说明。

因此，建议在实际项目中**锁定版本**、**以官方示例和 API 参考为准**，并优先掌握核心抽象（Chain、Agent、LCEL），再按需查阅具体集成。

---

## 7、LangChain 1.0 与 AI 原生的未来

LangChain 正在从“代码库”演化为一套更完整的 **“AI 开发操作系统”**：从开发、调试到部署与可观测性，提供一站式的抽象与工具。

同时，大模型能力本身在增强（如 Context 窗口变大、长文本理解更好），未来简单的 RAG 场景可能不再需要过于复杂的切片与检索设计；但 **Agent（智能体）** 的“规划—工具调用—多步执行”逻辑会越来越重要。

未来的 AI 应用，将更多从“一问一答”走向**“一句话解决复杂任务”**，例如：“帮我写个小游戏并发布到 App Store”可能涉及写代码、画图、测试、填表、上传等多步。LangChain 与 LangGraph 正是在为这类**多步、可编排、可观测**的 AI 应用打地基。
