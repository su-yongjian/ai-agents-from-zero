# 10 - LangChain 快速上手与 HelloWorld

---

**本章课程目标：**

- 完成从零到一的 HelloWorld：环境与约定、安装依赖、基于阿里百炼/通义接入与调用。
- 掌握**调用三件套**（API Key、模型名、Base URL）及 0.x 与 1.0 两种写法（`ChatOpenAI` / `init_chat_model`）。
- 会实现**多模型共存**（通义 + DeepSeek）、**企业级封装与流式输出**，为后续学习 [Model I/O](11-Model-I-O与模型接入.md)、[Ollama 本地调用](12-Ollama本地部署与调用.md)、[提示词与消息模板](13-提示词与消息模板.md)、[输出解析器](14-输出解析器.md)（第 11 ～ 14 章）打好基础。

**前置知识建议：** 已学习 [第 9 章 - LangChain 概述与架构](9-LangChain概述与架构.md)，了解 LangChain 的定位与六大核心模块；具备 Python 环境与包管理基础。环境与版本说明见 [新手入门与常见问题](新手入门与常见问题.md)。

**学习建议：** 按「环境与约定 → 安装依赖 → 百炼三件套 → HelloWorld → 多模型共存 → 企业级封装与流式」顺序动手。调用任何模型都离不开 **API Key、模型名、Base URL** 三件套，建议先在一个平台（如阿里百炼）跑通再扩展。

---

## 1、LangChain 环境与约定

### 1.1 支持的大模型与课程选用

LangChain 通过各厂商的集成包支持多种大模型，官方文档有完整列表：

- https://docs.langchain.com/oss/python/integrations/providers/overview#popular-providers

![LangChain支持的大模型](images/10/10-1-1-1.png)

本课程以 **阿里云百炼（通义千问）** 为主，辅以 **DeepSeek** 与 **OpenRouter**；通过统一配置规则，也适用于其他兼容 OpenAI 协议的模型，便于举一反三。

### 1.2 Python 环境与版本约定

实操前建议先确认：<strong>Python 3.8+（推荐 Python 3.10/3.11）</strong>、已创建虚拟环境（如 `venv` 或 `conda`）。若你在 Windows 上使用部分依赖时遇到兼容性问题（如个别包与解释器版本不匹配），优先切换到 3.10/3.11 通常更稳。

---

## 2、常见大模型服务平台介绍

LangChain 不提供大模型本身，需要配合 **第三方大模型服务平台**：注册、充值、创建 API Key 后，用 **API Key + Base URL** 调用对应模型。

| 平台           | 入口                                                   | API-Key 管理                                                        | API 文档                                                                          | 模型                                                                         | 说明                                                      |
| -------------- | ------------------------------------------------------ | ------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------- |
| **CloseAI**    | [平台](https://platform.closeai-asia.com/)             | [API-Key](https://platform.closeai-asia.com/developer/api)          | [文档](https://doc.closeai-asia.com/tutorial/api/openai.html)                     | [模型](https://platform.closeai-asia.com/pricing)                            | 企业级 OpenAI/国际大模型 API 代理（中转）                 |
| **OpenRouter** | [平台](https://openrouter.ai/)                         | [Keys](https://openrouter.ai/settings/keys)                         | [文档](https://openrouter.ai/docs/community/frameworks-and-integrations-overview) | [模型](https://openrouter.ai/models)                                         | 多模型统一 API 聚合（一个接口调 400+ 模型）               |
| **阿里云百炼** | [平台](https://bailian.console.aliyun.com/)            | [API-Key](https://bailian.console.aliyun.com/?tab=model#/api-key)   | [文档](https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model)              | [模型](https://bailian.console.aliyun.com/?tab=model#/model-market/all)      | 阿里系；本课程主要使用                                    |
| **百度千帆**   | [平台](https://console.bce.baidu.com/qianfan/overview) | [API-Key](https://console.bce.baidu.com/qianfan/ais/console/apiKey) | [文档](https://cloud.baidu.com/doc/qianfan-docs/s/Mm8r1mejk)                      | [模型](https://console.bce.baidu.com/qianfan/modelcenter/model/buildIn/list) | 百度系                                                    |
| **硅基流动**   | [平台](https://www.siliconflow.cn/)                    | [API-Key](https://cloud.siliconflow.cn/me/account/ak)               | [文档](https://docs.siliconflow.cn/cn/userguide/capabilities/text-generation)     | [模型](https://cloud.siliconflow.cn/me/models)                               | 国内 AI 能力平台（多模型 API + 推理加速 + 微调 + 私有化） |

| 模型/平台                   | 主要特点                                 | 优势                                           | 备注                                                                              |
| --------------------------- | ---------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------- |
| **OpenAI**                  | GPT 系列，文本生成与理解能力强           | 灵活、适用场景广，业界常用标杆                 | 国内 API 已暂停，需通过 Azure 等渠道访问                                          |
| **阿里百炼**                | 通义千问等大模型服务                     | 性能接近 GPT-4、价格较低、支持企业迁移与私有化 | 主要面向企业；新用户有较多免费 Token 与绘图额度                                   |
| **DeepSeek**                | 开源大模型，多语言支持                   | 推理与代码能力突出，社区活跃，可做多种应用     | 性价比高；注意内容审核与账号风控政策                                              |
| **智谱清言（Zhipu）**       | 基于 GLM 架构，多轮对话与复杂指令        | 指令理解好，支持多场景定制                     | 模型较全；活动期常有大幅降价或免费额度（如国庆 1 折、亿级 Token 等）              |
| **硅基流动（SiliconFlow）** | AI 基础设施与云平台                      | 推理高效、多模态、降低使用门槛，提升开发效率   | 面向开发者；多款开源模型（如 Qwen2、GLM4、Yi1.5）API **永久免费**，适合练手与开发 |
| **Ollama**                  | 本地部署，集成多款开源模型，数据不出本机 | 隐私与自主可控，不依赖外网 API                 | 需本机具备一定显卡/算力，适合有本地环境的同学                                     |

---

## 3、安装依赖

### 3.1 配置镜像源与安装包

**建议**：若希望 pip 默认使用国内源，可先执行：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**安装命令**：

> **提示**：若已按上文将 pip 全局切换为清华源，则下面命令可省略 `-i https://pypi.tuna.tsinghua.edu.cn/simple`，直接写 `pip install 包名` 即可。

```bash
# 核心框架（Chain、Agent、Memory、Retriever 等）
pip install langchain -i https://pypi.tuna.tsinghua.edu.cn/simple

# OpenAI 兼容组件（LLM、Chat、Embeddings 等），依赖 openai SDK
pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple

# 从 .env 加载环境变量
pip install python-dotenv -i https://pypi.tuna.tsinghua.edu.cn/simple

# 核心抽象与类型
pip install langchain-core
```

> **安全提示**：请将 API Key 写在项目根目录的 **`.env`** 文件中（例如：`QWEN_API_KEY=sk-xxx`），不要写进代码或提交到版本库。使用 `python-dotenv` 的 `load_dotenv()` 可在代码中自动加载这些环境变量。

### 3.2 验证安装

安装完成后，可用以下两种方式确认环境是否正常。

**方法一：运行环境检查脚本（终端）**

在项目目录下执行脚本，可一次性查看 LangChain、langchain_community 的版本与安装路径，以及当前 Python 版本。

【案例源码】环境检查脚本：`案例与源码-2-LangChain框架/01-helloworld/GetEnvInfo.py`

[GetEnvInfo.py](案例与源码-2-LangChain框架/01-helloworld/GetEnvInfo.py ":include :type=code")

**方法二：在 PyCharm 中查看已安装包（图形界面）**

使用 PyCharm 时，可打开底部 **「Python 软件包」** 面板，在列表中找到 `langchain`、`langchain-core`、`langchain-openai` 等包，点击即可在右侧查看版本号、许可证等详情，无需运行代码。

![图 10-3-1-2：在 PyCharm 中查看已安装的 Python 包与版本](images/10/10-3-1-2.png)

---

## 4、案例：基于阿里百炼的 HelloWorld

### 4.1 百炼平台入口与准备

- **官网**：https://bailian.console.aliyun.com/
- **步骤概览**：① 注册/登录阿里云 → ② 在百炼控制台创建 API Key → ③ 在模型广场确认模型名（如 `qwen-plus`）→ ④ 获取 OpenAI 兼容的 Base URL → ⑤ 在本地用 LangChain 写代码调用。

### 4.2 调用三件套：API Key、模型名、Base URL

任意兼容 OpenAI 协议的模型（含百炼、DeepSeek 等）都需准备这三项；更多厂商接入方式与参数说明见 [第 11 章 Model I/O](11-Model-I-O与模型接入.md)。

**① 获得 API Key**

在百炼控制台「API-KEY 管理」中创建并复制 Key（形如 `sk-xxx`）。

![阿里百炼 API Key 获取](images/10/10-4-2-1.jpeg)

**② 获得模型名**

在模型广场或文档中确认要调用的模型标识，例如 `qwen-plus`、`qwen3-max` 等。

![百炼模型列表与模型名](images/10/10-4-2-2.jpeg)

![模型详情与调用名](images/10/10-4-2-3.jpeg)

![模型名示例 qwen-plus / qwen3-max](images/10/10-4-2-4.jpeg)

**③ 获得 Base URL（开发地址）**

使用 SDK 调用时需配置兼容 OpenAI 的接口地址，例如：

![百炼 OpenAI 兼容接口地址](images/10/10-4-2-5.jpeg)

**本节小结**

| 项目         | 示例/说明                                           |
| ------------ | --------------------------------------------------- |
| **API Key**  | `sk-xxx`（在控制台创建）                            |
| **模型名**   | 如 `qwen-plus`、`qwen3-max`                         |
| **Base URL** | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

### 4.3 示例代码（0.3 与 1.0 两种写法）

> **说明**：本地安装的是 LangChain 1.0 及以上版本时，**也可以直接运行下面 0.3 的代码**，接口保持兼容；两种写法按需选用即可。

**方式一：LangChain 0.3（了解即可，目前仍在使用）**

【案例源码】`案例与源码-2-LangChain框架/01-helloworld/LangChainV0.3.py`

[LangChainV0.3.py](案例与源码-2-LangChain框架/01-helloworld/LangChainV0.3.py ":include :type=code")

**方式二：LangChain 1.0+（推荐）**

【案例源码】`案例与源码-2-LangChain框架/01-helloworld/LangChainV1.0.py`

[LangChainV1.0.py](案例与源码-2-LangChain框架/01-helloworld/LangChainV1.0.py ":include :type=code")

### 4.4 v0.3 与 1.0 写法差在哪

- **0.3 思路**：从具体包里导入「专用类」，例如 `from langchain_openai import ChatOpenAI`，再用 `ChatOpenAI(...)` 建模型。换一家平台或换一种封装时，往往要换导入的类名、记「这家该用哪个类」，代码骨架也容易跟着变。

- **1.0 思路**：用 **`init_chat_model` 做统一入口**——始终是「同一个函数 + 不同参数」。你只要改 **`model`（用哪个模型）**、**`model_provider`（按哪种厂商协议来建客户端）**，再配合 **`api_key`、`base_url`** 等，就能在同一种写法下切换模型或接入方式，不必为每家厂商各记一套类名。本教程里百炼走 **OpenAI 兼容** 接口时，`model_provider` 通常写 **`"openai"`**，并配上 4.2 节里的 Base URL；具体以示例源码 `LangChainV1.0.py` 中的注释为准。

---

## 5、案例：多模型共存（通义 + DeepSeek）

### 5.1 DeepSeek 平台与三件套

- 使用说明：https://platform.deepseek.com/usage
- API 文档：https://api-docs.deepseek.com/zh-cn/

**同样需要 4.2 节的「三件套」**：API Key、模型名、Base URL（详见上文 4.2 调用三件套）。

- **① 获得 API Key**：在 DeepSeek 平台创建并复制。

![DeepSeek 控制台中的 API Key 创建与查看入口](images/10/10-5-1-1.jpeg)

- **② 获得模型名**：如 `deepseek-chat`（非思考模式）、`deepseek-reasoner`（思考模式）。

![DeepSeek 模型列表与调用名示意（如 deepseek-chat、deepseek-reasoner）](images/10/10-5-1-2.jpeg)

- **③ Base URL**：一般为 `https://api.deepseek.com`（以官方文档为准）。

**备注**：`deepseek-chat` 对应 DeepSeek-V3.2 的普通模式；`deepseek-reasoner` 对应思考/推理模式。

![DeepSeek 普通模式与推理模式说明示意](images/10/10-5-1-3.jpeg)

### 5.2 多模型共存示例代码

下面示例使用 **不同变量名**（`model_qwen`、`model_deepseek`）保存两个模型实例，避免后者覆盖前者，便于后续扩展与维护。

【案例源码】`案例与源码-2-LangChain框架/01-helloworld/LangChain_MoreV1.0.py`

[LangChain_MoreV1.0.py](案例与源码-2-LangChain框架/01-helloworld/LangChain_MoreV1.0.py ":include :type=code")

---

## 6、实战：企业级封装与流式输出

### 6.1 流式输出说明

通过 `stream()` 可逐 token 返回结果，适合长文本或实时展示。

和 `invoke()` 一次性返回完整结果不同，`stream()` 会把模型回复拆成很多小片段连续返回。你可以在收到片段时立刻打印到终端，用户会更快看到“正在生成”的过程，体感更流畅。

最小用法示例：

```python
for chunk in model.stream("请介绍一下 LangGraph"):
    print(chunk.content, end="")
```

可以把它理解成“边生成边输出”：

- `invoke()`：等模型全部生成完，再一次性拿到完整文本。
- `stream()`：模型每生成一段就先给你一段，适合长回答、实时展示、聊天打字机效果。

实践建议：如果要做命令行或网页端的实时输出，优先使用 `stream()`；如果只是离线处理、一次性拿结果，`invoke()` 会更直接。

### 6.2 示例代码（封装、异常、流式）

下面示例将 LLM 初始化封装成函数、做环境变量校验、使用日志与异常处理，并演示流式调用。

【案例源码】`案例与源码-2-LangChain框架/01-helloworld/StandardDesc.py`

[StandardDesc.py](案例与源码-2-LangChain框架/01-helloworld/StandardDesc.py ":include :type=code")

---

**本章小结：**

- **上手路径**：安装 `langchain`、`langchain-openai`、`langchain-core`、`python-dotenv` 等；调用任意模型需准备**三件套**：**API Key**、**模型名**、**Base URL**（阿里百炼、DeepSeek 等均提供 OpenAI 兼容接口）。密钥放在 `.env`，用 `load_dotenv()` 加载，不要写进代码。
- **HelloWorld**：用 `init_chat_model` 或 `ChatOpenAI` 初始化模型，`model.invoke("问题")` 得到回复，`.content` 取文本；**多模型共存**用不同变量名保存多个模型实例；**流式输出**用 `model.stream(...)` 逐 token 返回；企业级写法可封装初始化函数、做环境校验与异常处理、打日志。

**建议下一步：** 在本地用阿里百炼或 DeepSeek 的 API Key、模型名、Base URL 跑通一次 `init_chat_model` + `invoke`，再试 `stream()` 流式输出；若需多模型，在同一脚本里初始化通义与 DeepSeek 各一个实例并分别调用。接着可学习 [第 11 章 Model I/O](11-Model-I-O与模型接入.md)、[第 12 章 Ollama 本地部署](12-Ollama本地部署与调用.md)、[第 13 章 提示词与消息模板](13-提示词与消息模板.md)、[第 14 章 输出解析器](14-输出解析器.md)，形成「输入 → 模型 → 输出解析」闭环；再学 [第 15 章 LCEL 与链式调用](15-LCEL与链式调用.md)，把三件套串成链。
