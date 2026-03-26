# 11 - Model I/O 与模型接入

---

**本章课程目标：**

- 理解 LangChain 的 **Model I/O** 模块：输入提示（Prompt）、调用模型（Model）、输出解析（Parser）三件套。
- 掌握 LangChain 中**模型分类**（LLM、Chat、Embedding）与**标准化参数、Message 返回**。
- 学会在 LangChain 中接入 **OpenAI 兼容接口**、**DeepSeek**、**通义千问**、**智谱** 等大模型的方式与选型。

**前置知识建议：** 已学习 [第 9 章 - LangChain 概述与架构](9-LangChain概述与架构.md)、[第 10 章 - LangChain 快速上手与 HelloWorld](10-LangChain快速上手与HelloWorld.md)，了解 LangChain 的定位与基本用法（含 `init_chat_model`、三件套）；具备 Python 环境与 API 调用基础。

---

## 1、Model I/O 概述与三件套

Model I/O 是 LangChain 中与大模型交互的**核心模块**，负责把大模型的**输入、调用、输出**标准化：通常对应**提示模板**、**模型调用**与**输出解析/格式化**。

**一句话定义**：**Model I/O = Format（输入格式化）→ Predict（模型调用）→ Parse（输出解析）**。

- **Format（输入格式化）**：把原始业务输入整理成模型更容易理解的形式，常见做法是使用提示词模板（Prompt Template）管理变量与上下文，避免手写字符串拼接。
- **Predict（模型调用）**：通过统一接口调用不同模型（OpenAI、DeepSeek、通义、Ollama 等），屏蔽厂商差异，聚焦业务逻辑。
- **Parse（输出解析）**：将模型返回从自然语言转为程序可直接处理的结构化结果（如 JSON、列表、对象），便于后续链路使用。

也可以用“点外卖”来理解这三步：

- **Format** 像“下单前把需求写清楚”：要不要辣、几人份、送到哪里。你描述越清楚，模型越不容易答偏。
- **Predict** 像“把订单交给餐厅做菜”：你可以换不同餐厅（不同模型），但下单方式尽量统一（统一接口）。
- **Parse** 像“拿到外卖后先分装打包”：把原始回答整理成程序能直接用的数据（例如固定 JSON 字段），后续系统才好自动处理。

![Model I/O 三步流程：Format（模板与变量）→ Predict（LLM / Chat Model）→ Parse（输出解析为 JSON 等结构化数据）](images/11/11-1-1-1.png)

> **图示说明**：**Format** 将业务变量填入 Prompt 模板，得到发给模型的完整输入；**Predict** 由 LLM 或 Chat Model 生成文本；**Parse** 用 Output Parser 把自然语言结果转成 JSON 等结构化数据，便于程序继续处理。

**官方文档与资源：**

- **Model I/O 总览**：https://docs.langchain.com/oss/python/langchain/models （英文）；https://docs.langchain.org.cn/oss/python/langchain/models （中文）
- **聊天模型集成**：https://docs.langchain.com/oss/python/integrations/chat （英文）；https://docs.langchain.org.cn/oss/python/integrations/chat （中文）

---

## 2、LangChain 模型分类、参数与返回

本节**不是**脱离 Model I/O 另起炉灶，而是紧扣第 1 节里的 **Predict（模型调用）** 这一环：先分清**调用哪一类模型**（LLM / Chat / Embedding）、**调用时常用参数**（temperature、`max_tokens` 等）、以及**调用后返回什么**（`AIMessage` 等）。**Format（如何把输入整理成提示）** 主要在 [第 13 章 提示词与消息模板](13-提示词与消息模板.md)；**Parse（如何把输出变成结构化数据）** 在 [第 14 章 输出解析器](14-输出解析器.md)。第 3 节则是在此基础上，把同一套「Predict」落到各家厂商的具体接法上。

### 2.1 模型分类（LLM / Chat / Embedding）

LangChain **不提供**大模型权重本身，而是通过**第三方集成**把各平台 API 接入到你的应用中，例如：OpenAI、Anthropic、Hugging Face、LLaMA、阿里通义、ChatGLM 等。

LangChain 将大语言模型按用途分为多种类型，**实际开发中最常用的是「聊天对话模型」**（ChatModel），用于多轮对话、系统角色与用户消息等场景。

**LangChain 中模型分类：LLM、Chat、Embedding 等**

| 模型类型                                | 输入形式                                                                            | 输出形式                           | 主要特点                                                                                   | 典型适用场景                                                 |
| --------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| **LLM**（大语言模型）                   | 纯文本字符串                                                                        | 文本字符串                         | ① 面向「补全式」单条文本接口<br>② 无多角色消息结构，多轮需在应用侧自行拼接<br>③ 高速、轻量 | 单轮问答；摘要生成；文本改写/扩写；指令执行（Instruct 模型） |
| **ChatModel**（聊天模型）               | 消息列表（`List[BaseMessage]`），如 `HumanMessage`、`SystemMessage`、`AIMessage` 等 | 聊天消息对象（`AIMessage`）        | ① 面向对话场景<br>② 支持多轮上下文<br>③ 更贴近人类对话逻辑                                 | 智能助手；客服机器人；多轮推理任务；LangChain Agent 工具调用 |
| **Embeddings**（文本向量模型/嵌入模型） | 文本字符串或列表（`str` 或 `List[str]`）                                            | 向量（`List[float]` 或 `ndarray`） | ① 将文本转化为语义向量<br>② 可用于相似度搜索<br>③ 通常不生成文本                           | 文本检索增强（RAG）；知识库问答；聚类/分类/推荐系统          |

> **为何第 1-1 章没有单独提 ChatModel？** [第 1-1 章](1-1-大模型认知与工程概览.md) 按**模型功能**（输出形态）分为四类：**生成式大模型**（Large Language Model，LLM）、**嵌入模型**（Embedding Model）、**重排序模型**（Reranking Model）、**分类模型**（Classification Model）。LangChain 里的 **LLM** 和 **ChatModel** 在功能上**都属于「生成式大模型（LLM）」**——都是「输入内容 → 生成自然语言」；区别只是**调用方式**（纯文本 vs 消息列表、单轮 vs 多轮）。故第 1-1 章只列「生成式大模型」这一大类；在 LangChain 里则按接口区分，便于选 API。

### 2.2 常用模型参数

在构建聊天模型时，常用 **`init_chat_model`（LangChain 1.0+）** 或各集成包中的模型类。LangChain 对聊天模型定义了一批**标准化参数**，名称在不同写法下基本一致，便于换厂商时少改代码。

- **官方说明**：https://docs.langchain.com/oss/python/langchain/models#parameters （英文）；https://docs.langchain.org.cn/oss/python/langchain/models#parameters （中文）

**`init_chat_model` 常用参数一览**

|      参数名      | 说明                                                                                                                                                                                                                             |
| :--------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|     `model`      | 模型名称或标识符，例如 `gpt-4o-mini`、`qwen-plus`、`deepseek-chat`。                                                                                                                                                             |
| `model_provider` | 按哪种厂商/协议创建底层客户端，例如 `"openai"`（OpenAI 协议，也常用于百炼/DeepSeek 等兼容端点）。                                                                                                                                |
|    `base_url`    | 请求发往的 API 根地址（Base URL），由服务商提供。                                                                                                                                                                                |
|    `api_key`     | 调用服务商接口时的身份凭证（API Key），建议放在环境变量中，不要写进代码仓库。                                                                                                                                                    |
|  `temperature`   | 控制解码时的随机性：数值越高，输出越多样、越有「创意」；越低则越稳定、越确定。与 top_p 等共同影响采样，原理可参考 [知乎专栏：大模型采样相关](https://zhuanlan.zhihu.com/p/1981752176578667658)（第三方科普，以各厂商文档为准）。 |
|    `timeout`     | 等待模型返回的最长时间（通常以秒为单位），超时则取消本次请求。                                                                                                                                                                   |
|   `max_tokens`   | 限制**本次生成**最多消耗多少个 token（上限），用于控制回答长度与费用。                                                                                                                                                           |
|      `stop`      | 停止词序列：生成中一旦遇到这些字符串（或 token），模型会停止继续输出，用于截断、控制格式边界。                                                                                                                                   |
|  `max_retries`   | 网络或服务端临时失败时，客户端自动重试的最大次数。                                                                                                                                                                               |

**Token 是什么（与 `max_tokens`、计费的关系）**

大模型在内部把文本切成 **token** 再计算与生成，可以理解为比「字」更细或更粗的片段（取决于分词器）：输出往往是**逐个 token 依次生成**。厂商计费、上下文长度上限、接口返回的用量统计，通常也都以 **token 数**为单位。

经验性换算（仅供直觉参考，**以各平台分词器实测为准**）：

- 中文：约 **1 个 token ≈ 1～1.8 个汉字**（因模型与分词器而异）。
- 英文：约 **1 个 token ≈ 3～4 个字母** 或更短/common 的词会合并为更少的 token。

因此设置 `max_tokens` 时，既是在限制**回答长度**，也直接影响**单次调用的费用上限**（在按输出 token 计费时）。

**Token / 字符 ↔ token 可视化工具（在线估算）**

- OpenAI：[Tokenizer](https://platform.openai.com/tokenizer)（主要适用于 OpenAI 系列分词规则）。
- 百度智能云：[Tokenizer](https://console.bce.baidu.com/support/#/tokenizer)。

![示例：在代码中设置模型参数](images/11/11-2-4-1.jpeg)

**标准化参数的适用范围与示例**

通过**动态传入模型名称**（以及 `temperature`、`api_key`、`base_url` 等），可以轻松切换使用不同模型。下面为 0.x 写法示例（直接使用 `ChatOpenAI`），参数名在 1.0 的 `init_chat_model` 中同样适用。

**0.x 写法示例：**

```python
from langchain_openai import ChatOpenAI

def chat_with_model(model_name, prompt):
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
    )
    return llm.invoke(prompt).content
```

**1.0 写法示例（推荐）：**

```python
from langchain.chat_models import init_chat_model

def chat_with_model(model_name, prompt):
    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        temperature=0.7,
        # 调用真实 OpenAI 时通常还需配置 OPENAI_API_KEY，或显式传入 api_key / base_url（兼容其它厂商端点时同理）
    )
    return model.invoke(prompt).content
```

> **说明**：上述标准化参数**并非对所有模型都生效**。通常仅对 LangChain 官方集成包（如 `langchain-openai`、`langchain-anthropic`）中的模型有完整支持；`langchain-community` 中的第三方模型可能不遵守这些规则，需以具体文档为准。

### 2.3 基本案例

下面是一个在代码中设置模型参数的简单示例，便于理解「参数 → 模型行为」的对应关系。

【案例源码】`案例与源码-2-LangChain框架/02-models_io/ModelIO_Params.py`

[ModelIO_Params.py](案例与源码-2-LangChain框架/02-models_io/ModelIO_Params.py ":include :type=code")

### 2.5 常见返回信息

```python
response = model.invoke("写一句关于春天的词，14 字以内")

print(type(model))             # 模型客户端，如 ChatOpenAI
print(type(response))          # AIMessage
print(type(response.content))  # str，模型生成的可见文本
```

**AIMessage 上常见字段**

| 字段 / 属性                         | 作用（读者怎么用）                                                                                                                                                                                                    |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `content`                           | 模型回复正文；业务上绝大多数时候只读它即可。                                                                                                                                                                          |
| `response_metadata`                 | 服务商返回的原始侧信息，常见包含：`token_usage`（如 `prompt_tokens`、`completion_tokens`、`total_tokens`）、`model_name`、`finish_reason`（如 `stop` 表示正常结束）、请求 `id` 等；排障、对账、看是否命中缓存时可查。 |
| `usage_metadata`                    | LangChain 侧整理的用量摘要（如 `input_tokens`、`output_tokens`、`total_tokens`），便于跨厂商做统一统计；与 `response_metadata` 里的 `token_usage` 含义相近，命名更统一。                                              |
| `tool_calls` / `invalid_tool_calls` | 工具调用场景下才有内容；未使用工具时多为空列表 `[]`。详见 [第 17 章 Tools 工具调用](17-Tools工具调用.md) 及后续 Agent 章节。                                                                                          |
| `additional_kwargs`                 | 厂商扩展字段（如部分接口的 `refusal` 等），按需查阅。                                                                                                                                                                 |

下面是与案例中一次调用同结构的示意（已省略正文过长部分，仅保留字段关系）：

```text
AIMessage(
  content='《鹧鸪天·春》\n一篙绿涨江南岸，…',   # 用户可见回复
  response_metadata={
    'token_usage': {
      'prompt_tokens': 14,
      'completion_tokens': 109,
      'total_tokens': 123,
      ...
    },
    'model_name': 'deepseek-chat',
    'finish_reason': 'stop',
    ...
  },
  usage_metadata={
    'input_tokens': 14,
    'output_tokens': 109,
    'total_tokens': 123,
    ...
  },
  tool_calls=[],
  ...
)
```

**其它消息类型（Human / System 等）的共性**

`HumanMessage`、`SystemMessage` 等与 `AIMessage` 一样，都继承自同一套消息基类，通常都带有 `type`、`content`；`response_metadata`、`tool_calls` 等主要出现在模型输出的 `AIMessage` 上。

| 属性名              | 典型出现场景       | 作用                                                                                     |
| ------------------- | ------------------ | ---------------------------------------------------------------------------------------- |
| `type`              | 各类消息           | `"human"` / `"ai"` / `"system"` / `"tool"` 等（有的文档里 `human` 也会写成 user 语义）。 |
| `content`           | 各类消息           | 多为 `str`；多模态时可能是结构化片段列表。                                               |
| `name`              | 可选               | 区分同名角色或多助手时使用；并非所有模型都支持。                                         |
| `response_metadata` | 多见于 `AIMessage` | 厂商返回的元数据与用量等。                                                               |
| `tool_calls`        | 多见于 `AIMessage` | 模型发起的工具调用；每项含 `name`、`args`、`id` 等。                                     |

小结：日常取回复用 `response.content`；要看本次耗了多少 token、用的哪个模型名、是否正常结束，看 `response.response_metadata` 或 `response.usage_metadata`；要做 Agent 再深入 `tool_calls`。

---

## 3、接入大模型

LangChain 支持通过不同集成包接入多种大模型，以下按「OpenAI 兼容」「DeepSeek」「通义千问」「智谱」分别说明，并给出选型与示例。

**集成总览（支持的模型提供商/厂商列表）**：https://docs.langchain.com/oss/python/integrations/providers/overview

### 3.1 接入 OpenAI 及兼容接口

- **文档**：
  - https://docs.langchain.com/oss/python/integrations/providers/openai （英文）
  - https://docs.langchain.org.cn/oss/python/integrations/providers/openai （中文）

**openai.OpenAI 与 langchain_openai.ChatOpenAI 的区别与选型：**

| 对比项       | openai.OpenAI                       | langchain_openai.ChatOpenAI                      |
| ------------ | ----------------------------------- | ------------------------------------------------ |
| **所属生态** | OpenAI 官方 Python SDK（`openai`）  | LangChain 生态（`langchain-openai`）             |
| **定位**     | 底层、纯粹的 API 调用               | LangChain 的「模型适配器」                       |
| **特点**     | 轻量、直接、贴近原生 API            | 可与 Prompt、Chain、Agent、Memory 等组件无缝配合 |
| **选型建议** | 仅需简单调用、不打算用 LangChain 时 | 需要链式编排、多组件协作时                       |

**类比**：可类比 Java 里的 **`Collection`** 与 <strong>`List<String>`</strong>——前者是通用、底层的集合接口，后者是在同一套类型体系下更具体、可组合、与业务（如遍历、流式处理）更贴合的一种形态；同理，`openai.OpenAI` 是通用 API 能力，`ChatOpenAI` 是 LangChain 生态里「带类型、可编排」的那一层。

**结论**：两者无绝对优劣，按需求选择——**简单调用用官方 SDK**，**复杂工作流用 LangChain 的 ChatOpenAI**。

**示例一：使用 OpenAI 官方 SDK（如对接 DeepSeek 兼容接口）**

【案例源码】`案例与源码-2-LangChain框架/02-models_io/ModelIO_OpenAI.py`

[ModelIO_OpenAI.py](案例与源码-2-LangChain框架/02-models_io/ModelIO_OpenAI.py ":include :type=code")

**示例二：使用 LangChain ChatOpenAI（0.x 写法，如对接通义千问兼容接口）**

【案例源码】`案例与源码-2-LangChain框架/02-models_io/ModelIO_ChatOpenAI.py`

[ModelIO_ChatOpenAI.py](案例与源码-2-LangChain框架/02-models_io/ModelIO_ChatOpenAI.py ":include :type=code")

**示例三：使用 init_chat_model 统一入口（1.0 写法，推荐）**

【案例源码】`案例与源码-2-LangChain框架/02-models_io/ModelIO_Init_chat_model.py`

[ModelIO_Init_chat_model.py](案例与源码-2-LangChain框架/02-models_io/ModelIO_Init_chat_model.py ":include :type=code")

### 3.2 接入 DeepSeek

- **文档**：
  - https://docs.langchain.com/oss/python/integrations/providers/deepseek （英文）
  - https://docs.langchain.org.cn/oss/python/integrations/providers/deepseek （中文）

> **为何官方文档里仍是 0.x 写法？** 集成页主要介绍的是**厂商包本身的用法**（如 `langchain-deepseek` 的 `ChatDeepSeek`），所以示例多为「直接导入厂商类并实例化」。1.0 的 `init_chat_model` 是 LangChain 提供的**统一入口**，内部会按 `model_provider` 路由到对应厂商实现；文档更新有先后，很多集成页尚未把 1.0 写法作为主推示例。两种写法目前都受支持，课程推荐以 `init_chat_model` 为主便于统一风格。

【案例源码】`案例与源码-2-LangChain框架/02-models_io/ModelIO_DeepSeek.py`

**0.x 写法：**

[ModelIO_DeepSeek.py](案例与源码-2-LangChain框架/02-models_io/ModelIO_DeepSeek.py ":include :type=code")

**1.0 写法一：OpenAI 兼容接口（通用、不依赖 langchain-deepseek）**

DeepSeek 提供 OpenAI 兼容的 API，因此可以用 `model_provider="openai"` + `base_url` 通过统一入口调用，**无需安装** `langchain-deepseek`，适合希望少装包、多模型统一用 `init_chat_model` 的场景。

```python
# .env 中配置 deepseek-api
import os
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("deepseek-api"),
    base_url="https://api.deepseek.com",
)

print(model.invoke("你是谁？").content)
```

**1.0 写法二：原生 DeepSeek 支持（若当前 LangChain 已注册 deepseek 厂商）**

DeepSeek **本身**有官方集成包 `langchain-deepseek`，若你使用的 LangChain 版本在 `init_chat_model` 中已注册 `model_provider="deepseek"`，则可用**原生**写法：不写 `base_url`，由 SDK 内部指定地址。需先安装 `pip install langchain-deepseek`，且是否支持以[官方文档](https://docs.langchain.com/oss/python/integrations/chat/deepseek)为准。

```python
# 需安装 langchain-deepseek：依赖包会注册 model_provider="deepseek"，代码中无需 import 包名
# .env 中配置 deepseek-api
import os
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("deepseek-api"),
)

print(model.invoke("你是谁？").content)
```

若运行时报错无法识别 `model_provider="deepseek"`，说明当前版本统一入口尚未接入该厂商，用上面的「写法一」即可。

### 3.3 接入通义千问（阿里云百炼）

- **控制台与 API 文档**：https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=2587654

通义千问（阿里云百炼）**既提供自家 DashScope 原生 API，也提供 OpenAI 兼容接口**。本节**不采用原生 API**，而是用 **OpenAI 兼容模式** 接入（上面 3.1 的 `ChatOpenAI` / `init_chat_model` + `base_url` + 阿里云 API Key），便于与其他模型统一写法；若需原生方式，可使用 `langchain_community` 的 `ChatTongyi` 等。

> **为何 [All providers](https://docs.langchain.com/oss/python/integrations/providers/all_providers) 里没有 ChatTongyi？** 该页按**厂商/提供商**（公司或平台）列出的，不是按「类名」或「产品名」。通义千问是**阿里云**的产品，因此会归在 **Alibaba Cloud** 下（点进 [Alibaba Cloud](https://docs.langchain.com/oss/python/integrations/providers/alibaba_cloud) 可看到其下的集成），而不会出现单独的 “Tongyi” 或 “ChatTongyi” 卡片；`ChatTongyi` 是代码里的类名，文档里对应的提供商是 **Alibaba Cloud**。

【案例源码】`案例与源码-2-LangChain框架/02-models_io/ModelIO_Qwen.py`

**0.x 写法：**

[ModelIO_Qwen.py](案例与源码-2-LangChain框架/02-models_io/ModelIO_Qwen.py ":include :type=code")

**1.0 写法（推荐）：**

```python
# .env 中配置 aliQwen-api
import os
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

print(model.invoke("你是谁？").content)
```

**原生 API 写法（ChatTongyi，直接调百炼/DashScope 接口）：**

不经过 OpenAI 兼容模式，使用 `langchain_community` 的 `ChatTongyi` 直接对接阿里云百炼原生 API，无需写 `base_url`（由 SDK 内部封装）。

```python
# 需先安装：pip install langchain-community dashscope；.env 中配置 aliQwen-api
import os
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(
    model="qwen-plus",
    api_key=os.getenv("aliQwen-api"),
)

print(model.invoke("你是谁？").content)
```

三种写法对比：**0.x / 1.0** 用兼容接口，与其他模型统一风格；**原生**用 `ChatTongyi`，依赖百炼 SDK，适合只用通义、希望走官方原生接口的场景。

### 3.4 接入智谱科技

- **文档**：https://docs.langchain.com/oss/python/integrations/chat/zhipuai
- **API Key**：在 [智谱开放平台](https://open.bigmodel.cn) 申请，配置到 `.env` 如 `ZHIPUAI_API_KEY=xxx`。

> **为何 [providers 总览页](https://docs.langchain.com/oss/python/integrations/providers/overview) 没有智谱？** 该页展示的是「Popular providers」等精选厂商，并非全部集成。智谱的文档在 **Chat 模型** 分类下（`/integrations/chat/zhipuai`），或通过 **langchain-community** 提供；完整列表可看 All providers 或侧栏的 Chat models。**智谱用的是自家开放平台 API，不是 OpenAI 兼容接口**——`ChatZhipuAI` 直接调智谱的接口，属于**原生集成**，所以能「直接支持」调用，而无需像通义那样走 `model_provider="openai"` + `base_url`。

**0.x 写法示例：**

```python
# 需先安装：pip install langchain-zhipuai（或 pip install langchain-community）；.env 中配置 ZHIPUAI_API_KEY
import os
from langchain_zhipuai import ChatZhipuAI

model = ChatZhipuAI(
    model="glm-4",  # 可选 glm-3-turbo、glm-4 等，见智谱文档
    temperature=0.7,
    api_key=os.getenv("ZHIPUAI_API_KEY"),
)

print(model.invoke("你是谁？").content)
```

若使用 `langchain_community`，则改为 `from langchain_community.chat_models import ChatZhipuAI`，参数一致。

---

**本章小结：**

- **Model I/O** 负责输入（[Prompt 模板](13-提示词与消息模板.md)）、模型调用、[输出解析](14-输出解析器.md) 三部分，是 LangChain 与各类大模型打交道的标准方式。
- **模型分类**：LLM（纯文本补全）、Chat Model（多轮对话，本课程重点）、Embeddings（向量化，用于 [RAG](19-RAG检索增强生成.md) 等，见 [第 18 章](18-向量数据库与Embedding实战.md)）；调用后返回 **AIMessage** 等消息对象，常用 `response.content`、`response.response_metadata` / `usage_metadata` 获取正文与用量。

**建议下一步：** 学习 [第 12 章 Ollama 本地部署与调用](12-Ollama本地部署与调用.md)，掌握在本地运行大模型并用 LangChain 调用；再学习 [第 13 章 提示词与消息模板](13-提示词与消息模板.md)、[第 14 章 输出解析器](14-输出解析器.md)，与本章的 Model I/O 三件套形成「输入 → 模型 → 输出解析」闭环；进而用 [第 15 章 LCEL 与链式调用](15-LCEL与链式调用.md) 将三件套串成链。
