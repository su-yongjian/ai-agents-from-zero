# 示例：使用 LangChain 调用本地 Ollama 服务
# 对应文档 10 章 2.5.2。无需 API Key，适合本地开发与离线使用。
#
# 前置：已安装 Ollama，并执行过 ollama pull qwen:4b（或 llama3 等）
# 依赖：pip install langchain-ollama

# ========== 1. 导入 ==========
from langchain_ollama import ChatOllama

# ========== 2. 连接本机 Ollama 服务 ==========
# 模型名需与 ollama list 中的名称一致；默认服务地址为 http://localhost:11434
llm = ChatOllama(
    model="qwen:4b",                      # 或 "llama3"、"deepseek-r1:14b" 等
    base_url="http://localhost:11434",
    temperature=0.7,
)

# ========== 3. 调用并打印回复 ==========
response = llm.invoke("你好，请用一句话介绍你自己。")
print(response.content)

# 若需与 Prompt、Chain、Agent 等结合，将 llm 传入对应组件即可，用法与 ChatOpenAI 一致。
