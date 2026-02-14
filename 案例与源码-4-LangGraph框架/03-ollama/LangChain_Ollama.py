"""
LangChain + Ollama 本地大模型对话示例

【这个文件是干什么的？】
用 Python 调用本机运行的 Ollama 大模型（如 qwen2.5），通过 LangChain 的接口发问题、拿回答。

【使用前要准备什么？】
1. 安装依赖（在终端执行）：
   pip install -qU langchain-ollama
   pip install -U ollama

2. 本机要先安装并启动 Ollama，并拉取好模型（如：ollama run qwen:4b）。
   Ollama 默认在 http://localhost:11434 提供服务。
"""

from langchain_ollama import ChatOllama

# ---------- 第一步：创建“聊天客户端” ----------
# ChatOllama 是 LangChain 里专门用来和 Ollama 对话的类，可以理解成“连接本地模型的客户端”
# 参数说明：
#   base_url  ：Ollama 服务地址，本机默认就是 "http://localhost:11434"
#   model     ：要用的模型名，必须是你用 ollama 拉取过的，如 "qwen:4b"
#   reasoning ：是否开启“深度思考”模式；False 表示不开启，回答更快、更省资源
model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen:4b",
    reasoning=False,
)

# ---------- 第二步：发一条消息并打印回复 ----------
# invoke(问题) 会把你写的问题发给上面的模型，并返回一个“消息对象”
# 这个对象里包含模型生成的文字等内容，直接 print 会显示摘要信息
response = model.invoke("什么是LangChain，100字以内回答")
print(response)

# 【可选】如果你只想看到模型回复的纯文字，可以这样取：
# print(response.content)

