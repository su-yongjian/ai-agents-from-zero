"""
【第 11 章 - 模型调用：同步 stream（流式输出）】
对应笔记：11-提示词与输出解析.md → 1.4 模型调用方法 → 1.4.2 流式调用

知识点速览：
- stream：流式响应。模型「生成一点就返回一点」，不必等整段话写完再一次性返回。
- 适用场景：聊天界面「打字机」效果、长文生成，用户能边等边看到内容，体验更好。
- 与 invoke 区别：invoke 等全部生成完再返回；stream 返回一个可迭代对象，用 for 循环逐块取内容。
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

# ---------- 1. 实例化模型 ----------
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ---------- 2. 构建多角色消息（同 invoke）----------
messages = [
    SystemMessage(content="你叫小问，是一个乐于助人的AI人工助手"),
    HumanMessage(content="你是谁")
]

# ---------- 3. 流式调用：model.stream(messages) ----------
# stream 返回的是一个「可迭代对象」，每次迭代得到一小块（chunk），类型一般为 AIMessageChunk。
# 不会等模型全部生成完才返回，而是边生成边 yield，实现「打字机」效果。
response = model.stream(messages)
print(f"响应类型：{type(response)}")

# 逐块打印：chunk.content 是当前这一小段的文本；end="" 表示不换行，flush=True 表示立即输出到屏幕
for chunk in response:
    print(chunk.content, end="", flush=True)
print("\n")
