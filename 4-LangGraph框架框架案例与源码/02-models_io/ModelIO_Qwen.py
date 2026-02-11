# 示例：使用 LangChain 通义千问集成（ChatTongyi）调用阿里云百炼
# 对应文档 10 章 1.6.3。通义也可用 1.6.1 的 ChatOpenAI + base_url 兼容方式，本文件演示原生集成。
#
# 依赖：pip install langchain-community dashscope
# 若报错可尝试：pip install --upgrade --force-reinstall cffi
# 运行前：在 .env 中配置 aliQwen-api

# ========== 1. 导入与环境 ==========
import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

# ========== 2. 初始化通义千问聊天模型 ==========
chat_llm = ChatTongyi(
    model="qwen-plus",
    api_key=os.getenv("aliQwen-api"),
    streaming=True,
)

# ========== 3. 调用方式一：invoke 一次性返回 ==========
print(chat_llm.invoke("你是谁").content)

print("*" * 60)

# ========== 4. 调用方式二：stream 流式返回 ==========
for chunk in chat_llm.stream([HumanMessage(content="你好，你是谁")], streaming=True):
    print(chunk.content, end="")
print()
