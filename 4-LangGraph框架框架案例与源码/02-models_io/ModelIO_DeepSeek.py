# 示例：使用官方 LangChain 集成包 langchain-deepseek 调用 DeepSeek
# 对应文档 10 章 1.6.2。无需手动写 base_url，SDK 内已封装。
#
# 依赖：pip install langchain-deepseek
# 运行前：在 .env 中配置 deepseek-api

# ========== 1. 导入与环境 ==========
import os
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

# ========== 2. 初始化 DeepSeek 聊天模型 ==========
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("deepseek-api"),
)

# ========== 3. 调用并打印回复 ==========
print(model.invoke("什么是 LangChain？100 字以内回答，简洁。").content)
