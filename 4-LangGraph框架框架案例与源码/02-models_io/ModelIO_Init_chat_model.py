# 示例：使用 LangChain 1.0 统一入口 init_chat_model 调用大模型
# 对应文档 10 章 1.6.1 示例三。一套写法可切换不同厂商（通过 model + base_url + api_key）。
#
# 依赖：pip install langchain langchain-openai（或对应 provider 包）
# 运行前：在 .env 中配置 deepseek-api

# ========== 1. 导入与环境 ==========
import os
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

# ========== 2. 实例化模型（无需指定 model_provider 时，可由 base_url 推断） ==========
model = init_chat_model(
    model="deepseek-chat",
    api_key=os.getenv("deepseek-api"),
    base_url="https://api.deepseek.com",
)

# ========== 3. 调用并取正文 ==========
print(model.invoke("你是谁").content)
