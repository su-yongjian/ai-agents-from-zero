"""
【案例】使用 init_chat_model 统一入口调用大模型（1.0 推荐写法）

对应教程章节：第 11 章 - Model I/O 与模型接入 → 3、接入大模型

知识点速览：
一套写法通过 model + base_url + api_key 切换不同厂商，无需改类名。DeepSeek 等可由 base_url 推断
model_provider，无需显式指定。invoke 既可传单条字符串，也可传与 ChatOpenAI 相同格式的 messages 列表。
依赖 langchain、langchain-openai（或对应 provider 包），运行前配置 .env。
"""

# ========== 1. 导入与环境 ==========
import os
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

# ========== 2. 实例化模型（无需指定 model_provider 时，可由 base_url 推断） ==========
model = init_chat_model(
    model="deepseek-chat",
    api_key=os.getenv("deepseek-api"),
    base_url="https://api.deepseek.com",
)

# ========== 3. 调用并取正文（两种写法均可） ==========
# 写法一：单条用户输入字符串（最简）
# 仅传 str 时，框架会当作一条 user/human 消息，无法在此形式下单独写 system；要 system 请用写法二或消息对象。
print(model.invoke("你是谁？").content)

# 写法二：与 ModelIO_ChatOpenAI.py 相同的多角色消息列表（system + user）
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"},
]
response = model.invoke(messages)
print(response.content)
