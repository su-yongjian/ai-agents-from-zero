"""
【案例】使用 Redis Stack 持久化对话历史：RunnableWithMessageHistory + RedisChatMessageHistory

对应教程章节：第 16 章 - 记忆与对话历史 → 6、案例代码 → 6.2 持久化：Redis 存储 → Redis Stack 示例

本示例与 Memory_RedisChatMessageHistory.py 逻辑相同，仅默认连接 Redis Stack 常用端口 26379。
Redis Stack 与原生 Redis 协议兼容，对话历史仍使用 list/string 等基础结构；使用 Redis Stack 可额外使用
RedisInsight 可视化管理数据，或为后续向量搜索等扩展预留环境。未安装 Redis Stack 时可用 REDIS_URL 覆盖为原生 Redis 地址。

知识点速览：
- 默认 REDIS_URL=redis://localhost:26379（Redis Stack 常见映射 -p 26379:6379）。
- 启动 Redis Stack（带 RedisInsight 用 redis/redis-stack）：docker run -d --name redis-stack -p 26379:6379 -p 8001:8001 redis/redis-stack
- 其余用法同 Memory_RedisChatMessageHistory.py。
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
import os
import redis
from loguru import logger

# 默认连接 Redis Stack（端口 26379）；可通过环境变量 REDIS_URL 覆盖
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:26379")


def _check_redis():
    """启动时检查 Redis/Redis Stack 是否可达，不可达时给出明确提示后退出。"""
    try:
        r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        r.close()
    except (redis.ConnectionError, redis.ResponseError) as e:
        logger.error(
            "Redis Stack / Redis 连接失败（{}）。请先启动 Redis Stack，例如：\n"
            "  docker run -d --name redis-stack -p 26379:6379 -p 8001:8001 redis/redis-stack\n"
            "若使用原生 Redis 或其他端口，可设置环境变量：REDIS_URL=redis://localhost:端口",
            REDIS_URL,
        )
        raise SystemExit(1) from e


_check_redis()

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder("history"), ("human", "{question}")]
)


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """为每个 session_id 创建/返回对应的 Redis 历史实例，实现持久化存储。"""
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
    )


chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
config = RunnableConfig(configurable={"session_id": "user-001"})

print("开始对话（Redis Stack 版，输入 'quit' 退出）")
while True:
    question = input("\n输入问题：")
    if question.lower() in ["quit", "exit", "q"]:
        break
    response = chain.invoke({"question": question}, config)
    logger.info(f"AI回答:{response.content}")
    redis_client.save()
