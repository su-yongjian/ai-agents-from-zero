"""
【第 11 章 - 模型调用：异步 abatch（异步批量调用）】
对应笔记：11-提示词与输出解析.md → 1.4 模型调用方法 → 1.4.3 批处理（异步）

知识点速览：
- abatch：异步版的 batch。批量提交多条输入，在等待过程中不阻塞主线程，适合高并发或与其他异步任务配合。
- 用法：abatch 是异步方法，必须在 async 函数里用 await 调用，得到的结果格式与 batch 一致（列表，一一对应）。
- 入口：用 asyncio.run() 运行封装好的 async 函数。
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# abatch 本示例用「字符串列表」作为输入，无需单独导入 Message 类型

load_dotenv()

# ---------- 1. 实例化模型 ----------
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ---------- 2. 准备批量问题（与同步 batch 相同）----------
questions = [
    "什么是redis?简洁回答，字数控制在100以内",
    "Python的生成器是做什么的？简洁回答，字数控制在100以内",
    "解释一下Docker和Kubernetes的关系?简洁回答，字数控制在100以内"
]


# ---------- 3. 异步批量调用（在 async 函数中）----------
async def async_batch_call():
    # await model.abatch(questions)：异步批量处理，返回的仍是「问题与回答一一对应」的列表
    response = await model.abatch(questions)
    print(f"响应类型：{type(response)}")

    for q, r in zip(questions, response):
        print(f"问题：{q}\n回答：{r.content}\n")


# ---------- 4. 运行异步函数 ----------
if __name__ == "__main__":
    asyncio.run(async_batch_call())
