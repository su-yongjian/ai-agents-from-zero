# 示例：模型常用参数在代码中的用法（temperature、max_tokens 等）
# 对应文档 10 章 1.5.2。演示标准化参数如何影响模型行为（如 temperature 控制随机性）。
#
# 依赖：pip install langchain langchain-openai
# 运行前：在 .env 中配置 deepseek-api

# ========== 1. 导入与环境 ==========
import os
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

# ========== 2. 实例化时设置常用参数 ==========
model = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("deepseek-api"),
    base_url="https://api.deepseek.com",
    temperature=0.7,   # 0～1，越高越随机；此处略高便于看到多次输出差异
    # max_tokens=256,  # 可选：限制单次回复长度
)

# ========== 3. 多次调用观察参数效果（如 temperature 对多样性的影响） ==========
for i in range(3):
    print(f"--- 第 {i + 1} 次 ---")
    print(model.invoke("写一句关于春天的词，14 字以内").content)
