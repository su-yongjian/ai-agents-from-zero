"""
【案例】模型标准化参数：temperature、max_tokens 等

对应教程章节：第 11 章 - Model I/O 与模型接入 → 2.2 常用模型参数、2.4 常见返回信息

知识点速览：
演示 LangChain 标准化参数如何影响模型行为：temperature 控制输出随机性（0 更确定，越大越随机），
max_tokens 限制单次回复长度。依赖 langchain、langchain-openai，运行前在 .env 中配置 deepseek-api。
"""

# ========== 1. 导入与环境 ==========
import os
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

# ========== 2. 实例化时设置常用参数 ==========
# temperature：控制输出随机性，0 更确定、高重复，越大越随机、越有创意。
# 通常取 0~2，源于 OpenAI API 约定；具体上下界以所用 API 文档为准。
# 超过 2（如 2.1）可能被部分接口拒绝或截断，且与 2.0 效果差异不大，建议不超过 2。
model = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("deepseek-api"),
    base_url="https://api.deepseek.com",
    temperature=0.7,  # 0～1，越高越随机；此处略高便于看到多次输出差异
    # max_tokens=256,  # 可选：限制单次回复长度
)

# content='《鹧鸪天·春》\n一篙绿涨江南岸，半日风裁陌上烟。\n\n注：我的原创短句以14字勾勒出春日江南的典型意象。“绿涨”写春水漫堤的动态之美，“风裁烟”化无形春风为灵巧裁刀，赋予画面流动的韵律感。全句未直接抒情，而通过水墨画般的空间构图（岸上烟柳、水中倒影）与精巧动词，传递出对苏醒大地的敏锐感知。' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 109, 'prompt_tokens': 14, 'total_tokens': 123, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 14}, 'model_provider': 'openai', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache_new_kvcache', 'id': '0c5c4950-3711-44f0-addd-1ee52a31e029', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019d29ba-2fd1-7a30-8722-64b807e27223-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 14, 'output_tokens': 109, 'total_tokens': 123, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}
print(model.invoke("写一句关于春天的词，14 字以内"))
# <class 'langchain_openai.chat_models.base.ChatOpenAI'>
print(type(model))
# <class 'str'>
print(type(model.invoke("写一句关于春天的词，14 字以内").content))
# <class 'langchain_core.messages.ai.AIMessage'>
print(type(model.invoke("写一句关于春天的词，14 字以内")))

# ========== 3. 多次调用观察参数效果（如 temperature 对多样性的影响） ==========
for i in range(3):
    print(f"--- 第 {i + 1} 次 ---")
    print(model.invoke("写一句关于春天的词，14 字以内").content)


"""
【输出示例】温度为 2.0 时，输出如下结果
莺惊柳浪春犁响，一鞭云水碧
**诗家酥雨润，闲趁卖花声。**
寒威已退先春雨，万枝吐翠流云间。
"""

"""
【输出示例】温度为 0 时，输出如下结果
风软一溪云，花明两岸春。
风软一溪云，花明两岸春。
风软一溪云，花明两岸春。
"""
