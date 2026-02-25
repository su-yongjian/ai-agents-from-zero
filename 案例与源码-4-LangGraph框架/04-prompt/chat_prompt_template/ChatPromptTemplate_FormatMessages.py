"""
【案例】from_messages 创建模板 + format_messages / invoke / format 的用法（对应教程 1.7 节）

一、from_messages 是什么？
  - ChatPromptTemplate.from_messages([...]) 是「用消息列表创建对话模板」的常用写法。
  - 列表里每条消息可以是：(role, content) 元组、dict、或 Message 类实例。
  - 例如：("system", "你是{role}")、("human", "{question}")，其中的 {role}、{question} 就是占位符。

二、填充模板的三种方式（初学者重点区分）：
  - format_messages(**kwargs)：传入占位符变量，返回「消息列表」List[BaseMessage]，
    可直接给 model.invoke(prompt_value) 使用，或用来调试、查看最终消息结构。
  - invoke({"变量名": 值})：传入字典，同样返回消息列表（PromptValue），
    常用 .to_string() 转成整段文本查看，或直接把返回值交给 model.invoke。
  - format(**kwargs)：传入占位符变量，返回的是「拼接后的纯文本字符串」，
    不是消息列表，适合只看一整段文字时用，不能直接替代 format_messages 交给聊天模型。

三、小结：和模型对话时，用 format_messages 或 invoke 得到消息列表，再 model.invoke(消息列表)。
"""

import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# 用 from_messages 创建模板：一条 system（带 {role}）、一条 human（带 {question}）
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个{role}，请回答我提出的问题"),
        ("human", "请回答:{question}")
    ]
)

# ---------- 方式一：format_messages ----------
# 关键字参数：role、question 对应模板里的 {role}、{question}
# 返回值是「消息列表」，可交给 llm.invoke(prompt_value)
# prompt_value = chat_prompt.format_messages(role="python开发工程师", question="冒泡排序怎么写")
prompt_value = chat_prompt.format_messages(**{"role": "python开发工程师", "question": "堆排序怎么写"})
print(prompt_value)

print()
# ---------- 方式二：invoke（传字典）----------
# 传入一个字典，键为占位符变量名，值为要填充的内容；返回的也是「消息列表」（PromptValue）
# .to_string() 可把整段对话转成纯文本，方便打印查看
prompt_value2 = chat_prompt.invoke({"role": "python开发工程师", "question": "堆排序怎么写"})
print(prompt_value2.to_string())

print()

# ---------- 方式三：format（注意：返回的是字符串，不是消息列表）----------
# 适合只想得到「一整段文本」时用；若要把对话发给聊天模型，请用 format_messages 或 invoke
prompt_value3 = chat_prompt.format(**{"role": "python开发工程师", "question": "快速排序怎么写"})
print(prompt_value3)


# llm = init_chat_model(
#     model="qwen-plus",
#     model_provider="openai",
#     api_key=os.getenv("aliQwen-api"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )
# print()
# print("======================")
#
# result = llm.invoke(prompt_value)
# print(result)
# print(result.content)
