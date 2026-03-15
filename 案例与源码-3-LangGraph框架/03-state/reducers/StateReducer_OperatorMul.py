"""
【案例】operator.mul 作为 Reducer（数值相乘）的「陷阱」演示：LangGraph 会用类型默认值（float 的 0.0）先做一次规约，导致 0.0 * 初始值 = 0，后续乘法始终为 0；理解后可用自定义 Reducer 解决。

对应教程章节：第 23 章 - LangGraph API：图与状态 → 2、Graph API 之 State（状态）

知识点速览：
- 未指定 Reducer 时，LangGraph 会先用「类型默认值」与 invoke 传入的初始值做一次规约。加法恒等元是 0，乘法恒等元是 1，但 float 默认值是 0.0，导致乘法第一次就变成 0。
- operator.mul 作为 Reducer 时：第一次规约为 0.0 * 初始值 = 0.0，之后所有节点乘上去仍是 0。这是设计使然，不是 bug。
- 解决方式：对该字段使用自定义 Reducer，在函数内判断「若 current == 0.0 则视为第一次，用 1.0 * update」再返回。参见 StateReducer_Custom.py。
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class MultiplyState(TypedDict):
    factor: Annotated[float, operator.mul]


def multiplier(state: MultiplyState) -> dict:
    return {"factor": 2.0}


def run_demo():
    print("4. operator.mul Reducer（数值相乘）演示:")
    builder = StateGraph(MultiplyState)
    builder.add_node("multiplier", multiplier)
    builder.add_edge(START, "multiplier")
    builder.add_edge("multiplier", END)
    graph = builder.compile()

    result = graph.invoke({"factor": 5.0})
    print(f"初始状态: {{'factor': 5.0}}")
    print(f"执行结果: {result}")
    print("说明: 因 float 默认 0.0 先参与规约，0.0 * 5.0 = 0.0，后续乘 2.0 仍为 0.0；乘法场景请用自定义 Reducer。\n")


if __name__ == "__main__":
    run_demo()
