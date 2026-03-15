"""
【案例】节点缓存（Node Caching）：为节点配置 CachePolicy(ttl=8)，编译时传入 InMemoryCache()，相同输入在 ttl 秒内直接返回缓存结果，避免重复执行耗时逻辑。

对应教程章节：第 24 章 - LangGraph API：节点、边与进阶 → 1、Graph API 之 Node（节点）

知识点速览：
- add_node(..., cache_policy=CachePolicy(ttl=秒数))：该节点的输出会按「输入」生成缓存键，ttl 内命中则跳过执行。
- compile(cache=InMemoryCache())：指定使用内存缓存实现；首次执行走节点，后续相同输入在 ttl 内直接返回缓存。
- set_entry_point / set_finish_point 可替代 add_edge(START, node) 与 add_edge(node, END)，用于单节点图或明确入口/出口。
"""

import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy


class State(TypedDict):
    x: int
    result: int


builder = StateGraph(State)


def expensive_node(state: State) -> dict[str, int]:
    """模拟耗时计算（sleep 3 秒），用于观察缓存命中时不再执行。"""
    time.sleep(3)
    return {"result": state["x"] * 2}


# 为该节点配置缓存，ttl=8 秒
builder.add_node(
    node="expensive_node",
    action=expensive_node,
    cache_policy=CachePolicy(ttl=8),
)
builder.set_entry_point("expensive_node")
builder.set_finish_point("expensive_node")

# 编译时指定使用内存缓存
app = builder.compile(cache=InMemoryCache())

# 第一次执行：无缓存，耗时约 3 秒
print("第一次执行（无缓存，耗时 3 秒）：")
print(app.invoke({"x": 5}))

# 第二次执行：命中缓存，立即返回
print("\n第二次运行利用缓存并快速返回：")
print(app.invoke({"x": 5}))

# 等待 ttl 过期后再次执行，将重新计算
print("\n等待 8 秒，缓存过期...")
time.sleep(8)
print("8 秒后第三次执行（重新计算，耗时 3 秒）：")
print(app.invoke({"x": 5}))
