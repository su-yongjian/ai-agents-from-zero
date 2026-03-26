"""
【案例】Command 对象：节点可返回 Command(update=..., goto=节点或END)，在「更新状态」的同时「指定下一跳」，实现状态更新与路由一步完成，适合人机闭环与多智能体交接。

对应教程章节：第 24 章 - LangGraph API：节点、边与进阶 → 3、Send、Command 与 Runtime 上下文

知识点速览：
- 节点函数返回 Command(update=字典, goto=节点名或END)，表示先对 state 应用 update，再跳转到 goto；若 goto=END 则结束图。
- 与条件边的区别：条件边只做路由；Command 在路由的同时写入 state，适合「决策节点」既要写日志/标记又要转交下一节点的场景。
- 常用于 decision_agent 根据消息内容路由到 math_agent / translation_agent，或检测到任务完成后 goto=END 并写结束消息。
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# 全局常量：统一递归限制，便于维护
RECURSION_LIMIT = 50


# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]  # 自动合并消息
    current_agent: str
    task_completed: bool


# 决策代理（核心路由节点）
def decision_agent(state: AgentState) -> Command[AgentState]:
    """根据消息内容路由代理，任务完成则直接终止"""
    print("执行节点: decision_agent")
    # 优先终止流程（核心防循环逻辑）
    if state["task_completed"]:
        print("✅ 检测到任务已完成，直接终止流程")
        return Command(
            update={"messages": [("system", "所有任务处理完成，流程正常结束")]},
            goto=END,
        )
    # 提取消息文本（兼容空消息）
    last_message = state["messages"][-1] if state["messages"] else ("", "")
    last_msg_content = last_message[1]
    print(f"最新消息文本: {last_msg_content}")

    # 动态路由
    if "数学" in last_msg_content:
        print("✅ 检测到数学任务，路由到数学代理")
        return Command(
            update={
                "messages": [("system", "路由到数学代理")],
                "current_agent": "math_agent",
            },
            goto="math_agent",
        )
    elif "翻译" in last_msg_content:
        print("✅ 检测到翻译任务，路由到翻译代理")
        return Command(
            update={
                "messages": [("system", "路由到翻译代理")],
                "current_agent": "translation_agent",
            },
            goto="translation_agent",
        )
    else:
        print("❌ 未识别任务类型，标记任务完成并终止")
        return Command(
            update={"messages": [("system", "任务完成")], "task_completed": True},
            goto=END,
        )


# 数学代理（业务节点）
def math_agent(state: AgentState) -> Command[AgentState]:
    """处理数学计算任务，完成后返回决策代理"""
    print("执行节点: math_agent")
    result = "2 + 2 = 4"
    print(f"计算结果: {result}")
    return Command(
        update={
            "messages": [("assistant", f"数学计算结果: {result}")],
            "current_agent": "decision_agent",
            "task_completed": True,
        },
        goto="decision_agent",
    )


# 翻译代理（业务节点）
def translation_agent(state: AgentState) -> Command[AgentState]:
    """处理中英翻译任务，完成后返回决策代理"""
    print("执行节点: translation_agent")
    translation = "Hello -> 你好"
    print(f"翻译结果: {translation}")
    return Command(
        update={
            "messages": [("assistant", f"翻译结果: {translation}")],
            "current_agent": "decision_agent",
            "task_completed": True,
        },
        goto="decision_agent",
    )


def main():
    """演示Command基础用法：状态更新+动态路由+流程终止"""
    print("=== Command 基础演示（LangGraph 1.0.6）===\n")

    # 1. 构建状态图
    builder = StateGraph(AgentState)
    builder.add_node("decision_agent", decision_agent)
    builder.add_node("math_agent", math_agent)
    builder.add_node("translation_agent", translation_agent)

    # 2. 定义边（完整节点关系）
    builder.add_edge(START, "decision_agent")
    builder.add_edge("math_agent", "decision_agent")
    builder.add_edge("translation_agent", "decision_agent")
    builder.add_edge("decision_agent", END)

    # 3. 编译图
    graph = builder.compile()

    # 测试1：数学任务
    print("【测试1: 数学任务】")
    initial_state = {
        "messages": [("user", "我需要计算数学题")],
        "current_agent": "user",
        "task_completed": False,
    }
    print("初始状态:", initial_state)
    result = graph.invoke(initial_state, recursion_limit=RECURSION_LIMIT)
    print(
        "最终状态(简化):", {k: v for k, v in result.items() if k != "messages"}
    )  # 简化输出
    print("\n" + "-" * 50 + "\n")

    # 测试2：翻译任务
    print("【测试2: 翻译任务】")
    initial_state = {
        "messages": [("user", "我需要翻译文本")],
        "current_agent": "user",
        "task_completed": False,
    }
    print("初始状态:", initial_state)
    result = graph.invoke(initial_state, recursion_limit=RECURSION_LIMIT)
    print("最终状态(简化):", {k: v for k, v in result.items() if k != "messages"})
    print("\n" + "-" * 50 + "\n")

    # 测试3：未识别任务
    print("【测试3: 未识别任务类型】")
    initial_state = {
        "messages": [("user", "你好")],
        "current_agent": "user",
        "task_completed": False,
    }
    print("初始状态:", initial_state)
    result = graph.invoke(initial_state, recursion_limit=RECURSION_LIMIT)
    print("最终状态(简化):", {k: v for k, v in result.items() if k != "messages"})

    # 新增：可视化图结构（教学演示必备）
    print("\n=== 图结构可视化 ===")
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()

"""
【输出示例】
=== Command 基础演示（LangGraph 1.0.6）===

【测试1: 数学任务】
初始状态: {'messages': [('user', '我需要计算数学题')], 'current_agent': 'user', 'task_completed': False}
执行节点: decision_agent
最新消息文本: 我需要计算数学题
✅ 检测到数学任务，路由到数学代理
执行节点: math_agent
计算结果: 2 + 2 = 4
执行节点: decision_agent
✅ 检测到任务已完成，直接终止流程
最终状态(简化): {'current_agent': 'decision_agent', 'task_completed': True}

--------------------------------------------------

【测试2: 翻译任务】
初始状态: {'messages': [('user', '我需要翻译文本')], 'current_agent': 'user', 'task_completed': False}
执行节点: decision_agent
最新消息文本: 我需要翻译文本
✅ 检测到翻译任务，路由到翻译代理
执行节点: translation_agent
翻译结果: Hello -> 你好
执行节点: decision_agent
✅ 检测到任务已完成，直接终止流程
最终状态(简化): {'current_agent': 'decision_agent', 'task_completed': True}

--------------------------------------------------

【测试3: 未识别任务类型】
初始状态: {'messages': [('user', '你好')], 'current_agent': 'user', 'task_completed': False}
执行节点: decision_agent
最新消息文本: 你好
❌ 未识别任务类型，标记任务完成并终止
最终状态(简化): {'current_agent': 'user', 'task_completed': True}

=== 图结构可视化 ===
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__(<p>__start__</p>)
        decision_agent(decision_agent)
        math_agent(math_agent)
        translation_agent(translation_agent)
        __end__(<p>__end__</p>)
        __start__ --> decision_agent;
        decision_agent --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
"""
