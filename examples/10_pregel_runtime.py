"""10 · 底层运行时 Pregel：StateGraph 编译后到底变成了什么？

📖 对应文档：LangGraph runtime（Pregel）
▶ 运行：python examples/10_pregel_runtime.py

StateGraph.compile() 的产物就是一个 Pregel 对象。Pregel = Actors（节点）+ Channels（通道）：
- 每个 State 字段 → 一个 channel（无 reducer 是 LastValue，有 reducer 是 BinaryOperatorAggregate）
- 每个节点 → 一个 actor，订阅某些 channel，执行后写入某些 channel

执行按 super-step（BSP 模型）进行，每一步三个阶段：
    Plan   ：找出订阅了“上一步被更新的 channel”的节点
    Execute：这些节点【并行】执行；本步内的写入彼此不可见
    Update ：把所有写入应用到 channel（有冲突时由 reducer 合并）
checkpoint 就是在每个 super-step 结束时保存所有 channel 的值。

这也解释了：
- 为什么并行节点写同一个字段需要 reducer（同一步多个写入要合并）；
- 为什么 recursion_limit 数的是 super-step；
- 为什么失败/中断后从“节点开头”重跑（checkpoint 只在步与步之间）。
"""

import operator

from langgraph.channels import EphemeralValue, LastValue
from langgraph.graph import START, StateGraph
from langgraph.pregel import NodeBuilder, Pregel
from typing_extensions import Annotated, TypedDict


def demo_raw_pregel():
    # 直接用 Pregel API：node1 读 a 写 b，node2 读 b 写 c
    node1 = NodeBuilder().subscribe_only("a").do(lambda x: x + x).write_to("b")
    node2 = NodeBuilder().subscribe_only("b").do(lambda x: x + "!").write_to("c")
    app = Pregel(
        nodes={"node1": node1, "node2": node2},
        channels={"a": EphemeralValue(str), "b": LastValue(str), "c": EphemeralValue(str)},
        input_channels=["a"],
        output_channels=["b", "c"],
    )
    print("① 原生 Pregel：", app.invoke({"a": "foo"}))


def demo_compiled_state_graph():
    class State(TypedDict):
        latest: str
        log: Annotated[list[str], operator.add]

    graph = StateGraph(State).add_node("n", lambda s: {"log": ["n"]}).add_edge(START, "n").compile()
    print("② StateGraph 编译后的类型：", type(graph).__mro__[1].__name__)
    for name, channel in graph.channels.items():
        if not name.startswith("branch:"):
            print(f"   channel {name!r:<16} → {type(channel).__name__}")
    print("   （latest 无 reducer → LastValue；log 有 operator.add → BinaryOperatorAggregate）")


if __name__ == "__main__":
    demo_raw_pregel()
    demo_compiled_state_graph()
