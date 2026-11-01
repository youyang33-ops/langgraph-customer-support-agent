"""01 · Graph API 基础：State、Node、Edge、Reducer。

📖 对应文档：Overview（hello world）、Graph API overview、Use the graph API
▶ 运行：python examples/01_graph_api_basics.py

看完你应该能回答：
- StateGraph 的三要素是什么？编译（compile）做了什么？
- 节点返回的 dict 是怎么合并进 State 的？什么时候需要 reducer？
- input_schema / output_schema / 私有 State 有什么用？
"""

import operator
from typing import Annotated

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Overwrite
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# 1. Hello world（文档 Overview 里的第一个例子）
# ---------------------------------------------------------------------------
def demo_hello_world():
    def mock_llm(state: MessagesState):
        return {"messages": [{"role": "ai", "content": "hello world"}]}

    graph = StateGraph(MessagesState)  # 1) 定义 State
    graph.add_node(mock_llm)  # 2) 添加节点（不写名字时默认用函数名）
    graph.add_edge(START, "mock_llm")  # 3) 连边
    graph.add_edge("mock_llm", END)
    app = graph.compile()  # 4) 编译：检查结构 + 生成可执行的 Pregel 对象

    result = app.invoke({"messages": [{"role": "user", "content": "hi!"}]})
    print("① hello world:", [m.content for m in result["messages"]])


# ---------------------------------------------------------------------------
# 2. Reducer：决定“节点返回值”如何合并进 State
# ---------------------------------------------------------------------------
class ReducerState(TypedDict):
    latest: str  # 没有 reducer：后写覆盖先写
    history: Annotated[list[str], operator.add]  # 有 reducer：列表拼接


def demo_reducers():
    builder = StateGraph(ReducerState)
    builder.add_node("a", lambda s: {"latest": "a", "history": ["a"]})
    builder.add_node("b", lambda s: {"latest": "b", "history": ["b"]})
    builder.add_node("clear", lambda s: {"history": Overwrite(["(已清空)"])})  # 绕过 reducer 直接覆盖
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "clear")
    app = builder.compile()

    print("② reducer（每一步之后的完整 State）：")
    for snapshot in app.stream({"latest": "", "history": ["start"]}, stream_mode="values"):
        print("   ", snapshot)
    # 注意：带 operator.add 的字段，返回 [] 并不能清空它（空列表被“加”进去而已），要用 Overwrite


# ---------------------------------------------------------------------------
# 3. 输入 / 输出 schema + 节点私有 State
# ---------------------------------------------------------------------------
class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    answer: str


class OverallState(InputState, OutputState):
    notes: str  # 内部字段，不对外暴露


class PrivateState(TypedDict):
    secret: str  # 只在 think → answer 两个节点之间传递


def think(state: InputState) -> PrivateState:
    return {"secret": f"关于「{state['question']}」的草稿"}


def answer(state: PrivateState) -> OutputState:
    return {"answer": state["secret"] + " → 最终答案"}


def demo_schemas():
    builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    builder.add_node("think", think)
    builder.add_node("answer", answer)
    builder.add_edge(START, "think")
    builder.add_edge("think", "answer")
    builder.add_edge("answer", END)
    app = builder.compile()
    print("③ 输出只有 answer：", app.invoke({"question": "LangGraph 是什么"}))


# ---------------------------------------------------------------------------
# 4. 可视化：把图导出成 Mermaid（贴到 https://mermaid.live 查看）
# ---------------------------------------------------------------------------
def demo_visualize():
    builder = StateGraph(ReducerState)
    builder.add_node("a", lambda s: {})
    builder.add_edge(START, "a")
    print("④ Mermaid：\n" + builder.compile().get_graph().draw_mermaid())


if __name__ == "__main__":
    demo_hello_world()
    demo_reducers()
    demo_schemas()
    demo_visualize()
