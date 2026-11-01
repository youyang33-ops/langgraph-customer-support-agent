"""06 · 子图：把复杂流程拆成可复用、可单独测试的模块。

📖 对应文档：Subgraphs、Time travel → Subgraphs、Interrupts → Using with subgraphs
▶ 运行：python examples/06_subgraphs.py

两种接入方式（主项目里各用了一次）：
    A. 子图直接作为节点：父子图有【同名共享字段】→ support_agent/subgraphs/billing_agent.py
    B. 在节点函数里调用子图：State 完全不同，节点负责转换输入输出 → support_agent/subgraphs/rag.py

子图持久化（compile(checkpointer=...)）：
    None（默认）按调用持久化：支持 interrupt，但每次调用从头开始
    True        按线程持久化：同一 thread 多次调用会累积子图状态（多轮记忆）
    False       无状态：不支持 interrupt，最轻量
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# A. 共享字段：子图直接 add_node
# ---------------------------------------------------------------------------
class SubState(TypedDict):
    shared: str  # 和父图同名 → 自动传入、自动写回
    private: str  # 只有子图知道


class ParentState(TypedDict):
    shared: str


def build_subgraph():
    def step1(state: SubState):
        return {"private": "子图内部数据"}

    def step2(state: SubState):
        answer = interrupt("子图里也能暂停等人")  # 父图有 checkpointer，子图自动继承
        return {"shared": f"{state['shared']} → 子图处理（{state['private']}，人说：{answer}）"}

    return (
        StateGraph(SubState)
        .add_node("step1", step1)
        .add_node("step2", step2)
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .compile(name="my_subgraph")
    )


def demo_subgraph_as_node():
    parent = (
        StateGraph(ParentState)
        .add_node("subgraph", build_subgraph())
        .add_edge(START, "subgraph")
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "a"}}
    parent.invoke({"shared": "父图输入"}, config)

    # subgraphs=True：同时看到暂停中的子图内部状态
    state = parent.get_state(config, subgraphs=True)
    print("A. 子图内部状态：", state.tasks[0].state.values)
    print("A. 恢复后：", parent.invoke(Command(resume="同意"), config))


# ---------------------------------------------------------------------------
# B. 不同 State：在节点函数里调用子图
# ---------------------------------------------------------------------------
class TranslateState(TypedDict):
    text: str
    translated: str


translate_graph = (
    StateGraph(TranslateState)
    .add_node("translate", lambda s: {"translated": f"[EN] {s['text']}"})
    .add_edge(START, "translate")
    .compile()
)


class ArticleState(TypedDict):
    article: str
    article_en: str


def call_translator(state: ArticleState):
    result = translate_graph.invoke({"text": state["article"]})  # 父 → 子：转换输入
    return {"article_en": result["translated"]}  # 子 → 父：转换输出


def demo_subgraph_in_node():
    parent = StateGraph(ArticleState).add_node(call_translator).add_edge(START, "call_translator").compile()
    print("B. ", parent.invoke({"article": "你好，LangGraph"}))


# ---------------------------------------------------------------------------
# C. 流式观察子图（stream_events v3 的 subgraphs 投影）
# ---------------------------------------------------------------------------
def demo_stream_subgraphs():
    parent = StateGraph(ArticleState).add_node(call_translator).add_edge(START, "call_translator").compile()
    stream = parent.stream_events({"article": "流式"}, version="v3")
    for sub in stream.subgraphs:
        print("C. 发现子图：", sub.graph_name, sub.path)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message=".*v3 streaming protocol.*")
    demo_subgraph_as_node()
    demo_subgraph_in_node()
    demo_stream_subgraphs()
