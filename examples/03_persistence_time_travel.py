"""03 · 持久化（短期记忆）与时间旅行。

📖 对应文档：Persistence、Checkpointers、Memory（短期记忆部分）、Use time-travel、Test（partial execution）
▶ 运行：python examples/03_persistence_time_travel.py

核心概念（面试高频）：
- checkpointer：每个 super-step 结束时把 State 存成一个 checkpoint；
- thread_id：一个线程 = 一串 checkpoint（一次对话 / 一张工单）；
- 有了 checkpoint 才有：多轮记忆、interrupt 人工介入、故障恢复、时间旅行。
"""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from typing_extensions import NotRequired, TypedDict


# ---------------------------------------------------------------------------
# 1. 短期记忆：同一个 thread_id 的多次调用共享消息历史
# ---------------------------------------------------------------------------
def chatbot(state: MessagesState):
    count = sum(1 for m in state["messages"] if m.type == "human")
    return {"messages": [AIMessage(f"这是你第 {count} 句话，你上一句说的是「{state['messages'][-1].content}」")]}


def demo_short_term_memory():
    app = StateGraph(MessagesState).add_node(chatbot).add_edge(START, "chatbot").compile(
        checkpointer=InMemorySaver()  # 生产环境换成 PostgresSaver / SqliteSaver
    )
    alice = {"configurable": {"thread_id": "alice"}}
    bob = {"configurable": {"thread_id": "bob"}}
    app.invoke({"messages": [("user", "你好")]}, alice)
    print("① alice 第二轮：", app.invoke({"messages": [("user", "我叫小明")]}, alice)["messages"][-1].content)
    print("① bob 第一轮：  ", app.invoke({"messages": [("user", "嗨")]}, bob)["messages"][-1].content)


# ---------------------------------------------------------------------------
# 2. 查看状态与历史
# ---------------------------------------------------------------------------
class JokeState(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]


def build_joke_graph():
    return (
        StateGraph(JokeState)
        .add_node("generate_topic", lambda s: {"topic": "洗衣机里的袜子"})
        .add_node("write_joke", lambda s: {"joke": f"为什么{s['topic']}总会消失？因为它们私奔了！"})
        .add_edge(START, "generate_topic")
        .add_edge("generate_topic", "write_joke")
        .compile(checkpointer=InMemorySaver())
    )


def demo_state_and_history(graph, config):
    graph.invoke({}, config)
    snapshot = graph.get_state(config)
    print("② 最新状态：", snapshot.values, "| next =", snapshot.next)

    print("② 历史（最新在前）：")
    for s in graph.get_state_history(config):
        print(f"   step={s.metadata['step']:>2}  next={s.next}  values={s.values}")


# ---------------------------------------------------------------------------
# 3. 时间旅行：重放（replay）与分叉（fork）
# ---------------------------------------------------------------------------
def demo_time_travel(graph, config):
    history = list(graph.get_state_history(config))
    before_joke = next(s for s in history if s.next == ("write_joke",))

    # Replay：用旧 checkpoint 的 config 调用，之后的节点会【重新执行】（不是读缓存）
    replay = graph.invoke(None, before_joke.config)
    print("③ replay：", replay["joke"])

    # Fork：在旧 checkpoint 上修改 State，生成一个新分支（原历史不受影响）
    fork_config = graph.update_state(before_joke.config, {"topic": "程序员的头发"})
    print("③ fork：  ", graph.invoke(None, fork_config)["joke"])

    # as_node：假装这次更新是某个节点产出的 → 从该节点的下一个节点继续执行
    # （文档 Test → Partial execution 就是用这个技巧只测试图的一部分）
    new_thread = {"configurable": {"thread_id": "partial"}}
    graph.update_state(new_thread, {"topic": "产品经理的需求"}, as_node="generate_topic")
    print("③ as_node：", graph.invoke(None, new_thread)["joke"])


# ---------------------------------------------------------------------------
# 4. 持久化模式 durability
# ---------------------------------------------------------------------------
def demo_durability(graph):
    # "exit"：只在结束/中断/报错时存（最快，崩溃会丢中间状态）
    # "async"（默认）：下一步执行的同时异步写 checkpoint
    # "sync"：每一步写完 checkpoint 才继续（最安全）
    config = {"configurable": {"thread_id": "durability"}}
    graph.invoke({}, config, durability="exit")
    print("④ durability='exit' 的 checkpoint 数：", len(list(graph.get_state_history(config))))


if __name__ == "__main__":
    demo_short_term_memory()
    joke_graph = build_joke_graph()
    cfg = {"configurable": {"thread_id": "joke"}}
    demo_state_and_history(joke_graph, cfg)
    demo_time_travel(joke_graph, cfg)
    demo_durability(joke_graph)
