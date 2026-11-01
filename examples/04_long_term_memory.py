"""04 · 长期记忆（Store）与运行时上下文（Runtime Context）。

📖 对应文档：Stores、Memory（长期记忆部分）、Persistence → Checkpointer vs. store、Graph API → Runtime context
▶ 运行：python examples/04_long_term_memory.py

checkpointer vs store（面试必问）：
    checkpointer：按 thread 保存【整个图的 State】→ 短期记忆，一次对话内有效
    store       ：按 namespace 保存【任意 JSON 文档】→ 长期记忆，跨对话、跨线程共享
"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


# ---------------------------------------------------------------------------
# 1. Store 的基本 API：put / get / search / list_namespaces / delete
# ---------------------------------------------------------------------------
def demo_store_api():
    store = InMemoryStore()  # 生产环境换成 PostgresStore / SqliteStore / RedisStore
    ns = ("alice", "memories")  # namespace 像文件夹路径（每段不能包含 "."）

    store.put(ns, "food", {"text": "喜欢吃火锅", "category": "饮食"})
    store.put(ns, "lang", {"text": "主要用 Python", "category": "技术"})
    store.put(("bob", "memories"), "food", {"text": "不吃辣", "category": "饮食"})

    print("① get：", store.get(ns, "food").value)
    print("① search + 过滤：", [i.value for i in store.search(ns, filter={"category": "技术"})])
    print("① list_namespaces：", store.list_namespaces(prefix=("alice",)))
    # 语义搜索：InMemoryStore(index={"embed": embeddings, "dims": 1536}) 后，
    # store.search(ns, query="她喜欢吃什么") 就会按向量相似度排序（需要 embedding 模型）


# ---------------------------------------------------------------------------
# 2. 在图里使用 Store + Context
# ---------------------------------------------------------------------------
@dataclass
class Context:
    user_id: str  # 调用方传入，节点只读


def remember_and_reply(state: MessagesState, runtime: Runtime[Context]):
    ns = (runtime.context.user_id, "memories")
    text = state["messages"][-1].content
    if text.startswith("记住"):
        runtime.store.put(ns, str(len(runtime.store.search(ns))), {"text": text[2:]})
        return {"messages": [AIMessage("好的，记下了。")]}
    memories = [item.value["text"] for item in runtime.store.search(ns)]
    return {"messages": [AIMessage(f"我记得关于你的事：{memories}")]}


def demo_store_in_graph():
    store = InMemoryStore()
    app = (
        StateGraph(MessagesState, context_schema=Context)
        .add_node(remember_and_reply)
        .add_edge(START, "remember_and_reply")
        .compile(checkpointer=InMemorySaver(), store=store)  # 两者一起用
    )
    alice = Context(user_id="alice")
    app.invoke({"messages": [("user", "记住我在找 Agent 实习")]}, {"configurable": {"thread_id": "1"}}, context=alice)
    # 换一个全新的 thread（新对话），短期记忆是空的，但长期记忆还在
    result = app.invoke({"messages": [("user", "你还记得我吗")]}, {"configurable": {"thread_id": "2"}}, context=alice)
    print("② 新线程里：", result["messages"][-1].content)
    other = app.invoke(
        {"messages": [("user", "你还记得我吗")]}, {"configurable": {"thread_id": "3"}}, context=Context("bob")
    )
    print("② bob：    ", other["messages"][-1].content)


if __name__ == "__main__":
    demo_store_api()
    demo_store_in_graph()
