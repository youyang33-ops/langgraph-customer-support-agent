"""07 · 流式输出：两套 API。

📖 对应文档：Streaming（stream_mode，底层）、Event streaming（stream_events v3，推荐）
▶ 运行：python examples/07_streaming.py

1. graph.stream(..., stream_mode=[...], version="v2")
   每个 chunk 都是 {"type": 模式名, "ns": 子图命名空间, "data": 数据}
   values=每步后的完整 State | updates=每个节点的增量 | messages=LLM token
   custom=节点里 get_stream_writer() 写的任意数据 | checkpoints / tasks / debug

2. graph.stream_events(..., version="v3")（目前标记为实验特性）
   返回一个 run 对象，提供类型化的“投影”：
   stream.messages / stream.values / stream.subgraphs / stream.output / stream.interrupts / stream.extensions
   多个投影可以同时消费，互不影响。
"""

import warnings

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer

warnings.filterwarnings("ignore", message=".*v3 streaming protocol.*")


def build_graph():
    # GenericFakeChatModel：LangChain 自带的假模型，按空格切分逐个吐 token
    model = GenericFakeChatModel(messages=iter([AIMessage("流式 输出 让 用户 不用 干等")] * 10))

    def prepare(state: MessagesState):
        writer = get_stream_writer()
        for pct in (30, 60, 100):
            writer({"progress": pct})  # → custom 流
        return {}

    def call_model(state: MessagesState):
        return {"messages": [model.invoke(state["messages"])]}  # LLM token → messages 流

    return (
        StateGraph(MessagesState)
        .add_node(prepare)
        .add_node(call_model)
        .add_edge(START, "prepare")
        .add_edge("prepare", "call_model")
        .compile()
    )


INPUT = {"messages": [("user", "你好")]}


def demo_stream_modes(graph):
    print("① stream_mode（v2 格式）：")
    for part in graph.stream(INPUT, stream_mode=["updates", "custom", "messages"], version="v2"):
        if part["type"] == "updates":
            print("   [updates] 节点完成：", list(part["data"]))
        elif part["type"] == "custom":
            print("   [custom]  ", part["data"])
        elif part["type"] == "messages":
            token, metadata = part["data"]
            print(f"   [messages] {metadata['langgraph_node']}: {token.content!r}")


def demo_event_streaming(graph):
    print("② stream_events v3 —— messages 投影逐 token 打印：", end=" ")
    stream = graph.stream_events(INPUT, version="v3")
    for message in stream.messages:
        for token in message.text:
            print(token, end="|", flush=True)
    print("\n   最终 State 的最后一条消息：", stream.output["messages"][-1].text)

    print("② interleave 按到达顺序同时消费多个投影：")
    stream = graph.stream_events(INPUT, version="v3")
    for name, item in stream.interleave("values", "messages"):
        print(f"   [{name}]", f"{len(item['messages'])} 条消息" if name == "values" else f"来自 {item.node}")


# ---------------------------------------------------------------------------
# 自定义投影：把 custom 流里的进度数据做成 stream.extensions["progress"]
# ---------------------------------------------------------------------------
class ProgressTransformer(StreamTransformer):
    required_stream_modes = ("custom",)  # 声明需要的底层模式，否则 custom 事件不会产生

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.channel = StreamChannel()  # 不带名字：只作为旁路投影

    def init(self) -> dict:
        return {"progress": self.channel}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "custom" and "progress" in event["params"]["data"]:
            self.channel.push(event["params"]["data"]["progress"])
        return True


def demo_custom_transformer(graph):
    stream = graph.stream_events(INPUT, version="v3", transformers=[ProgressTransformer])
    print("③ 自定义投影 progress：", list(stream.extensions["progress"]))


if __name__ == "__main__":
    g = build_graph()
    demo_stream_modes(g)
    demo_event_streaming(g)
    demo_custom_transformer(g)
