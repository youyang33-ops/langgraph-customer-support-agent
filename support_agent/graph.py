"""把节点连成图 —— 整个项目最核心的文件，建议第一个读。

📖 对应文档：
- Thinking in LangGraph → Step 5: Wire it together
- Graph API overview → StateGraph / Compiling / Edges / Node caching
- Use the graph API → input/output schema、Map-Reduce（这里用 Command goto 列表做并行）
- Fault tolerance → RetryPolicy / TimeoutPolicy / error_handler / set_node_defaults
- Subgraphs → 子图作为节点
- Persistence → compile(checkpointer=..., store=...)

完整流程：

    START → read_email → classify_intent ─┬→ search_docs ──┐   (question/feature/complex)
                                          ├→ bug_tracking ─┤   (bug)
                                          ├→ billing_agent ┤   (billing，ReAct 子图，退款需审批)
                                          └→ lookup_customer┘  (总是并行执行，带缓存)
                                                     ↓
                                              draft_response ─(紧急/复杂/账单)→ human_review ─(拒绝)─┐
                                                     │                              │(批准)          │
                                                     └───────(普通)──────→ send_reply ←┘             │
                                                                              │ (失败→error_handler)  │
                                                                              ↓                       │
                                                                         save_memory ←────────────────┘
                                                                              ↓
                                                                             END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy

from support_agent import nodes
from support_agent.config import Context
from support_agent.services import EmailServiceError
from support_agent.state import InputState, OutputState, SupportState
from support_agent.streaming import ProgressTransformer
from support_agent.subgraphs.billing_agent import billing_graph


def build_graph(checkpointer=None, store=None, cache=None):
    """构建并编译主图。

    checkpointer / store / cache 由调用方注入：
    - 命令行运行（main.py）：传入 SQLite，进程重启后还能恢复；
    - 单元测试：传入 InMemorySaver，每个测试独立；
    - `langgraph dev` / LangSmith 部署：不传，由平台自动提供持久化。
    """
    builder = StateGraph(
        SupportState,
        input_schema=InputState,  # 调用方只能传这些字段
        output_schema=OutputState,  # invoke 只返回这些字段
        context_schema=Context,  # 运行时只读上下文
    )

    # 图级默认配置：所有节点默认重试 2 次（LLM 调用偶尔超时/限流很常见）。
    # 单个节点在 add_node 里的配置优先级更高。
    builder.set_node_defaults(retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.5))

    builder.add_node("read_email", nodes.read_email)
    builder.add_node("classify_intent", nodes.classify_intent)
    builder.add_node("search_docs", nodes.search_docs)
    builder.add_node("bug_tracking", nodes.bug_tracking, retry_policy=RetryPolicy(max_attempts=3))
    # 编译好的子图本身就是一个节点（它和父图共享 messages 字段）
    builder.add_node("billing_agent", billing_graph)
    builder.add_node(
        "lookup_customer",
        nodes.lookup_customer,
        # 节点缓存：key_func 决定缓存键（默认是整个输入的哈希），ttl 单位秒
        cache_policy=CachePolicy(key_func=lambda state: state.get("sender_email", ""), ttl=300),
    )
    builder.add_node("draft_response", nodes.draft_response)
    builder.add_node("human_review", nodes.human_review)
    builder.add_node(
        "send_reply",
        nodes.send_reply,
        # 单次尝试最多 10 秒（timeout 只支持 async 节点）
        timeout=TimeoutPolicy(run_timeout=10),
        # 注意：默认 retry_on 不重试 OSError（ConnectionError 是它的子类），所以要显式指定
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.2, retry_on=EmailServiceError),
        # 重试耗尽后的兜底
        error_handler=nodes.handle_send_failure,
    )
    builder.add_node("save_memory", nodes.save_memory)

    # 只需要连“固定”的边；会分叉的地方由节点返回 Command 自己决定
    builder.add_edge(START, "read_email")
    builder.add_edge("read_email", "classify_intent")
    for upstream in ("search_docs", "bug_tracking", "billing_agent", "lookup_customer"):
        # 并行的两个节点在同一个 super-step 完成，draft_response 只会执行一次
        builder.add_edge(upstream, "draft_response")
    builder.add_edge("send_reply", "save_memory")
    builder.add_edge("save_memory", END)

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
        cache=cache,
        transformers=[ProgressTransformer],  # 每次 stream_events 都带上进度投影
        name="customer_support",
    )


# 供 langgraph.json 使用：`langgraph dev` 会加载这个变量，并自动注入持久化
graph = build_graph()
