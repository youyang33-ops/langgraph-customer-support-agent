"""知识库检索子图（Agentic RAG）。

📖 对应文档：
- Build a custom RAG agent with LangGraph：检索 → 评估相关性 → 不相关就改写问题再检索
- Subgraphs → Call a subgraph inside a node（父子图 State 不同，由父节点负责转换）
- Fault tolerance → Retries / Inspect retry state（execution_info.node_attempt）

流程：
    START → retrieve → grade_documents ─(相关)──────────→ END
                 ↑              └─(不相关且还能改写)→ rewrite_query
                 └──────────────────────────────────────────┘
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from typing_extensions import NotRequired, TypedDict

from support_agent import prompts, services
from support_agent.config import get_model, get_structured_model
from support_agent.state import GradeDocuments

MAX_REWRITES = 1


class RAGState(TypedDict):
    """子图自己的 State，和父图完全不同 —— 子图是独立、可单独测试的模块。"""

    question: str
    query: NotRequired[str]
    documents: NotRequired[list[str]]
    relevant: NotRequired[bool]
    rewrite_count: NotRequired[int]


def retrieve(state: RAGState, runtime: Runtime) -> dict:
    query = state.get("query") or state["question"]
    # 重试时可以换一种降级策略：第 2 次起只取 1 条结果
    k = 3 if runtime.execution_info.node_attempt == 1 else 1
    docs = services.search_knowledge_base(query, k=k)
    get_stream_writer()({"node": "rag.retrieve", "message": f"检索「{query[:20]}」命中 {len(docs)} 条"})
    return {"query": query, "documents": docs}


def grade_documents(state: RAGState) -> dict:
    documents = "\n".join(f"- {d}" for d in state.get("documents", [])) or "（无）"
    grade = get_structured_model(GradeDocuments).invoke(
        [
            SystemMessage(prompts.GRADE_SYSTEM),
            HumanMessage(prompts.GRADE_HUMAN.format(question=state["question"], documents=documents)),
        ]
    )
    return {"relevant": grade.relevant == "yes"}


def route_after_grade(state: RAGState) -> Literal["rewrite_query", "__end__"]:
    """条件边的路由函数：只负责“去哪”，不修改 State。"""
    if state["relevant"] or state.get("rewrite_count", 0) >= MAX_REWRITES:
        return END
    return "rewrite_query"


def rewrite_query(state: RAGState) -> dict:
    response = get_model().invoke(
        [SystemMessage(prompts.REWRITE_SYSTEM), HumanMessage(state["question"])]
    )
    get_stream_writer()({"node": "rag.rewrite_query", "message": f"改写检索语句为「{response.text}」"})
    return {"query": str(response.text), "rewrite_count": state.get("rewrite_count", 0) + 1}


def build_rag_graph():
    builder = StateGraph(RAGState)
    # 检索是“数据步骤”，调用外部服务，配置重试（文档：Data steps → Retry strategy）
    builder.add_node("retrieve", retrieve, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.2))
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges("grade_documents", route_after_grade, ["rewrite_query", END])
    builder.add_edge("rewrite_query", "retrieve")
    # name 会出现在 stream.subgraphs 的 graph_name 和 LangSmith 链路里
    return builder.compile(name="rag")


rag_graph = build_rag_graph()
