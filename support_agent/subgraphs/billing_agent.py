"""账单处理子图：一个标准的 ReAct 工具调用 Agent。

📖 对应文档：
- Quickstart（Graph API 版计算器 Agent）：llm_call ⇄ tool_node 循环
- Workflows and agents → Agents / ToolNode
- Graph API overview → RemainingSteps（主动处理递归上限）
- Subgraphs → Add a subgraph as a node（父子图共享 `messages` 字段，直接 add_node）

流程：
    START → llm_call ─(有 tool_calls)→ tools ─┐
               ↑                                │
               └────────────────────────────────┘
            llm_call ─(没有 tool_calls)→ END
"""

from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from support_agent import prompts
from support_agent.config import get_model
from support_agent.tools import BILLING_TOOLS


class BillingState(TypedDict):
    # 和父图 SupportState 同名同 reducer 的字段 → 父子图自动共享
    messages: Annotated[list[AnyMessage], add_messages]
    # 托管值：LangGraph 自动填入“距离 recursion_limit 还剩几步”，不需要调用方传
    remaining_steps: RemainingSteps


def llm_call(state: BillingState) -> dict:
    if state["remaining_steps"] <= 2:
        # 主动兜底：步数快用完时直接收尾，而不是等 GraphRecursionError 抛出来
        return {"messages": [AIMessage("处理步骤过多，已转交人工账单专员。")]}
    model = get_model().bind_tools(BILLING_TOOLS)
    response = model.invoke([SystemMessage(prompts.BILLING_SYSTEM)] + state["messages"])
    return {"messages": [response]}


def build_billing_graph():
    builder = StateGraph(BillingState)
    builder.add_node("llm_call", llm_call)
    # ToolNode：预置节点，执行上一条 AIMessage 里的所有 tool_calls，返回 ToolMessage
    builder.add_node("tools", ToolNode(BILLING_TOOLS))

    builder.add_edge(START, "llm_call")
    # tools_condition：预置路由函数，有 tool_calls 去 "tools"，否则去 END
    builder.add_conditional_edges("llm_call", tools_condition)
    builder.add_edge("tools", "llm_call")
    # 不传 checkpointer → 默认“按调用持久化”(per-invocation)：
    # 继承父图的 checkpointer，所以工具里的 interrupt() 能暂停/恢复；
    # 但每次调用都从空白开始，不会记住上一张工单的对话。
    return builder.compile(name="billing_agent")


billing_graph = build_billing_graph()
