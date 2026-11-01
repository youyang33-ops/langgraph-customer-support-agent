"""State 设计：整个 Agent 的“共享笔记本”。

📖 对应文档：
- Thinking in LangGraph → Step 3: Design your state
- Graph API overview → State / Schema / Reducers / Multiple schemas
- Use the graph API → Define input and output schemas
- Backward compatibility → 新字段用 NotRequired

设计原则（文档原话的意思）：
1. State 里只放【原始数据】，不放拼好的 prompt —— prompt 在节点里按需格式化；
2. 能从别的字段推导出来的，不要存；
3. 多个节点可能【同一步并行】写的字段，必须配 reducer，否则会报 InvalidUpdateError。
"""

import operator
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

# ---------------------------------------------------------------------------
# 1. 给 LLM 用的结构化输出 schema（Pydantic）
# ---------------------------------------------------------------------------

Intent = Literal["question", "bug", "billing", "feature", "complex"]
Urgency = Literal["low", "medium", "high", "critical"]


class EmailClassification(BaseModel):
    """对客户邮件的分类结果。"""

    intent: Intent = Field(
        description="question=使用咨询, bug=故障报告, billing=账单/扣费/退款, "
        "feature=功能建议, complex=复杂技术问题"
    )
    urgency: Urgency = Field(description="紧急程度")
    topic: str = Field(description="一句话主题，不超过 15 个字")
    summary: str = Field(description="问题摘要，不超过 40 个字")


class GradeDocuments(BaseModel):
    """RAG 子图里，判断检索到的资料是否和问题相关。"""

    relevant: Literal["yes", "no"] = Field(description="资料是否能回答问题")


class ReviewDecision(BaseModel):
    """人工审核的输入格式，作为 interrupt(response_schema=...) 传入。

    📖 Interrupts → Define an interrupt response schema（langgraph>=1.2.12）
    恢复时传入的 dict 会被自动校验并转成这个对象；Studio 还会据此渲染表单。
    """

    approved: bool = Field(description="是否批准发送")
    edited_response: str | None = Field(default=None, description="修改后的回复，不改则留空")
    note: str | None = Field(default=None, description="审核备注")


# ---------------------------------------------------------------------------
# 2. 图的 State（TypedDict）
# ---------------------------------------------------------------------------


class InputState(TypedDict):
    """输入 schema：调用方只需要（也只能）传这三个字段。"""

    email_id: str
    sender_email: str
    email_content: str


class OutputState(TypedDict):
    """输出 schema：invoke 的返回值只包含这些字段，内部细节不暴露给调用方。"""

    email_id: str
    status: NotRequired[str]
    classification: NotRequired[dict]
    draft_response: NotRequired[str]
    ticket_id: NotRequired[str]


class SupportState(InputState):
    """图内部完整的 State = 输入字段 + 处理过程中产生的字段。"""

    # 对话/处理日志。add_messages 是内置 reducer：按消息 id 追加或替换，
    # 账单子图（ReAct Agent）的 AI/Tool 消息也会通过这个共享字段回到父图。
    messages: Annotated[list[AnyMessage], add_messages]

    # 分类结果：直接存 LLM 返回的原始 dict（Store raw data）
    classification: NotRequired[dict]

    # 检索与外部系统的原始结果（重新获取代价高，所以存下来）
    search_results: NotRequired[list[str]]
    customer_profile: NotRequired[dict]
    ticket_id: NotRequired[str]

    # 生成内容
    draft_response: NotRequired[str]
    review: NotRequired[dict]
    status: NotRequired[str]

    # 审计日志：classify 之后会有 2 个节点【并行】执行，它们都会往这里写。
    # operator.add 作为 reducer 把多次写入拼接起来；如果去掉 reducer，并行写入会直接报错。
    audit_log: Annotated[list[str], operator.add]
