"""离线假模型：不调用任何 API，用关键词规则模拟 LLM 的行为。

【学习时不需要读这个文件】。它的作用是：没有 API key 也能完整跑通整张图，
并且行为是确定的，方便写测试（真实 LLM 每次输出不同，不适合做断言）。

它是一个真正的 LangChain BaseChatModel 子类，所以支持：
- bind_tools()               → 返回 tool_calls（账单子图的 ReAct 循环）
- with_structured_output()   → 返回结构化结果（分类、文档评分）
- 流式输出 _stream()          → 逐段吐字，stream_events 能看到 token 流
"""

import json
import re
import uuid
from typing import Any, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

SYNONYMS = {
    ("暗黑", "深色", "黑色主题", "dark"): "夜间模式 深色模式 路线图",
    ("登不上", "进不去", "忘记密码", "忘了密码"): "重置密码",
    ("导出", "pdf"): "导出 PDF 崩溃",
    ("504", "超时"): "API 504 排查",
    ("扣", "退款", "退钱"): "退款政策 重复扣费",
}


def _tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}],
    )


def _classify(email: str) -> dict:
    text = email.lower()
    if re.search(r"扣费|扣了|扣款|退款|发票|付款|账单|charged|refund", text):
        intent = "billing"
    elif re.search(r"504|api|集成|间歇", text):
        intent = "complex"
    elif re.search(r"崩溃|闪退|报错|bug|crash", text):
        intent = "bug"
    elif re.search(r"能不能|希望|建议|增加|加一个|dark mode", text):
        intent = "feature"
    else:
        intent = "question"

    if re.search(r"所有用户|生产事故|数据丢失", text):
        urgency = "critical"
    elif re.search(r"紧急|马上|立刻|两次|重复|urgent", text) or intent == "complex":
        urgency = "high"
    elif intent == "bug":
        urgency = "medium"
    else:
        urgency = "low"

    first_line = email.strip().splitlines()[0]
    return {"intent": intent, "urgency": urgency, "topic": first_line[:15], "summary": first_line[:40]}


def _rewrite(question: str) -> str:
    lowered = question.lower()
    for keywords, rewritten in SYNONYMS.items():
        if any(k in lowered for k in keywords):
            return rewritten
    return question[:20]


def _billing(messages: list[BaseMessage]) -> AIMessage:
    email = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
    match = re.search(r"ORD-\d+", email, re.IGNORECASE)
    if not match:
        return AIMessage("邮件里没有订单号，需要客户补充订单号后才能核实扣费情况。")
    order_id = match.group(0).upper()
    tool_results = {m.name: m.content for m in messages if isinstance(m, ToolMessage)}

    if "get_order" not in tool_results:
        return _tool_call("get_order", {"order_id": order_id})
    try:
        order = json.loads(tool_results["get_order"])
    except json.JSONDecodeError:
        return AIMessage(f"没有查到订单 {order_id}，需要客户确认订单号。")

    double_charged = len(order.get("charges", [])) >= 2
    if double_charged and "issue_refund" not in tool_results:
        return _tool_call(
            "issue_refund", {"order_id": order_id, "amount": order["amount"], "reason": "重复扣费"}
        )
    if "issue_refund" in tool_results:
        return AIMessage(f"已核实订单 {order_id} 存在重复扣费。{tool_results['issue_refund']}。")
    return AIMessage(f"已核实订单 {order_id}，只有一笔扣款，没有重复扣费。")


def _draft(system: str, human: str) -> str:
    team = re.search(r"「(.+?)」", system)
    name = re.search(r"客户称呼：(\S+?)（", human)
    facts = [line[2:] for line in human.splitlines() if line.startswith("- ")]
    body = "\n".join(f"· {fact}" for fact in facts) or "· 我们已收到您的反馈，专员会尽快跟进。"
    team_name = team.group(1) if team else "客服团队"
    return (
        f"{name.group(1) if name else '您'}您好：\n\n感谢您联系{team_name}，针对您反馈的问题答复如下：\n"
        f"{body}\n\n如有其他问题，欢迎随时回复本邮件。\n{team_name}"
    )


class MockChatModel(BaseChatModel):
    """规则驱动的假聊天模型。"""

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def with_structured_output(self, schema, *, include_raw=False, method=None, **kwargs):
        # 真实模型需要 method="function_calling"，假模型忽略这个参数
        return super().with_structured_output(schema, include_raw=include_raw)

    def _respond(self, messages: list[BaseMessage], tools: list[dict]) -> AIMessage:
        names = {t["function"]["name"] for t in tools}
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        last = messages[-1].content if messages else ""

        if "EmailClassification" in names:
            return _tool_call("EmailClassification", _classify(last))
        if "GradeDocuments" in names:
            relevant = "no" if "检索到的资料：\n（无）" in last else "yes"
            return _tool_call("GradeDocuments", {"relevant": relevant})
        if "get_order" in names:
            return _billing(messages)
        if "【改写检索问题】" in system:
            return AIMessage(_rewrite(last))
        if "【撰写客服回复】" in system:
            return AIMessage(_draft(system, last))
        return AIMessage("（MockChatModel 默认回复）")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        message = self._respond(messages, kwargs.get("tools") or [])
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        message = self._respond(messages, kwargs.get("tools") or [])
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": tc["name"], "args": json.dumps(tc["args"], ensure_ascii=False),
                         "id": tc["id"], "index": i}
                        for i, tc in enumerate(message.tool_calls)
                    ],
                )
            )
            return
        text = message.content
        for i in range(0, len(text), 8):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text[i : i + 8]))
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
