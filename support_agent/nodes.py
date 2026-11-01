"""主图的所有节点。

📖 对应文档：
- Thinking in LangGraph → Step 2（节点分四类）、Step 4（实现节点）
- Graph API overview → Nodes / Command（同时更新 State + 决定下一步）
- Interrupts → Pause using interrupt / response_schema
- Fault tolerance → error_handler / NodeError / execution_info
- Stores → 在节点里通过 runtime.store 读写长期记忆
- Streaming → Custom data（get_stream_writer）

节点 = 普通函数：接收 State，返回“要更新的字段”(dict) 或 Command。
四类节点：
    LLM 步骤：classify_intent、draft_response
    数据步骤：search_docs、lookup_customer
    动作步骤：bug_tracking、send_reply
    人工步骤：human_review
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.errors import NodeError
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from support_agent import prompts, services
from support_agent.config import Context, get_model, get_structured_model
from support_agent.state import EmailClassification, ReviewDecision, SupportState
from support_agent.subgraphs.rag import rag_graph


def _progress(node: str, message: str) -> None:
    """往 custom 流里写一条进度事件，前端/CLI 可以实时显示。"""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return  # 在图外直接调用节点函数（比如单元测试）时没有流，忽略即可
    writer({"node": node, "message": message})


def _memory_namespace(state: SupportState) -> tuple[str, ...]:
    """长期记忆的命名空间：按客户隔离（类似文件夹路径）。

    注意：namespace 的每一段不能包含 "."，所以邮箱里的点要替换掉。
    """
    return ("customers", state["sender_email"].replace(".", "_"), "tickets")


# ---------------------------------------------------------------------------
# 1. 读取邮件
# ---------------------------------------------------------------------------


def read_email(state: SupportState) -> dict:
    _progress("read_email", f"收到 {state['sender_email']} 的邮件 {state['email_id']}")
    return {
        "messages": [HumanMessage(state["email_content"], name="customer")],
        "audit_log": [f"read_email: 收到 {state['sender_email']} 的邮件"],
    }


# ---------------------------------------------------------------------------
# 2. 分类 + 路由（LLM 步骤）
# ---------------------------------------------------------------------------


def classify_intent(
    state: SupportState,
) -> Command[Literal["search_docs", "bug_tracking", "billing_agent", "lookup_customer"]]:
    """用结构化输出分类，然后用 Command 同时完成“写 State”和“选下一步”。

    返回值类型注解 Command[Literal[...]] 告诉 LangGraph 这个节点可能去哪些节点，
    这样画图时能画出这些边（运行时并不依赖它）。
    """
    result: EmailClassification = get_structured_model(EmailClassification).invoke(
        [SystemMessage(prompts.CLASSIFY_SYSTEM), HumanMessage(state["email_content"])]
    )
    classification = result.model_dump()  # 存原始 dict，不存格式化文本

    specialist = {"billing": "billing_agent", "bug": "bug_tracking"}.get(
        classification["intent"], "search_docs"
    )
    _progress("classify_intent", f"意图={classification['intent']} 紧急度={classification['urgency']} → {specialist}")
    return Command(
        update={
            "classification": classification,
            "audit_log": [f"classify_intent: {classification['intent']}/{classification['urgency']}"],
        },
        # goto 传列表 = 下一步【并行】执行这两个节点（同一个 super-step）
        goto=[specialist, "lookup_customer"],
    )


# ---------------------------------------------------------------------------
# 3. 数据步骤 / 动作步骤（与 lookup_customer 并行）
# ---------------------------------------------------------------------------


def search_docs(state: SupportState) -> dict:
    """在节点内部调用子图：父图 State 和子图 State 结构不同时用这种方式，自己做转换。"""
    _progress("search_docs", "进入知识库检索子图")
    result = rag_graph.invoke({"question": state["email_content"]})
    docs = result.get("documents", []) if result.get("relevant") else []
    return {"search_results": docs, "audit_log": [f"search_docs: 找到 {len(docs)} 条相关资料"]}


def bug_tracking(state: SupportState) -> dict:
    known_issues = services.search_knowledge_base(state["email_content"], k=1)
    ticket_id = services.create_bug_ticket(state["classification"]["summary"])
    _progress("bug_tracking", f"已创建故障工单 {ticket_id}")
    return {
        "ticket_id": ticket_id,
        "search_results": known_issues + [f"已为您创建故障工单 {ticket_id}，工程师会在 24 小时内跟进"],
        "audit_log": [f"bug_tracking: 创建工单 {ticket_id}"],
    }


def lookup_customer(state: SupportState) -> dict:
    """查询 CRM。在 graph.py 里给它配了 CachePolicy：同一个邮箱 5 分钟内不重复查询。"""
    profile = services.fetch_customer_profile(state["sender_email"])
    _progress("lookup_customer", f"客户：{profile['name']}（{profile['tier']}）")
    return {"customer_profile": profile, "audit_log": [f"lookup_customer: {profile['name']}"]}


# ---------------------------------------------------------------------------
# 4. 起草回复（LLM 步骤）+ 读取长期记忆
# ---------------------------------------------------------------------------


async def draft_response(
    state: SupportState, runtime: Runtime[Context]
) -> Command[Literal["human_review", "send_reply"]]:
    classification = state["classification"]
    profile = state.get("customer_profile") or {"name": "客户", "tier": "未知"}

    # 在这里才把原始数据格式化成 prompt（Format prompts on-demand）
    sections = []
    if state.get("search_results"):
        sections.append("参考资料：\n" + "\n".join(f"- {doc}" for doc in state["search_results"]))
    if classification["intent"] == "billing":
        # 账单子图的结论在共享的 messages 里：取最后一条不带工具调用的 AI 消息
        final = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and not m.tool_calls), None
        )
        if final:
            sections.append(f"账单处理结果：\n- {final.text}")

    # 长期记忆：跨线程（跨工单）共享，按客户邮箱查历史工单
    past = await runtime.store.asearch(_memory_namespace(state), limit=3) if runtime.store else []
    if past:
        sections.append(
            "该客户的历史工单（仅供参考）：\n"
            + "\n".join(f"* {item.value['summary']}（{item.value['status']}）" for item in past)
        )

    response = await get_model().ainvoke(
        [
            SystemMessage(prompts.DRAFT_SYSTEM.format(team_name=runtime.context.team_name)),
            HumanMessage(
                prompts.DRAFT_HUMAN.format(
                    email_content=state["email_content"],
                    customer_name=profile["name"],
                    customer_tier=profile["tier"],
                    intent=classification["intent"],
                    urgency=classification["urgency"],
                    context="\n\n".join(sections) or "（没有检索到资料）",
                )
            ),
        ]
    )

    needs_review = classification["urgency"] in ("high", "critical") or classification["intent"] in (
        "complex",
        "billing",
    )
    return Command(
        update={
            "draft_response": str(response.text),
            "audit_log": [f"draft_response: 已起草，{'需要' if needs_review else '无需'}人工审核"],
        },
        goto="human_review" if needs_review else "send_reply",
    )


# ---------------------------------------------------------------------------
# 5. 人工审核（Human-in-the-loop）
# ---------------------------------------------------------------------------


def human_review(state: SupportState) -> Command[Literal["send_reply", "save_memory"]]:
    # ⚠️ interrupt 的规则（文档 Rules of interrupts）：
    # 1. 恢复时整个节点会从头重新执行，所以 interrupt 之前不要放有副作用的代码；
    # 2. 不要用 try/except Exception 包住 interrupt；
    # 3. payload 必须能 JSON 序列化。
    decision: ReviewDecision = interrupt(
        {
            "type": "reply_review",
            "question": "请审核这封回复：批准发送 / 修改后发送 / 拒绝（转人工处理）",
            "email_id": state["email_id"],
            "classification": state["classification"],
            "draft_response": state["draft_response"],
        },
        response_schema=ReviewDecision,  # 恢复值会被校验并转换成 ReviewDecision 对象
    )

    if decision.approved:
        return Command(
            update={
                "draft_response": decision.edited_response or state["draft_response"],
                "review": decision.model_dump(),
                "audit_log": [f"human_review: 批准{'（已修改）' if decision.edited_response else ''}"],
            },
            goto="send_reply",
        )
    return Command(
        update={
            "status": "escalated",
            "review": decision.model_dump(),
            "audit_log": [f"human_review: 拒绝，转人工处理（{decision.note or '无备注'}）"],
        },
        goto="save_memory",
    )


# ---------------------------------------------------------------------------
# 6. 发送回复（动作步骤）+ 失败兜底
# ---------------------------------------------------------------------------


async def send_reply(state: SupportState, runtime: Runtime[Context]) -> dict:
    """在 graph.py 里配置了：timeout（只支持 async 节点）+ RetryPolicy + error_handler。"""
    attempt = runtime.execution_info.node_attempt  # 第几次尝试（从 1 开始）
    _progress("send_reply", f"第 {attempt} 次尝试发送邮件")
    await services.send_email(
        state["sender_email"],
        state["draft_response"],
        mode=runtime.context.email_service_mode,
        attempt=attempt,
    )
    return {"status": "sent", "audit_log": [f"send_reply: 第 {attempt} 次尝试发送成功"]}


def handle_send_failure(state: SupportState, error: NodeError) -> Command[Literal["save_memory"]]:
    """error_handler：重试全部失败后才会调用。参数 error: NodeError 按类型注解自动注入。

    这是 Saga / 补偿模式：不让整张图崩掉，而是记录失败并转到后续流程。
    """
    _progress("send_reply", f"重试耗尽，转人工：{error.error}")
    return Command(
        update={"status": "send_failed", "audit_log": [f"{error.node}: 重试耗尽已兜底（{error.error}）"]},
        goto="save_memory",
    )


# ---------------------------------------------------------------------------
# 7. 写入长期记忆
# ---------------------------------------------------------------------------


async def save_memory(state: SupportState, runtime: Runtime[Context]) -> dict:
    if runtime.store:
        await runtime.store.aput(
            _memory_namespace(state),
            state["email_id"],  # key：同一张工单重复写会覆盖
            {
                "summary": state["classification"]["summary"],
                "intent": state["classification"]["intent"],
                "status": state.get("status", "unknown"),
            },
        )
    return {"audit_log": ["save_memory: 已写入客户历史"]}
