"""集成测试：跑完整张图，覆盖路由、人工审核、容错、长期记忆和局部执行。

📖 对应文档：Test → Getting started / Partial execution，Interrupts，Fault tolerance
"""

from langgraph.types import Command

from support_agent.config import Context
from tests.conftest import make_input


def config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


async def test_simple_question_is_sent_without_review(graph):
    result = await graph.ainvoke(
        make_input("Q1", "我忘记密码了，怎么重置？"), config("Q1"), context=Context(), version="v2"
    )
    assert result.interrupts == ()
    assert result.value["status"] == "sent"
    assert "重置" in result.value["draft_response"]
    # output_schema 生效：内部字段（messages、audit_log…）不会返回给调用方
    assert "messages" not in result.value


async def test_billing_needs_refund_approval_then_reply_review(graph):
    cfg = config("B1")
    result = await graph.ainvoke(
        make_input("B1", "订单 ORD-1001 被扣了两次钱！"), cfg, context=Context(), version="v2"
    )
    # 第一次暂停：工具 issue_refund 里的 interrupt（在账单子图内部）
    assert result.interrupts[0].value["type"] == "refund_approval"

    result = await graph.ainvoke(Command(resume={"approved": True}), cfg, context=Context(), version="v2")
    # 第二次暂停：human_review 节点
    assert result.interrupts[0].value["type"] == "reply_review"
    assert "退款" in result.interrupts[0].value["draft_response"]

    result = await graph.ainvoke(
        Command(resume={"approved": False, "note": "金额较大，电话沟通"}), cfg, context=Context(), version="v2"
    )
    assert result.value["status"] == "escalated"


async def test_flaky_email_service_is_retried(graph):
    result = await graph.ainvoke(
        make_input("F1", "密码有什么要求？"), config("F1"), context=Context(email_service_mode="flaky")
    )
    assert result["status"] == "sent"
    state = await graph.aget_state(config("F1"))
    assert "send_reply: 第 2 次尝试发送成功" in state.values["audit_log"]


async def test_email_service_down_triggers_error_handler(graph):
    result = await graph.ainvoke(
        make_input("D1", "密码有什么要求？"), config("D1"), context=Context(email_service_mode="down")
    )
    # error_handler 兜底：图没有崩，而是记录失败并继续到 save_memory
    assert result["status"] == "send_failed"


async def test_ticket_is_saved_to_long_term_memory(graph, store):
    await graph.ainvoke(make_input("M1", "密码有什么要求？"), config("M1"), context=Context())
    items = store.search(("customers", "zhang@example_com", "tickets"))
    assert [item.key for item in items] == ["M1"]
    assert items[0].value["status"] == "sent"


async def test_partial_execution_only_draft_response(graph):
    """只测试 draft_response：用 update_state(as_node=...) 伪造上游节点的输出，再用 interrupt_after 截停。"""
    cfg = config("P1")
    await graph.aupdate_state(
        cfg,
        {
            **make_input("P1", "我想退订"),
            "classification": {"intent": "question", "urgency": "low", "topic": "退订", "summary": "退订"},
            "search_results": ["订阅 7 天内可无理由全额退款。"],
            "customer_profile": {"name": "测试用户", "tier": "基础版"},
        },
        as_node="lookup_customer",  # 假装 lookup_customer 刚执行完 → 下一个是 draft_response
    )
    await graph.ainvoke(None, cfg, context=Context(), interrupt_after=["draft_response"])

    state = await graph.aget_state(cfg)
    assert state.next == ("send_reply",)  # 低紧急度 → 不需要审核
    assert "测试用户" in state.values["draft_response"]
    assert "全额退款" in state.values["draft_response"]
