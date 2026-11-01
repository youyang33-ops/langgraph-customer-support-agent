"""账单子图可以调用的工具。

📖 对应文档：
- Quickstart → Define tools and model
- Interrupts → Interrupts in tools：把“审批”写在工具内部，任何地方调用这个工具都会先暂停等人确认
"""

from langchain_core.tools import tool
from langgraph.types import interrupt

from support_agent import services


@tool
def get_order(order_id: str) -> str:
    """根据订单号查询订单详情和扣款记录。"""
    return services.get_order_json(order_id)


@tool
def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """给订单发起退款。会先暂停，等待人工批准。"""
    # interrupt() 会抛出一个特殊异常让整张图暂停，并把这个 dict 交给调用方。
    # 调用方用 Command(resume={...}) 恢复后，整个工具函数会【从头重新执行】，
    # 这一次 interrupt() 直接返回 resume 的值。
    # 所以 interrupt() 之前的代码必须是幂等的（这里之前没有副作用，安全）。
    decision = interrupt(
        {
            "type": "refund_approval",
            "question": f"是否批准给订单 {order_id} 退款 {amount} 元？（原因：{reason}）",
            "order_id": order_id,
            "amount": amount,
        }
    )
    if not decision.get("approved"):
        return f"退款被人工拒绝：{decision.get('note') or '无备注'}"
    # 审批人可以修改金额
    final_amount = decision.get("amount", amount)
    return services.refund(order_id, final_amount)


BILLING_TOOLS = [get_order, issue_refund]
