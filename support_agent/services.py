"""模拟的外部系统：知识库、CRM、订单库、工单系统、邮件服务。

真实项目里这些是向量数据库、内部 HTTP 接口等。这里用内存数据代替，
让你专注于 LangGraph 本身。学习时可以快速浏览，不必细读。
"""

import itertools
import json


class EmailServiceError(ConnectionError):
    """邮件服务暂时不可用（瞬时错误，值得重试）。"""


# ---------------------------------------------------------------------------
# 知识库（Doc Search 用）
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    "重置密码：进入「设置 > 账号安全 > 修改密码」，点击「忘记密码」会向注册邮箱发送重置链接，链接 30 分钟内有效。",
    "密码规则：至少 12 位，必须同时包含大小写字母、数字和符号。",
    "导出 PDF 崩溃：v3.2.0 已知问题，导出超过 200 页的报表时会崩溃。临时方案：先按月份筛选再导出，或改用 Excel 格式。修复将在 v3.2.1 发布。",
    "退款政策：重复扣费会在核实后 1-3 个工作日原路退回；订阅 7 天内可无理由全额退款。",
    "发票：在「账单中心 > 发票管理」申请电子发票，支持增值税普通发票和专用发票。",
    "产品路线图：移动端夜间模式（深色模式）已排期，预计下个季度上线，可在「设置 > 实验功能」提前试用。",
    "API 504 排查：504 通常是网关超时。请检查单次请求是否超过 30 秒、是否触发了限流（每分钟 600 次），建议开启重试并使用分页接口。",
    "套餐说明：基础版每月 99 元，专业版每月 299 元，企业版按需报价，支持按年付费享 8 折。",
]


def search_knowledge_base(query: str, k: int = 3) -> list[str]:
    """极简检索：按字符二元组重合度打分（真实项目用向量检索）。"""

    def bigrams(text: str) -> set[str]:
        text = text.lower()
        return {text[i : i + 2] for i in range(len(text) - 1)}

    q = bigrams(query)
    scored = [(len(q & bigrams(doc)), doc) for doc in KNOWLEDGE_BASE]
    scored = [item for item in scored if item[0] >= 2]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:k]]


# ---------------------------------------------------------------------------
# CRM（客户资料）
# ---------------------------------------------------------------------------

CUSTOMERS = {
    "zhang@example.com": {"name": "张先生", "tier": "专业版", "since": "2023-05"},
    "li@example.com": {"name": "李女士", "tier": "企业版", "since": "2021-11"},
}


def fetch_customer_profile(email: str) -> dict:
    return CUSTOMERS.get(email, {"name": "客户", "tier": "基础版", "since": "未知"})


# ---------------------------------------------------------------------------
# 订单库（账单子图里的工具用）
# ---------------------------------------------------------------------------

ORDERS = {
    "ORD-1001": {"order_id": "ORD-1001", "product": "专业版月度订阅", "amount": 299,
                 "charges": ["2026-10-01 扣款 299 元", "2026-10-01 扣款 299 元"]},
    "ORD-1002": {"order_id": "ORD-1002", "product": "基础版月度订阅", "amount": 99,
                 "charges": ["2026-10-03 扣款 99 元"]},
}
REFUNDS: list[dict] = []


def get_order_json(order_id: str) -> str:
    order = ORDERS.get(order_id.upper())
    return json.dumps(order, ensure_ascii=False) if order else f"未找到订单 {order_id}"


def refund(order_id: str, amount: float) -> str:
    REFUNDS.append({"order_id": order_id, "amount": amount})
    return f"已为订单 {order_id} 发起退款 {amount} 元，1-3 个工作日到账"


# ---------------------------------------------------------------------------
# 工单系统（Bug Track 用）
# ---------------------------------------------------------------------------

_ticket_counter = itertools.count(12345)


def create_bug_ticket(summary: str) -> str:
    return f"BUG-{next(_ticket_counter)}"


# ---------------------------------------------------------------------------
# 邮件服务（Send Reply 用）
# ---------------------------------------------------------------------------

SENT_EMAILS: list[dict] = []


async def send_email(to: str, body: str, *, mode: str, attempt: int) -> None:
    """mode 由 Context.email_service_mode 控制，用来演示重试和兜底。"""
    if mode == "down" or (mode == "flaky" and attempt == 1):
        raise EmailServiceError(f"邮件网关无响应（第 {attempt} 次尝试）")
    SENT_EMAILS.append({"to": to, "body": body})
