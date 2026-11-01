"""单元测试：单独测试节点和子图。

📖 对应文档：Test → Testing individual nodes and edges
"""

from langgraph.types import Command

from support_agent import nodes
from support_agent.subgraphs.rag import rag_graph


def test_read_email_node_via_graph_nodes(graph):
    # compiled_graph.nodes["名字"] 可以单独执行一个节点（不经过 checkpointer）
    result = graph.nodes["read_email"].invoke(
        {"email_id": "E1", "sender_email": "a@b.com", "email_content": "你好"}
    )
    assert result["messages"][0].content == "你好"
    assert result["audit_log"] == ["read_email: 收到 a@b.com 的邮件"]


def test_classify_routes_billing_in_parallel_with_lookup():
    # 节点本质就是普通函数，也可以直接调用
    command = nodes.classify_intent(
        {"email_id": "E1", "sender_email": "a@b.com", "email_content": "我被重复扣费了，请退款"}
    )
    assert isinstance(command, Command)
    assert command.update["classification"]["intent"] == "billing"
    assert command.goto == ["billing_agent", "lookup_customer"]


def test_rag_subgraph_rewrites_query_when_nothing_found():
    # 子图是独立编译的图，可以单独测试
    result = rag_graph.invoke({"question": "手机上能用暗黑主题吗"})
    assert result["rewrite_count"] == 1
    assert result["relevant"] is True
    assert any("夜间模式" in doc for doc in result["documents"])
