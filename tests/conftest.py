"""测试公共配置。

📖 对应文档：Test → Getting started
“每个测试都重新编译图、用一个新的 checkpointer”，保证测试之间互不影响。
"""

import os

os.environ["MOCK_LLM"] = "1"  # 必须在导入 support_agent 之前设置：测试永远用确定性的假模型

import pytest  # noqa: E402
from langgraph.cache.memory import InMemoryCache  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from support_agent.graph import build_graph  # noqa: E402


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def graph(store):
    return build_graph(checkpointer=InMemorySaver(), store=store, cache=InMemoryCache())


def make_input(email_id: str, content: str, sender: str = "zhang@example.com") -> dict:
    return {"email_id": email_id, "sender_email": sender, "email_content": content}
