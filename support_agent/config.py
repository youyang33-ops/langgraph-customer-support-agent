"""模型与运行时上下文配置。

📖 对应文档：
- Graph API overview → Runtime context（context_schema）
- Workflows and agents → LLMs and augmentations（结构化输出 / 绑定工具）

所有节点都通过 get_model() 拿模型，换模型只改 .env，不用改业务代码。
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Context:
    """运行时上下文（Runtime Context），在 invoke/stream 时通过 `context=Context(...)` 传入。

    和 State 的区别（面试常问）：
    - State：图运行过程中节点不断读写的数据，会被 checkpointer 保存，可以回放/分叉；
    - Context：调用方传入的“只读依赖”，比如租户、用户、开关配置，不会存进 checkpoint。

    节点里这样读：`def node(state, runtime: Runtime[Context]): runtime.context.team_name`
    """

    team_name: str = "小L智能客服"
    # 演示容错用：模拟邮件服务故障
    #   "ok"    正常发送
    #   "flaky" 第 1 次失败、重试后成功（演示 RetryPolicy）
    #   "down"  一直失败（演示 error_handler 兜底）
    email_service_mode: Literal["ok", "flaky", "down"] = "ok"


def use_mock() -> bool:
    """没配置 API key，或显式设置 MOCK_LLM=1 时，使用离线假模型。"""
    return os.getenv("MOCK_LLM") == "1" or not os.getenv("LLM_API_KEY")


@lru_cache
def get_model():
    """返回聊天模型（LangChain 的 BaseChatModel）。

    LangGraph 本身不关心你用什么模型——节点就是普通 Python 函数。
    这里用 LangChain 的模型类，是因为它统一了“工具调用 / 结构化输出 / 流式输出”的接口。
    """
    if use_mock():
        from support_agent.mock_llm import MockChatModel

        return MockChatModel()

    from langchain_openai import ChatOpenAI

    # DeepSeek、通义千问都兼容 OpenAI 接口，只需要换 base_url 和 model
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=0,
    )


def get_structured_model(schema):
    """让模型按 Pydantic schema 输出结构化结果（文档：LLMs and augmentations）。

    method="function_calling"：langchain-openai 默认走 OpenAI 的 json_schema 模式，
    DeepSeek / 通义千问不支持；用“工具调用”实现结构化输出则都兼容。
    """
    return get_model().with_structured_output(schema, method="function_calling")
