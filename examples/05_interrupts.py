"""05 · Human-in-the-loop：interrupt() 暂停 + Command(resume=...) 恢复。

📖 对应文档：Interrupts（全部小节）、Graph API → Command → resume
▶ 运行：python examples/05_interrupts.py

三条铁律（文档 Rules of interrupts）：
1. 恢复时，节点会【从头重新执行】，interrupt() 之前的代码会再跑一遍 → 之前的副作用必须幂等；
2. 不要用裸 try/except 包住 interrupt()（它靠抛异常实现暂停）；
3. 一个节点里有多个 interrupt 时，按【调用顺序】匹配恢复值，不要动态改变顺序。
"""

import operator
from typing import Annotated

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from typing_extensions import TypedDict


class State(TypedDict):
    action: str
    result: str


# ---------------------------------------------------------------------------
# 1. 审批：approve / reject，并演示“节点会重跑”
# ---------------------------------------------------------------------------
def approval_node(state: State):
    print("   （approval_node 开始执行）")  # 恢复时这行会再打印一次！
    approved = interrupt({"question": f"确定要执行「{state['action']}」吗？"})
    return {"result": "已执行" if approved else "已取消"}


def demo_approve():
    graph = (
        StateGraph(State)
        .add_node("approval_node", approval_node)
        .add_edge(START, "approval_node")
        .compile(checkpointer=InMemorySaver())  # interrupt 必须有 checkpointer
    )
    config = {"configurable": {"thread_id": "approve"}}

    # version="v2"：invoke 返回 GraphOutput，State 和 interrupt 分开放（推荐写法）
    out = graph.invoke({"action": "删除数据库"}, config, version="v2")
    print("① 暂停：", out.interrupts[0].value)
    out = graph.invoke(Command(resume=False), config, version="v2")
    print("① 恢复后：", out.value)


# ---------------------------------------------------------------------------
# 2. response_schema：恢复值自动校验 + 转换成对象（langgraph>=1.2.12）
# ---------------------------------------------------------------------------
class Decision(BaseModel):
    approved: bool
    note: str | None = None


def review_node(state: State):
    decision = interrupt({"question": "审核"}, response_schema=Decision)
    return {"result": f"approved={decision.approved}, note={decision.note}"}


def demo_response_schema():
    graph = StateGraph(State).add_node("review", review_node).add_edge(START, "review").compile(
        checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "schema"}}
    out = graph.invoke({"action": "x"}, config, version="v2")
    print("② 前端可以用这个 JSON Schema 渲染表单：", list(out.interrupts[0].response_schema["properties"]))
    print("② ", graph.invoke(Command(resume={"approved": True, "note": "ok"}), config, version="v2").value)


# ---------------------------------------------------------------------------
# 3. 校验人工输入：在节点里循环 interrupt，直到输入合法
# ---------------------------------------------------------------------------
class AgeState(TypedDict):
    age: int


def ask_age(state: AgeState):
    prompt = "请输入年龄"
    while True:
        answer = interrupt(prompt)
        if isinstance(answer, int) and answer > 0:
            return {"age": answer}
        prompt = f"「{answer}」不是合法年龄，请重新输入"


def demo_validation():
    graph = StateGraph(AgeState).add_node("ask_age", ask_age).add_edge(START, "ask_age").compile(
        checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "age"}}
    graph.invoke({"age": 0}, config)
    out = graph.invoke(Command(resume="十八"), config, version="v2")
    print("③ 第二次提示：", out.interrupts[0].value)
    print("③ 最终：", graph.invoke(Command(resume=18), config, version="v2").value)


# ---------------------------------------------------------------------------
# 4. 多个并行 interrupt：用 {interrupt_id: 值} 一次性恢复
# ---------------------------------------------------------------------------
class MultiState(TypedDict):
    answers: Annotated[list[str], operator.add]


def demo_multiple():
    graph = (
        StateGraph(MultiState)
        .add_node("a", lambda s: {"answers": [f"a:{interrupt('问题A')}"]})
        .add_node("b", lambda s: {"answers": [f"b:{interrupt('问题B')}"]})
        .add_edge(START, "a")
        .add_edge(START, "b")
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "multi"}}
    out = graph.invoke({"answers": []}, config, version="v2")
    resume = {item.id: f"回答{item.value}" for item in out.interrupts}
    print("④ ", graph.invoke(Command(resume=resume), config, version="v2").value)


# ---------------------------------------------------------------------------
# 5. 静态断点：compile(interrupt_before=[...])，常用于调试
# ---------------------------------------------------------------------------
def demo_breakpoint():
    graph = (
        StateGraph(State)
        .add_node("step1", lambda s: {"result": "step1 done"})
        .add_node("step2", lambda s: {"result": "step2 done"})
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .compile(checkpointer=InMemorySaver(), interrupt_before=["step2"])
    )
    config = {"configurable": {"thread_id": "bp"}}
    graph.invoke({"action": ""}, config)
    print("⑤ 断点停在：", graph.get_state(config).next)
    print("⑤ 继续：", graph.invoke(None, config))  # 传 None = 从断点继续


if __name__ == "__main__":
    demo_approve()
    demo_response_schema()
    demo_validation()
    demo_multiple()
    demo_breakpoint()
