"""02 · 五种工作流模式 + Agent：用 Graph API 控制流程走向。

📖 对应文档：Workflows and agents、Use the graph API（branches / Send / Command / loops）、
   Graph API overview（Send、Command、RemainingSteps、recursion_limit）
▶ 运行：python examples/02_workflows_and_agents.py

为了离线可运行，这里用普通函数代替 LLM 调用（fake_llm），重点看图的结构。
把 fake_llm 换成 model.invoke(...) 就是真实项目。

模式对照：
    提示链 Prompt chaining    → 普通边串起来
    并行   Parallelization     → 一个节点连多条边（fan-out），再汇合（fan-in）
    路由   Routing             → add_conditional_edges 或 Command(goto=...)
    编排-执行 Orchestrator-worker → Send API 动态创建 N 个并行 worker
    评估-优化 Evaluator-optimizer → 带条件边的循环
    Agent                     → LLM 决定调用哪个工具、何时停止（见 support_agent/subgraphs/billing_agent.py）
"""

import operator
from typing import Annotated, Literal

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from typing_extensions import TypedDict


def fake_llm(prompt: str) -> str:
    return f"<{prompt}>"


# ---------------------------------------------------------------------------
# 1. 提示链：生成 → 检查 → 润色
# ---------------------------------------------------------------------------
class ChainState(TypedDict):
    topic: str
    draft: str
    final: str


def demo_prompt_chaining():
    builder = StateGraph(ChainState)
    builder.add_node("generate", lambda s: {"draft": fake_llm(f"写一个关于{s['topic']}的笑话")})
    builder.add_node("polish", lambda s: {"final": fake_llm(f"润色：{s['draft']}")})
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "polish")
    builder.add_edge("polish", END)
    print("① 提示链：", builder.compile().invoke({"topic": "猫"})["final"])


# ---------------------------------------------------------------------------
# 2. 并行：同时写故事、笑话、诗，再汇总
# ---------------------------------------------------------------------------
class ParallelState(TypedDict):
    topic: str
    parts: Annotated[list[str], operator.add]  # 三个节点同一步写入，必须有 reducer
    combined: str


def demo_parallelization():
    builder = StateGraph(ParallelState)
    for kind in ("故事", "笑话", "诗"):
        builder.add_node(kind, lambda s, kind=kind: {"parts": [fake_llm(f"{s['topic']}的{kind}")]})
        builder.add_edge(START, kind)  # fan-out
    builder.add_node("aggregate", lambda s: {"combined": " + ".join(s["parts"])})
    builder.add_edge(["故事", "笑话", "诗"], "aggregate")  # fan-in：等三个都完成
    builder.add_edge("aggregate", END)
    print("② 并行：", builder.compile().invoke({"topic": "猫", "parts": []})["combined"])


# ---------------------------------------------------------------------------
# 3. 路由：两种写法
# ---------------------------------------------------------------------------
class RouteState(TypedDict):
    request: str
    output: str


def route(state: RouteState) -> Literal["write_poem", "write_joke"]:
    """写法 A：条件边的路由函数，只返回下一个节点名，不改 State。"""
    return "write_poem" if "诗" in state["request"] else "write_joke"


def classify(state: RouteState) -> Command[Literal["write_poem", "write_joke"]]:
    """写法 B：节点返回 Command，同时“更新 State + 决定去哪”。"""
    goto = "write_poem" if "诗" in state["request"] else "write_joke"
    return Command(update={"output": f"[路由到 {goto}] "}, goto=goto)


def demo_routing():
    # 写法 A
    a = StateGraph(RouteState)
    a.add_node("write_poem", lambda s: {"output": fake_llm("诗")})
    a.add_node("write_joke", lambda s: {"output": fake_llm("笑话")})
    a.add_conditional_edges(START, route)
    print("③ 路由(条件边)：", a.compile().invoke({"request": "来首诗"})["output"])

    # 写法 B
    b = StateGraph(RouteState)
    b.add_node("classify", classify)
    b.add_node("write_poem", lambda s: {"output": s["output"] + fake_llm("诗")})
    b.add_node("write_joke", lambda s: {"output": s["output"] + fake_llm("笑话")})
    b.add_edge(START, "classify")
    print("③ 路由(Command)：", b.compile().invoke({"request": "讲个笑话"})["output"])


# ---------------------------------------------------------------------------
# 4. 编排-执行（Map-Reduce）：编排者拆出 N 个小节，Send 动态创建 N 个 worker
# ---------------------------------------------------------------------------
class ReportState(TypedDict):
    topic: str
    sections: list[str]
    completed: Annotated[list[str], operator.add]
    report: str


class WorkerState(TypedDict):
    section: str  # worker 有自己的输入 State，由 Send 传入


def orchestrator(state: ReportState) -> dict:
    return {"sections": [f"{state['topic']}-背景", f"{state['topic']}-方法", f"{state['topic']}-结论"]}


def assign_workers(state: ReportState) -> list[Send]:
    # 返回 Send 列表 = 下一步并行运行 len(sections) 个 write_section，各自拿不同的输入
    return [Send("write_section", {"section": s}) for s in state["sections"]]


def demo_orchestrator_worker():
    builder = StateGraph(ReportState)
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("write_section", lambda s: {"completed": [fake_llm(s["section"])]})
    builder.add_node("synthesizer", lambda s: {"report": "\n".join(s["completed"])})
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", assign_workers, ["write_section"])
    builder.add_edge("write_section", "synthesizer")
    builder.add_edge("synthesizer", END)
    result = builder.compile().invoke({"topic": "LangGraph", "completed": []})
    print("④ 编排-执行：\n" + result["report"])


# ---------------------------------------------------------------------------
# 5. 评估-优化：生成 → 评估 → 不合格就带着反馈重来（循环）
# ---------------------------------------------------------------------------
class LoopState(TypedDict):
    attempt: int
    joke: str
    feedback: str


def generator(state: LoopState) -> dict:
    attempt = state.get("attempt", 0) + 1
    return {"attempt": attempt, "joke": fake_llm(f"第{attempt}版笑话，参考反馈:{state.get('feedback', '无')}")}


def evaluator(state: LoopState) -> dict:
    return {"feedback": "不够好笑" if state["attempt"] < 3 else "通过"}


def demo_evaluator_optimizer():
    builder = StateGraph(LoopState)
    builder.add_node("generator", generator)
    builder.add_node("evaluator", evaluator)
    builder.add_edge(START, "generator")
    builder.add_edge("generator", "evaluator")
    builder.add_conditional_edges(
        "evaluator", lambda s: END if s["feedback"] == "通过" else "generator", ["generator", END]
    )
    app = builder.compile()
    print("⑤ 评估-优化：", app.invoke({"attempt": 0})["joke"])

    # 循环一定要有出口；recursion_limit 是兜底（默认很大，这里设小一点演示）
    try:
        app.invoke({"attempt": 0}, {"recursion_limit": 4})
    except GraphRecursionError:
        print("⑤ recursion_limit=4 时触发 GraphRecursionError（每执行一个节点算一步）")


if __name__ == "__main__":
    demo_prompt_chaining()
    demo_parallelization()
    demo_routing()
    demo_orchestrator_worker()
    demo_evaluator_optimizer()
