"""08 · 容错：重试、超时、错误兜底、图级默认值、节点缓存、优雅停机。

📖 对应文档：Fault tolerance、Use the graph API（retry / timeout / caching）、Thinking in LangGraph → Handle errors
▶ 运行：python examples/08_fault_tolerance.py

执行顺序（固定）：节点抛异常（含超时）→ retry_policy 判断是否重试 → 重试耗尽 → error_handler 兜底
没有 error_handler 时，异常向上抛出，整个 run 失败（但 checkpoint 还在，可以修好后继续）。

错误分类（文档表格）：
    瞬时错误（网络、限流）   → RetryPolicy 自动重试
    LLM 可修复（工具报错）  → 把错误写进 State，让 LLM 看到后重来
    需要人补充信息          → interrupt()
    重试后仍失败但可补偿    → error_handler（Saga 模式）
    未知错误               → 直接抛出，别吞掉
"""

import asyncio
import time

from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained, NodeError
from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime, RunControl
from langgraph.types import CachePolicy, Command, RetryPolicy, TimeoutPolicy
from typing_extensions import TypedDict


class State(TypedDict):
    status: str


# ---------------------------------------------------------------------------
# 1. 重试 + execution_info：第几次尝试
# ---------------------------------------------------------------------------
def flaky_api(state: State, runtime: Runtime):
    attempt = runtime.execution_info.node_attempt
    print(f"   flaky_api 第 {attempt} 次尝试")
    if attempt < 3:
        raise TimeoutError("网络抖动")  # 默认会重试的异常（OSError 系列除外，见下方注释）
    return {"status": f"第 {attempt} 次成功"}


def demo_retry():
    graph = (
        StateGraph(State)
        # 默认 retry_on 不重试 ValueError/TypeError/OSError 等“代码 bug 类”异常；
        # 这里显式指定 TimeoutError（它是 OSError 子类，默认不会重试）
        .add_node("flaky_api", flaky_api, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, retry_on=TimeoutError))
        .add_edge(START, "flaky_api")
        .compile()
    )
    print("① 重试：", graph.invoke({"status": ""}))


# ---------------------------------------------------------------------------
# 2. 超时（只支持 async 节点）+ error_handler 兜底
# ---------------------------------------------------------------------------
async def slow_llm(state: State):
    await asyncio.sleep(5)
    return {"status": "不会执行到这里"}


def on_error(state: State, error: NodeError) -> Command:
    # error.node：哪个节点失败；error.error：原始异常（这里是 NodeTimeoutError）
    return Command(update={"status": f"{error.node} 失败已兜底：{type(error.error).__name__}"}, goto="fallback")


async def demo_timeout_and_handler():
    graph = (
        StateGraph(State)
        .add_node("slow_llm", slow_llm, timeout=TimeoutPolicy(run_timeout=0.3), error_handler=on_error)
        .add_node("fallback", lambda s: {"status": s["status"] + " → 返回兜底话术"})
        .add_edge(START, "slow_llm")
        .compile()
    )
    print("② 超时+兜底：", await graph.ainvoke({"status": ""}))


# ---------------------------------------------------------------------------
# 3. set_node_defaults：一次性给所有节点配置（单节点配置优先）
# ---------------------------------------------------------------------------
def demo_node_defaults():
    def boom(state: State):
        raise RuntimeError("意外错误")

    # error 参数靠【类型注解 NodeError】注入，所以不能写成没有注解的 lambda
    def default_handler(state: State, error: NodeError) -> dict:
        return {"status": f"默认兜底：{error.error}"}

    graph = (
        StateGraph(State)
        .set_node_defaults(error_handler=default_handler)
        .add_node("boom", boom)
        .add_edge(START, "boom")
        .compile()
    )
    print("③ 图级默认 error_handler：", graph.invoke({"status": ""}))


# ---------------------------------------------------------------------------
# 4. 节点缓存：相同输入在 ttl 内直接返回缓存结果
# ---------------------------------------------------------------------------
def demo_cache():
    class CalcState(TypedDict):
        x: int
        result: int

    def expensive(state: CalcState):
        time.sleep(1)
        return {"result": state["x"] * 2}

    graph = (
        StateGraph(CalcState)
        .add_node("expensive", expensive, cache_policy=CachePolicy(ttl=60))
        .add_edge(START, "expensive")
        .compile(cache=InMemoryCache())
    )
    for i in range(2):
        start = time.time()
        updates = graph.invoke({"x": 5}, stream_mode="updates")
        print(f"④ 第 {i + 1} 次：{updates}  耗时 {time.time() - start:.1f}s")


# ---------------------------------------------------------------------------
# 5. 优雅停机：RunControl.request_drain()，在 super-step 边界停下并保存 checkpoint
# ---------------------------------------------------------------------------
def demo_drain():
    control = RunControl()

    def step1(state: State):
        control.request_drain("收到 SIGTERM")  # 真实场景在信号处理函数里调用
        return {"status": "step1 完成"}

    graph = (
        StateGraph(State)
        .add_node("step1", step1)
        .add_node("step2", lambda s: {"status": s["status"] + " → step2 完成"})
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "drain"}}
    try:
        graph.invoke({"status": ""}, config, control=control)
    except GraphDrained as e:
        print(f"⑤ 停机：{e.reason}，下一步是 {graph.get_state(config).next}")
    print("⑤ 重启后继续：", graph.invoke(None, config))


if __name__ == "__main__":
    demo_retry()
    asyncio.run(demo_timeout_and_handler())
    demo_node_defaults()
    demo_cache()
    demo_drain()
