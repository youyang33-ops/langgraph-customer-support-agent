"""09 · Functional API：不画图，用普通函数 + 装饰器获得同样的持久化/中断/流式能力。

📖 对应文档：Functional API overview、Use the functional API、Choosing between Graph and Functional APIs
▶ 运行：python examples/09_functional_api.py

@entrypoint：工作流入口（相当于一张图），可以挂 checkpointer / store
@task      ：一个可以被 checkpoint 的工作单元（相当于节点），调用后返回 future，.result() 取值

和 Graph API 怎么选？
    Graph API：流程复杂、需要可视化、多人协作、需要显式的 State 和分支 → 本项目主体
    Functional API：在已有代码里“加上”持久化和人工审核，改动最小，用 if/for 控制流程
两者底层都是同一个 Pregel 运行时，可以混用。
"""

import time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt


@task
def write_essay(topic: str) -> str:
    print(f"   （write_essay 真正执行了：{topic}）")
    time.sleep(0.5)  # 模拟耗时的 LLM 调用
    return f"一篇关于「{topic}」的文章"


@task
def translate(text: str, lang: str) -> str:
    return f"[{lang}] {text}"


@entrypoint(checkpointer=InMemorySaver())
def workflow(topic: str) -> dict:
    essay = write_essay(topic).result()

    # 多个 task 同时调用 = 并行执行
    futures = [translate(essay, lang) for lang in ("EN", "JP")]
    translations = [f.result() for f in futures]

    # 和 Graph API 一样用 interrupt 等人工审核。
    # 恢复时 entrypoint 会从头执行，但已完成的 task 结果直接从 checkpoint 读取，不会重跑
    approved = interrupt({"essay": essay, "action": "请审核"})
    return {"essay": essay, "translations": translations, "approved": approved}


# previous：拿到同一 thread 上一次的返回值 → 短期记忆
# entrypoint.final(value=返回给调用方的值, save=存进 checkpoint 给下次用的值)
@entrypoint(checkpointer=InMemorySaver())
def counter(increment: int, *, previous: int | None = None) -> entrypoint.final[str, int]:
    total = (previous or 0) + increment
    return entrypoint.final(value=f"累计 {total}", save=total)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "essay"}}
    out = workflow.invoke("LangGraph", config, version="v2")
    print("① 暂停，等待审核：", out.interrupts[0].value)
    out = workflow.invoke(Command(resume=True), config, version="v2")
    print("① 恢复后（注意 write_essay 没有再执行）：", out.value)

    config = {"configurable": {"thread_id": "counter"}}
    print("② ", counter.invoke(3, config), "→", counter.invoke(5, config))
