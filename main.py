"""命令行入口：运行客服 Agent，并在终端里完成人工审核。

用法：
    python main.py                        # 依次处理 5 封示例邮件，需要审核时在终端询问你
    python main.py run -s billing         # 只跑一个场景：question/bug/billing/feature/complex
    python main.py run --auto-approve     # 所有审核自动批准
    python main.py run --email-mode flaky # 邮件服务第一次失败 → 看 RetryPolicy 重试
    python main.py run --email-mode down  # 邮件服务一直失败 → 看 error_handler 兜底
    python main.py history <thread_id>    # 查看一张工单的 checkpoint 历史
    python main.py resume <thread_id>     # 审核时按 Ctrl+C 退出，之后用它继续（持久化执行）
    python main.py memory <email>         # 查看某个客户的长期记忆

📖 对应文档：Event streaming（stream_events v3）、Interrupts（Command(resume=...)）、
   Checkpointers（SQLite、get_state、get_state_history）、Stores（长期记忆）
"""

import argparse
import asyncio
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite import AsyncSqliteStore
from langgraph.types import Command

from support_agent.config import Context, use_mock
from support_agent.graph import build_graph

# stream_events(version="v3") 目前标记为实验特性，会打印 Beta 警告，这里静默掉
warnings.filterwarnings("ignore", message=".*v3 streaming protocol.*")

DATA_DIR = Path(__file__).parent / "data"

SCENARIOS = {
    "question": ("zhang@example.com", "你好，我忘记密码了，请问怎么重置？"),
    "bug": ("li@example.com", "导出 PDF 的时候软件直接崩溃了，报表大概 300 页。"),
    "billing": ("zhang@example.com", "订单 ORD-1001 这个月被扣了两次钱！请马上处理！"),
    "feature": ("li@example.com", "能不能给手机 App 加一个暗黑主题？晚上看太刺眼了。"),
    "complex": ("wang@example.com", "我们的 API 集成间歇性返回 504 错误，已经影响线上业务。"),
}


@asynccontextmanager
async def open_graph():
    """用 SQLite 做 checkpointer 和 store：进程退出后状态和记忆都还在。"""
    DATA_DIR.mkdir(exist_ok=True)
    async with (
        AsyncSqliteSaver.from_conn_string(str(DATA_DIR / "checkpoints.sqlite")) as checkpointer,
        AsyncSqliteStore.from_conn_string(str(DATA_DIR / "memory.sqlite")) as store,
    ):
        await store.setup()
        yield build_graph(checkpointer=checkpointer, store=store, cache=InMemoryCache())


# ---------------------------------------------------------------------------
# 人工审核：把 interrupt 的 payload 展示给人，收集输入作为 resume 值
# ---------------------------------------------------------------------------


def ask_human(payload: dict, auto_approve: bool) -> dict:
    print(f"\n  ⏸  图已暂停，等待人工：{payload['question']}")
    if payload["type"] == "reply_review":
        print("  ┌─ 回复草稿 " + "─" * 40)
        for line in payload["draft_response"].splitlines():
            print(f"  │ {line}")
        print("  └" + "─" * 50)
    if auto_approve:
        print("  → 自动批准")
        return {"approved": True}

    if payload["type"] == "refund_approval":
        answer = input("  批准退款吗？[y/n] ").strip().lower()
        return {"approved": answer != "n", "note": None if answer != "n" else "人工拒绝"}

    answer = input("  y=批准发送 / e=修改后发送 / n=拒绝转人工 > ").strip().lower()
    if answer == "e":
        return {"approved": True, "edited_response": input("  输入新的回复内容：")}
    if answer == "n":
        return {"approved": False, "note": input("  拒绝原因：") or None}
    return {"approved": True}


# ---------------------------------------------------------------------------
# 运行一次图，并实时打印进度
# ---------------------------------------------------------------------------


async def run_until_pause(graph, graph_input, config, context) -> list:
    """跑到结束或下一次 interrupt，返回待处理的 interrupt 列表。"""
    stream = await graph.astream_events(graph_input, config, version="v3", context=context)

    async def show_progress():
        # ProgressTransformer 提供的自定义投影（见 support_agent/streaming.py）
        async for event in stream.extensions["progress"]:
            print(f"  ▸ [{event['node']}] {event['message']}")

    async def show_draft_tokens():
        # stream.messages：LLM 的流式输出，这里只打印“起草回复”节点的 token
        async for message in stream.messages:
            if message.node == "draft_response":
                print("  ✍  正在起草回复：", end="", flush=True)
                async for token in message.text:
                    print(token.replace("\n", " "), end="", flush=True)
                print()

    # 多个投影可以并发消费，互不影响（文档：Stream multiple projections）
    await asyncio.gather(show_progress(), show_draft_tokens())
    return await stream.interrupts()


async def finish(graph, interrupts, config, context, auto_approve: bool) -> None:
    """遇到 interrupt 就问人，再用 Command(resume=...) 恢复，直到整张图跑完。"""
    while interrupts:
        # 用 {interrupt_id: 值} 的形式恢复，同时有多个 interrupt 时也不会混淆
        answers = {item.id: ask_human(item.value, auto_approve) for item in interrupts}
        interrupts = await run_until_pause(graph, Command(resume=answers), config, context)

    values = (await graph.aget_state(config)).values
    print(f"\n  ✅ 完成：status={values.get('status')}  ticket={values.get('ticket_id', '-')}")
    print("  审计日志：" + " → ".join(entry.split(":")[0] for entry in values.get("audit_log", [])))


async def cmd_run(args) -> None:
    names = [args.scenario] if args.scenario else list(SCENARIOS)
    context = Context(email_service_mode=args.email_mode)
    async with open_graph() as graph:
        for name in names:
            sender, content = SCENARIOS[name]
            email_id = f"{name}-{int(time.time())}"
            # thread_id：一张工单 = 一个线程，所有 checkpoint 都挂在它下面
            config = {"configurable": {"thread_id": email_id}}
            print(f"\n{'=' * 60}\n📧 [{name}] {sender}：{content}\n   thread_id = {email_id}")
            graph_input = {"email_id": email_id, "sender_email": sender, "email_content": content}
            interrupts = await run_until_pause(graph, graph_input, config, context)
            await finish(graph, interrupts, config, context, args.auto_approve)


async def cmd_resume(args) -> None:
    """进程重启后继续：checkpointer 里存着暂停时的完整状态。"""
    config = {"configurable": {"thread_id": args.thread_id}}
    async with open_graph() as graph:
        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            print("这张工单没有处于暂停状态。")
            return
        print(f"工单暂停在节点：{snapshot.next}")
        # snapshot.interrupts：这个线程里尚未处理的 interrupt（从 SQLite 里读出来的）
        await finish(graph, snapshot.interrupts, config, Context(), args.auto_approve)


async def cmd_history(args) -> None:
    """列出一个线程的所有 checkpoint（最新的在前）—— time travel 的基础。"""
    config = {"configurable": {"thread_id": args.thread_id}}
    async with open_graph() as graph:
        print(f"{'step':>4}  {'来源':<6} {'下一步要执行的节点':<36} checkpoint_id")
        async for snap in graph.aget_state_history(config):
            meta = snap.metadata or {}
            print(
                f"{meta.get('step', ''):>4}  {meta.get('source', ''):<6} "
                f"{str(snap.next):<36} {snap.config['configurable']['checkpoint_id']}"
            )
        print("\n想“回到过去”重跑或分叉，见 examples/03_persistence_time_travel.py")


async def cmd_memory(args) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    async with AsyncSqliteStore.from_conn_string(str(DATA_DIR / "memory.sqlite")) as store:
        await store.setup()
        namespace = ("customers", args.email.replace(".", "_"), "tickets")
        items = await store.asearch(namespace, limit=20)
        print(f"{args.email} 的历史工单（{len(items)} 条）：")
        for item in items:
            print(f"  - {item.key}: {item.value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph 智能客服工单 Agent")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="处理示例邮件（默认命令）")
    run.add_argument("-s", "--scenario", choices=SCENARIOS)
    run.add_argument("--auto-approve", action="store_true")
    run.add_argument("--email-mode", choices=["ok", "flaky", "down"], default="ok")

    resume = sub.add_parser("resume", help="继续一张暂停中的工单")
    resume.add_argument("thread_id")
    resume.add_argument("--auto-approve", action="store_true")

    history = sub.add_parser("history", help="查看工单的 checkpoint 历史")
    history.add_argument("thread_id")

    memory = sub.add_parser("memory", help="查看客户长期记忆")
    memory.add_argument("email")

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["run"])

    print(f"🤖 模型：{'离线 MockChatModel（未配置 LLM_API_KEY）' if use_mock() else '真实 LLM'}")
    handlers = {"run": cmd_run, "resume": cmd_resume, "history": cmd_history, "memory": cmd_memory}
    try:
        asyncio.run(handlers[args.command](args))
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。状态已保存在 checkpointer 里，可用 `python main.py resume <thread_id>` 继续。")


if __name__ == "__main__":
    main()
