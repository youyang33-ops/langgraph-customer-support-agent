"""自定义流式投影（StreamTransformer）。

📖 对应文档：Event streaming → Build your own projection

节点里用 get_stream_writer() 写的数据会进入 "custom" 流。
这个 Transformer 把其中的进度事件挑出来，做成 `stream.extensions["progress"]` 这个投影，
调用方就能像读 `stream.messages` 一样直接读进度，而不用自己解析原始事件。
"""

from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer


class ProgressTransformer(StreamTransformer):
    # 声明需要的底层 stream_mode；不声明的话 custom 事件根本不会被产出
    required_stream_modes = ("custom",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        # 带名字的 channel：每次 push 也会作为 "custom:progress" 事件进入主事件流
        self.progress = StreamChannel("progress")

    def init(self) -> dict:
        return {"progress": self.progress}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "custom":
            data = event["params"]["data"]
            if isinstance(data, dict) and "node" in data:
                self.progress.push(data)
        return True  # True = 保留原始事件
