#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一阶段 PYNQ-Z2 接收端: UDP 接收 RK3588 检测结果 -> HDMI 叠加输出

== 部署步骤(在 PYNQ 上执行) ==
1. 把本文件拷到 PYNQ:  scp pynq_receiver.py xilinx@<PYNQ_IP>:/home/xilinx/
2. 登录 PYNQ:  ssh xilinx@<PYNQ_IP>   (默认密码 xilinx)
3. 确认 base overlay 的 HDMI 通路可用(电脑 HDMI 输出 -> PYNQ HDMI_IN)
4. 运行:  sudo python3 pynq_receiver.py --port 5000 --mode overlay
   调试期可先用 --mode print 只看数据不接 HDMI
5. RK3588 端:  python3 rk3588_sender.py   (默认即发往 192.168.2.99:5000)

== 时钟同步(测链路延迟需要) ==
两块板子都指向同一 NTP(如 RK3588 做 NTP server 或用同一上级),
否则 timestamp_ms 算出的延迟不可信。不跑延迟统计可忽略。

== 结构说明 ==
ReceiverThread: 收 UDP -> 解析 JSON -> 写入 DetectionStore(最新帧)
OverlayLoop:    从 HDMI_IN 抓帧 -> FPGA 边缘检测(占位) -> 叠加检测框/
                类别/置信度/FPS -> HDMI_OUT
FPGA 加速点已在 fpga_edge() 标出, 第二阶段把卷积实现挂到这里。
"""
import argparse
import json
import socket
import threading
import time
from collections import deque


class DetectionStore:
    """线程安全的最新检测结果缓存"""

    def __init__(self):
        self._lock = threading.Lock()
        self._msg = None

    def update(self, msg):
        with self._lock:
            self._msg = msg

    def latest(self):
        with self._lock:
            return self._msg


class ReceiverThread(threading.Thread):
    def __init__(self, port, store, stats):
        super().__init__(daemon=True)
        self.port = port
        self.store = store
        self.stats = stats

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        s.bind(("0.0.0.0", self.port))
        print(f"[rx] listening UDP :{self.port}")
        while True:
            data, _ = s.recvfrom(65535)
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            self.store.update(msg)
            self.stats.on_frame(msg)


class Stats:
    """接收速率 / 链路延迟统计(每 60 帧打印一行 JSON)"""

    def __init__(self, every=60):
        self.every = every
        self.reset()

    def reset(self):
        self.n = 0
        self.t0 = time.monotonic()
        self.lat = deque(maxlen=1000)

    def on_frame(self, msg):
        self.n += 1
        if "timestamp_ms" in msg:
            self.lat.append(time.time() * 1000 - msg["timestamp_ms"])
        if self.n >= self.every:
            span = time.monotonic() - self.t0
            lat_ms = sum(self.lat) / len(self.lat) if self.lat else -1
            print(json.dumps({"stat": 1, "rx_frames": self.n,
                              "rx_fps": round(self.n / span, 1),
                              "link_latency_ms_avg": round(lat_ms, 1),
                              "frame_id": msg.get("frame_id")}))
            self.reset()


def fpga_edge(frame):
    """FPGA 边缘检测占位函数。

    第二阶段在这里接入 HLS 实现的边缘检测/INT8 卷积 IP:
      - 通过 pynq.Overlay 加载自定义 bitstream
      - 用 mmio/axi_dma 把 frame 送 PL, 读回边缘图
    当前返回 None 表示跳过(纯透传)。
    """
    return None


class OverlayLoop:
    """HDMI 采集 -> 叠加 -> HDMI 输出(需要 pynq 环境)"""

    def __init__(self, store):
        from pynq import Overlay
        from pynq.lib.video import VideoMode
        base = Overlay("base.bit")
        base.init_video()
        self.hdmi_in = base.video.hdmi_in
        self.hdmi_out = base.video.hdmi_out
        self.mode = VideoMode(1280, 720, 24)
        self.hdmi_in.configure()
        self.hdmi_out.configure(self.mode)
        self.hdmi_in.start()
        self.hdmi_out.start()
        self.store = store
        print("[video] HDMI 1280x720 pipeline started")

    def run(self):
        import cv2
        fps_t0, fps_n, fps_val = time.monotonic(), 0, 0.0
        while True:
            fr = self.hdmi_in.readframe()
            edge = fpga_edge(fr)  # 占位, 后续挂 FPGA
            msg = self.store.latest()
            if msg:
                for d in msg.get("detections", []):
                    p1, p2 = (d["x1"], d["y1"]), (d["x2"], d["y2"])
                    cv2.rectangle(fr, p1, p2, (0, 255, 0), 2)
                    cv2.putText(fr, f'{d["class_name"]} {d["score"]:.2f}',
                                (d["x1"], max(20, d["y1"] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if edge is not None:
                    fr = cv2.addWeighted(fr, 0.7, edge, 0.3, 0)
            fps_n += 1
            if fps_n >= 30:
                fps_val = fps_n / (time.monotonic() - fps_t0)
                fps_t0, fps_n = time.monotonic(), 0
            cv2.putText(fr, f"FPS {fps_val:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            self.hdmi_out.writeframe(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--mode", choices=["print", "overlay"], default="print",
                    help="print=只打印统计; overlay=接 HDMI 叠加输出")
    args = ap.parse_args()

    store = DetectionStore()
    stats = Stats()
    ReceiverThread(args.port, store, stats).start()

    if args.mode == "print":
        print("[mode] print only, Ctrl-C 退出")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        OverlayLoop(store).run()


if __name__ == "__main__":
    main()
