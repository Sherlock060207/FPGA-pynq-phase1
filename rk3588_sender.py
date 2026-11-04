#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一阶段 RK3588 发送端:
  视频源 -> RKNN YOLOv8 INT8 推理(cat/dog) -> JSON -> UDP 发送到 PYNQ

用法示例:
  # HDMI RX 采集(电脑 DP/HDMI 输出直连板子), 发往 PYNQ(默认 192.168.2.99:5000)
  python3 rk3588_sender.py --source hdmirx:///dev/video20

  # TCP 收流(电脑 pc_video_sender.py 推送)
  python3 rk3588_sender.py --source tcp://:5006 --target-ip 192.168.0.1

  # 测试图片源(默认), 回环自测
  python3 rk3588_sender.py --target-ip 127.0.0.1

  # 视频文件
  python3 rk3588_sender.py --source /path/test.mp4

  # MIPI/USB 摄像头(驱动修复后)
  python3 rk3588_sender.py --source /dev/video0
"""
import argparse
import glob
import json
import os
import socket
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rknnlite.api import RKNNLite
from func_yolov8_optimize import (CLASSES, IMG_SIZE, letterbox,
                                  yolov8_post_process)


def fit_canvas(bgr, out_w, out_h):
    """等比缩放到画布(黑边填充), 保持宽高比不失真"""
    fit, _, _ = letterbox(bgr, (out_h, out_w))
    return fit


def canvas_transform(w0, h0, out_w, out_h):
    """返回原图->显示画布的缩放系数和偏移(等比缩放+居中黑边)"""
    r = min(out_w / w0, out_h / h0)
    off_x = (out_w - w0 * r) / 2.0
    off_y = (out_h - h0 * r) / 2.0
    return r, off_x, off_y

DEFAULT_MODEL = ("/home/elf/archive_unused_20260628/old_models/"
                 "backup_20260627_pet/best_int8_catdog.rknn")
DEFAULT_SRC = "/home/elf/demo/pet_toy_test_images"


# ---------------- 视频源 ----------------
class HdmiRxSource:
    """HDMI RX 采集源: 电脑 DP/HDMI 输出 -> RK3588 HDMI RX(/dev/video20)

    该设备是 V4L2 multiplanar, OpenCV 读不了, 用 v4l2-ctl 流式输出
    原始 BGR3 帧到管道, 按帧读取。读取速度跟不上时内核自动丢帧。
    """

    def __init__(self, dev="/dev/video20", wait_s=120):
        import subprocess
        w, h = self._wait_signal(dev, wait_s)
        self.width, self.height = w, h
        self.frame_bytes = w * h * 3
        self.fifo = f"/tmp/hdmirx_{os.getpid()}.raw"
        try:
            os.unlink(self.fifo)
        except FileNotFoundError:
            pass
        os.mkfifo(self.fifo)
        cmd = ["v4l2-ctl", "-d", dev,
               f"--set-fmt-video=width={w},height={h},pixelformat=BGR3",
               "--stream-mmap=4", f"--stream-to={self.fifo}"]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.fd = open(self.fifo, "rb", buffering=0)
        print(f"[src] hdmirx {dev} capturing {w}x{h} BGR3")

    @staticmethod
    def _wait_signal(dev, wait_s):
        import subprocess
        t0 = time.time()
        while True:
            out = subprocess.run(
                ["v4l2-ctl", "-d", dev, "--query-dv-timings"],
                capture_output=True, text=True).stdout
            w = h = 0
            for line in out.splitlines():
                if "Active width" in line:
                    w = int(line.split(":")[1])
                if "Active height" in line:
                    h = int(line.split(":")[1])
            if w > 0 and h > 0:
                return w, h
            if time.time() - t0 > wait_s:
                raise RuntimeError(
                    "HDMI RX 无信号, 请检查电脑 DP/HDMI 输出是否已接上")
            print("[src] 等待 HDMI 信号接入(电脑输出)...")
            time.sleep(2)

    def read(self):
        need = self.frame_bytes
        chunks = []
        while need > 0:
            c = self.fd.read(need)
            if not c:
                return False, None
            chunks.append(c)
            need -= len(c)
        fr = np.frombuffer(b"".join(chunks), np.uint8)
        return True, fr.reshape(self.height, self.width, 3)

    def release(self):
        try:
            self.proc.terminate()
            self.fd.close()
            os.unlink(self.fifo)
        except Exception:
            pass


class TcpJpegSource:
    """TCP 收流源: 电脑推送 [4字节大端长度][JPEG数据] 帧序列

    由 pc_video_sender.py 推送, 用于无摄像头阶段联调。
    """

    def __init__(self, listen_port):
        import struct
        self._struct = struct
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", listen_port))
        srv.listen(1)
        print(f"[src] waiting PC video push on tcp://0.0.0.0:{listen_port} ...")
        self.conn, addr = srv.accept()
        self.peer_ip = addr[0]  # 推送端(电脑)的 IP, 供 UDP 回发自动寻址
        print(f"[src] PC connected: {addr}")
        srv.close()
        self.buf = b""

    def _read_exact(self, n):
        while len(self.buf) < n:
            chunk = self.conn.recv(1 << 16)
            if not chunk:
                return None
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def read(self):
        hdr = self._read_exact(4)
        if hdr is None:
            return False, None
        (length,) = self._struct.unpack(">I", hdr)
        payload = self._read_exact(length)
        if payload is None:
            return False, None
        fr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        return (fr is not None), fr

    def release(self):
        try:
            self.conn.close()
        except Exception:
            pass


class ImageDirSource:
    """把测试图片目录循环播放成视频流(保留原分辨率, 保证检测质量)"""

    def __init__(self, path):
        files = sorted(glob.glob(os.path.join(path, "*.jpg")) +
                       glob.glob(os.path.join(path, "*.png")))
        if not files:
            raise RuntimeError(f"no images in {path}")
        self.frames = []
        for f in files:
            im = cv2.imread(f)
            if im is None:
                continue
            self.frames.append(im)
        if not self.frames:
            raise RuntimeError(f"no readable images in {path}")
        self.idx = 0
        h, w = self.frames[0].shape[:2]
        print(f"[src] image-dir {path}: {len(self.frames)} images, native {w}x{h}")

    def read(self):
        fr = self.frames[self.idx]
        self.idx = (self.idx + 1) % len(self.frames)
        return True, fr.copy()

    def release(self):
        pass


def open_source(src):
    if src.startswith("hdmirx://"):
        # hdmirx:///dev/video20 或 hdmirx://video20
        dev = src.split("://", 1)[1] or "/dev/video20"
        if not dev.startswith("/dev/"):
            dev = "/dev/" + dev
        return HdmiRxSource(dev)
    if src.startswith("tcp://"):
        # tcp://:5006 或 tcp://5006
        port = int(src.split("://", 1)[1].lstrip(":") or "5006")
        return TcpJpegSource(port)
    if os.path.isdir(src):
        return ImageDirSource(src)
    if os.path.isfile(src):
        cap = cv2.VideoCapture(src)
        cap._loop_file = True
    else:
        # /dev/videoN 或数字 -> 摄像头
        dev = int(src) if src.isdigit() else src
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap._loop_file = False
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[src] {src} opened, {w}x{h}")
    return cap


def read_frame(cap):
    ok, fr = cap.read()
    if not ok and getattr(cap, "_loop_file", False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = cap.read()
    return ok, fr


# ---------------- 推理 ----------------
class Detector:
    def __init__(self, model_path):
        self.rknn = RKNNLite(verbose=False)
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError("load rknn failed")
        if self.rknn.init_runtime() != 0:
            raise RuntimeError("init runtime failed")
        print(f"[rknn] loaded {model_path}")

    def infer(self, bgr):
        """返回 detections 列表(原图像素坐标)"""
        h0, w0 = bgr.shape[:2]
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img, ratio, padding = letterbox(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, 0)
        outputs = self.rknn.inference(inputs=[img], data_format=["nhwc"])
        boxes, scores, class_ids = yolov8_post_process(outputs)
        dets = []
        if boxes is not None:
            for box, score, cid in zip(boxes, scores, class_ids):
                x1 = int(np.clip((box[0] - padding[0]) / ratio[0], 0, w0 - 1))
                y1 = int(np.clip((box[1] - padding[1]) / ratio[1], 0, h0 - 1))
                x2 = int(np.clip((box[2] - padding[0]) / ratio[0], 0, w0 - 1))
                y2 = int(np.clip((box[3] - padding[1]) / ratio[1], 0, h0 - 1))
                cid = int(cid)
                dets.append({
                    "class_id": cid,
                    "class_name": CLASSES[cid] if cid < len(CLASSES) else "pet",
                    "score": round(float(score), 3),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
        return w0, h0, dets

    def release(self):
        self.rknn.release()


# ---------------- 主循环 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--source", default=DEFAULT_SRC,
                    help="图片目录 / 视频文件 / /dev/videoN")
    ap.add_argument("--target-ip", default="auto",
                    help="检测结果发送目标, auto=回发给 TCP 推流端(电脑)")
    ap.add_argument("--target-port", type=int, default=5000)
    ap.add_argument("--out-width", type=int, default=0,
                    help="JSON 画面尺寸(PYNQ HDMI 分辨率), 0=跟随源帧")
    ap.add_argument("--out-height", type=int, default=0)
    ap.add_argument("--fps-limit", type=float, default=30.0,
                    help="发送限速, 0=不限速")
    ap.add_argument("--max-frames", type=int, default=0, help="0=持续运行")
    ap.add_argument("--stats-every", type=int, default=60)
    args = ap.parse_args()

    det = Detector(args.model)   # 先 NPU 预热, 避免 accept 后 NPU init 崩掉
    cap = open_source(args.source)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_ip = args.target_ip
    if target_ip == "auto":
        target_ip = getattr(cap, "peer_ip", None)
        if not target_ip:
            raise SystemExit(
                "[net] --target-ip auto 需要 TCP 推流源; "
                "其他源请显式指定目标 IP")
    target = (target_ip, args.target_port)
    print(f"[net] UDP -> {target_ip}:{args.target_port}")

    frame_id = 0
    out_w, out_h = args.out_width, args.out_height  # 0=跟随源帧, 首帧解析
    t_run0 = time.monotonic()
    t_stat = t_run0
    n_stat = 0
    infer_ms_acc = 0.0
    min_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0

    try:
        while True:
            t_loop0 = time.monotonic()
            ok, fr = read_frame(cap)
            if not ok:
                print("[src] frame grab failed, stop")
                break
            frame_id += 1

            t0 = time.monotonic()
            w0, h0, dets = det.infer(fr)  # 原始分辨率推理, 原生坐标
            infer_ms_acc += (time.monotonic() - t0) * 1000.0

            if not out_w or not out_h:  # 首帧: 跟随源帧尺寸
                out_w, out_h = w0, h0
                print(f"[out] canvas auto -> {out_w}x{out_h}")

            # 坐标映射到 PYNQ HDMI 显示画布(等比缩放+居中黑边)
            r, off_x, off_y = canvas_transform(w0, h0, out_w, out_h)
            for d in dets:
                d["x1"] = int(d["x1"] * r + off_x)
                d["y1"] = int(d["y1"] * r + off_y)
                d["x2"] = int(d["x2"] * r + off_x)
                d["y2"] = int(d["y2"] * r + off_y)

            msg = {
                "frame_id": frame_id,
                "width": out_w,
                "height": out_h,
                "timestamp_ms": int(time.time() * 1000),  # Unix毫秒, PYNQ端测延迟(需NTP对时)
                "detections": dets,
            }
            sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), target)

            n_stat += 1
            if n_stat >= args.stats_every:
                now = time.monotonic()
                span = now - t_stat
                print(json.dumps({
                    "stat": 1, "frames": n_stat,
                    "fps": round(n_stat / span, 1),
                    "infer_ms_avg": round(infer_ms_acc / n_stat, 1),
                    "last_dets": len(dets),
                }))
                t_stat, n_stat, infer_ms_acc = now, 0, 0.0

            if args.max_frames and frame_id >= args.max_frames:
                break
            dt = time.monotonic() - t_loop0
            if dt < min_interval:
                time.sleep(min_interval - dt)
    except KeyboardInterrupt:
        pass
    finally:
        total = time.monotonic() - t_run0
        print(f"[done] {frame_id} frames in {total:.1f}s "
              f"({frame_id / total:.1f} fps avg)")
        cap.release()
        det.release()


if __name__ == "__main__":
    main()
