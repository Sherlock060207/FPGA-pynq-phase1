#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电脑端视频推送脚本(运行在你的 PC 上, Windows/Linux/macOS 均可)

作用: 把要识别的画面(电脑屏幕/视频文件/图片目录/摄像头)编码成 JPEG,
      通过 TCP 推给 RK3588 (rk3588_sender.py --source tcp:// 源)。
      同时电脑 HDMI 输出给 PYNQ HDMI 输入, PYNQ 叠加检测框显示。

依赖:
  pip install opencv-python numpy mss

用法:
  # 推电脑主屏幕画面(最常用, 屏幕上拖窗口/播放视频 RK3588 实时识别)
  python pc_video_sender.py --source screen --target <RK3588_IP>

  # 推指定显示器(多显示器环境下)
  python pc_video_sender.py --source screen:1 --target <RK3588_IP>

  # 推视频文件/图片目录/摄像头
  python pc_video_sender.py --source cat_dog.mp4 --target <RK3588_IP>
  python pc_video_sender.py --source 0 --target <RK3588_IP>

  target 默认 192.168.1.33(RK3588 WiFi), 端口默认 5006。
  --fps 控制推流帧率, 屏幕源默认 30, 视频文件按原始帧率。
"""
import argparse
import glob
import os
import socket
import struct
import sys
import time

import cv2
import numpy as np


class ScreenSource:
    """电脑屏幕截取: 用 mss 直接抓屏幕 BGR 帧, 比 pyautogui 快 5-10x"""

    def __init__(self, monitor_id=0, fps=30):
        try:
            import mss
        except ImportError:
            print("[ERR] 需要 mss 库: pip install mss")
            sys.exit(1)
        self.sct = mss.mss()
        # monitor_id: 0=全部(拼接), 1/2/3=具体显示器
        self.monitor_id = monitor_id
        mon = self.sct.monitors[monitor_id]
        self.fps = fps
        self.width = mon["width"]
        self.height = mon["height"]
        self._interval = 1.0 / fps
        print(f"[src] screen monitor={monitor_id} {self.width}x{self.height} "
              f"@{fps:.0f}fps")

    def read(self):
        t0 = time.monotonic()
        mon = self.sct.monitors[self.monitor_id]
        shot = self.sct.grab(mon)
        # mss 输出 BGRA, 去掉 A 通道
        fr = np.array(shot)[:, :, :3]
        dt = time.monotonic() - t0
        if dt < self._interval:
            time.sleep(self._interval - dt)
        return True, fr

    def list_monitors(self):
        print("[src] 可用显示器列表:")
        for i, m in enumerate(self.sct.monitors):
            print(f"  monitor {i}: {m['width']}x{m['height']} "
                  f"(top={m['top']} left={m['left']})")


def open_source(src, width, height, fps_default):
    # ---- 屏幕源(推荐) ----
    if src.startswith("screen"):
        mid = 0
        if ":" in src:
            try:
                mid = int(src.split(":", 1)[1])
            except ValueError:
                pass
        sc = ScreenSource(mid, fps_default)
        return ("screen", sc)

    # ---- 图片目录 ----
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.jpg")) +
                       glob.glob(os.path.join(src, "*.png")) +
                       glob.glob(os.path.join(src, "*.jpeg")))
        if not files:
            raise RuntimeError(f"目录里没有图片: {src}")
        print(f"[src] 图片目录 {src}, 共 {len(files)} 张")
        return ("images", files)

    # ---- 视频文件 / 摄像头 ----
    if os.path.isfile(src):
        cap = cv2.VideoCapture(src)
        is_file = True
    else:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        is_file = False
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频源: {src}")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[src] {src} 已打开, "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, fps={fps:.1f}")
    return ("video" if is_file else "camera", cap)


def iter_frames(src_desc, default_fps):
    kind, obj = src_desc
    if kind == "screen":
        obj.list_monitors()  # 打印可用显示器
        while True:
            ok, fr = obj.read()
            if ok:
                yield fr, obj.fps
    elif kind == "images":
        imgs = []
        for f in obj:
            im = cv2.imread(f)
            if im is not None:
                imgs.append(im)
        while True:
            for im in imgs:
                yield im, default_fps
    else:
        fps = obj.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1 or fps > 240:
            fps = default_fps
        while True:
            ok, fr = obj.read()
            if not ok:
                if kind == "video":
                    obj.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue
            yield fr, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="screen / screen:1 / 视频文件 / 图片目录 / 摄像头编号 0")
    ap.add_argument("--target", default="192.168.1.33",
                    help="RK3588 的 IP(WiFi 或 eth0 均可)")
    ap.add_argument("--port", type=int, default=5006)
    ap.add_argument("--fps", type=float, default=30.0,
                    help="屏幕/图片源的推送帧率")
    ap.add_argument("--width", type=int, default=0, help="0=源分辨率")
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--quality", type=int, default=80, help="JPEG 质量 1-100")
    args = ap.parse_args()

    src_desc = open_source(args.source, args.width, args.height, args.fps)

    sock = None
    while True:  # 断线自动重连
        try:
            if sock is None:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect((args.target, args.port))
                print(f"[net] 已连接 RK3588 {args.target}:{args.port}")

            fid = 0
            t_stat = time.monotonic()
            n_stat = 0
            for fr, src_fps in iter_frames(src_desc, args.fps):
                t0 = time.monotonic()
                if args.width and args.height and (
                        fr.shape[1] != args.width or fr.shape[0] != args.height):
                    fr = cv2.resize(fr, (args.width, args.height))
                ok, jpg = cv2.imencode(
                    ".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if not ok:
                    continue
                payload = jpg.tobytes()
                sock.sendall(struct.pack(">I", len(payload)) + payload)
                fid += 1
                n_stat += 1
                if n_stat >= 60:
                    span = time.monotonic() - t_stat
                    print(f"[stat] 已推 {fid} 帧, 发送 {n_stat/span:.1f} fps")
                    t_stat, n_stat = time.monotonic(), 0
                dt = time.monotonic() - t0
                interval = 1.0 / max(src_fps, 1)
                if dt < interval:
                    time.sleep(interval - dt)
        except (ConnectionError, OSError) as e:
            print(f"[net] 连接中断({e}), 3s 后重连...")
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n退出")
            sys.exit(0)


if __name__ == "__main__":
    main()
