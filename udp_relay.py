#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电脑端 UDP 转发脚本(运行在你的 PC 上)

背景: RK3588 和 PYNQ 各自直连电脑, 两块板子之间没有直接链路,
      检测结果需要经过电脑转发到 PYNQ。

用法:
  python udp_relay.py --listen-port 5000 --pynq 192.168.2.99 --pynq-port 5000

然后 RK3588 端把 --target-ip 指向电脑(而不是 PYNQ):
  python3 rk3588_sender.py --target-ip <电脑在RK3588那侧的IP> --target-port 5000

依赖: 纯标准库, 无需安装任何东西, Windows/Linux 通用。
"""
import argparse
import socket
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-ip", default="0.0.0.0", help="监听地址")
    ap.add_argument("--listen-port", type=int, default=5000)
    ap.add_argument("--pynq", default="192.168.2.99", help="PYNQ 的 IP")
    ap.add_argument("--pynq-port", type=int, default=5000)
    args = ap.parse_args()

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((args.listen_ip, args.listen_port))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (args.pynq, args.pynq_port)
    print(f"[relay] UDP :{args.listen_port} -> {args.pynq}:{args.pynq_port}")

    n, t0 = 0, time.monotonic()
    while True:
        data, src = rx.recvfrom(65535)
        tx.sendto(data, dst)
        n += 1
        if n % 300 == 0:
            span = time.monotonic() - t0
            print(f"[relay] 已转发 {n} 包 ({n/span:.1f} pps), 最近来自 {src}")


if __name__ == "__main__":
    main()
