#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单/多图 RKNN YOLOv8 cat/dog 检测测试

用法:
  python test_image.py <图片路径>           # 单张
  python test_image.py <目录>              # 目录下所有 jpg/png/webp
  python test_image.py /home/elf/test_imgs
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rk3588_sender import Detector  # noqa

MODEL = "/home/elf/archive_unused_20260628/old_models/backup_20260627_pet/best_int8_catdog.rknn"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_images(path):
    if os.path.isdir(path):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(path, "**", "*" + ext),
                                   recursive=True))
        return sorted(files)
    return [path]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    targets = sys.argv[1:]
    files = []
    for t in targets:
        files.extend(find_images(t))
    files = list(dict.fromkeys(files))  # 去重
    if not files:
        print("[ERR] 没找到可识别的图片")
        sys.exit(1)

    det = Detector(MODEL)
    print(f"\n=== 共 {len(files)} 张图, 开始推理 ===\n")
    results = []
    for fp in files:
        print(f"-- {fp}")
        im = cv2.imread(fp)
        if im is None:
            print("   [跳过] cv2 读失败(可能是 webp 格式不支持)")
            # 试用 Pillow
            try:
                from PIL import Image
                pil = Image.open(fp).convert("RGB")
                im = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"   [跳过] Pillow 也读失败: {e}")
                continue
        print(f"   尺寸 {im.shape[1]}x{im.shape[0]}")
        w, h, dets = det.infer(im)
        if dets:
            for d in dets:
                print(f"   ✅ {d['class_name']} score={d['score']:.3f} "
                      f"box=[{d['x1']},{d['y1']}->{d['x2']},{d['y2']}]")
        else:
            print("   ❌ 未检出 (OBJ_THRESH=0.45)")
        results.append({"file": fp, "detections": dets})
    det.release()

    out = "/home/elf/test_imgs_results.json"
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已存: {out}")


if __name__ == "__main__":
    main()
