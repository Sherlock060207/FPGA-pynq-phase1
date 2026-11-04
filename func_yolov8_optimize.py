# YOLOv8 检测后处理 — 精简版
# 只检测宠物（猫/狗），画框 + 串口极坐标输出
# 去掉 ClassSmoother / PetRegistry / ORB 等复杂机制
import cv2
import numpy as np
from dataclasses import dataclass

OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.45, 0.45, 640
CLASSES = ("cat", "dog")

# ======================== 后处理 ========================

def yolov8_post_process(outputs):
    """YOLOv8 post-process.

    Supported RKNN export layouts:
      1) single output: (1, 6, 8400) or (1, 84, 8400)
      2) decoupled DFL outputs: [reg, cls, obj] x 3 scales, e.g.
         reg=(1,64,80,80), cls=(1,80,80,80), obj=(1,1,80,80), ...

    Returns boxes in letterbox coordinates and class ids remapped to:
      0 = cat, 1 = dog
    """
    if outputs is None or len(outputs) == 0:
        return None, None, None

    # Newer RKNN YOLOv8 DFL export: 9 outputs = [reg, cls, obj] x 3.
    if len(outputs) >= 9 and np.asarray(outputs[0]).ndim == 4:
        return _yolov8_post_process_dfl(outputs)

    # Legacy single-output export.
    data = np.asarray(outputs[0])
    if data.ndim == 3:
        # Accept both (1, C, N) and (1, N, C).
        if data.shape[1] < data.shape[2]:
            data = data.transpose(0, 2, 1)[0]
        else:
            data = data[0]
    elif data.ndim == 2:
        pass
    else:
        raise ValueError(f"unsupported YOLO output shape: {data.shape}")

    if data.shape[1] < 6:
        raise ValueError(f"unsupported YOLO output channels: {data.shape}")

    cls_scores = data[:, 4:]
    class_ids_raw = np.argmax(cls_scores, axis=1)
    max_scores = cls_scores[np.arange(len(cls_scores)), class_ids_raw]

    # If this is COCO-80, keep only cat/dog and remap 15->0, 16->1.
    if cls_scores.shape[1] >= 80:
        cat_scores = cls_scores[:, 15]
        dog_scores = cls_scores[:, 16]
        is_dog = dog_scores > cat_scores
        max_scores = np.where(is_dog, dog_scores, cat_scores)
        class_ids = np.where(is_dog, 1, 0)
    else:
        class_ids = class_ids_raw

    mask = max_scores > OBJ_THRESH
    data = data[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(data) == 0:
        return None, None, None

    cx, cy, w, h = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms_boxes_fn(boxes, max_scores)
    return boxes[keep], max_scores[keep], class_ids[keep]


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _dist2bbox(distance, anchor_points, stride):
    x1 = anchor_points[:, 0] - distance[:, 0]
    y1 = anchor_points[:, 1] - distance[:, 1]
    x2 = anchor_points[:, 0] + distance[:, 2]
    y2 = anchor_points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1) * stride


def _decode_dfl(reg, stride):
    # reg: (1, 64, H, W), reg_max=16 -> distances l,t,r,b.
    reg = np.asarray(reg)[0]
    c, h, w = reg.shape
    reg_max = c // 4
    reg = reg.reshape(4, reg_max, h, w).transpose(2, 3, 0, 1)  # H,W,4,reg_max
    prob = _softmax(reg, axis=-1)
    proj = np.arange(reg_max, dtype=np.float32)
    dist = np.sum(prob * proj, axis=-1).reshape(-1, 4)

    gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    anchors = np.stack([gx.reshape(-1) + 0.5, gy.reshape(-1) + 0.5], axis=1).astype(np.float32)
    boxes = _dist2bbox(dist, anchors, stride)
    return boxes


def _yolov8_post_process_dfl(outputs):
    all_boxes = []
    all_scores = []
    all_cls = []

    # Infer strides from feature map sizes for 640 input: 80->8, 40->16, 20->32.
    for i in range(0, min(len(outputs), 9), 3):
        reg = np.asarray(outputs[i])
        cls = np.asarray(outputs[i + 1])
        obj = np.asarray(outputs[i + 2]) if i + 2 < len(outputs) else None
        if reg.ndim != 4 or cls.ndim != 4:
            continue
        _, _, h, w = reg.shape
        stride = float(IMG_SIZE) / float(h)
        boxes = _decode_dfl(reg, stride)

        cls_data = cls[0].transpose(1, 2, 0).reshape(-1, cls.shape[1])
        if obj is not None and obj.ndim == 4 and obj.shape[1] == 1:
            obj_data = obj[0, 0].reshape(-1)
        else:
            obj_data = np.ones((cls_data.shape[0],), dtype=np.float32)

        # Custom 2-class model: cat=0, dog=1. COCO-80 model: cat=15, dog=16.
        if cls_data.shape[1] >= 80:
            cat_scores = cls_data[:, 15] * obj_data
            dog_scores = cls_data[:, 16] * obj_data
            is_dog = dog_scores > cat_scores
            scores = np.where(is_dog, dog_scores, cat_scores)
            class_ids = np.where(is_dog, 1, 0)
        elif cls_data.shape[1] >= 2:
            raw_ids = np.argmax(cls_data[:, :2], axis=1)
            scores = cls_data[np.arange(cls_data.shape[0]), raw_ids] * obj_data
            class_ids = raw_ids
        else:
            scores = cls_data[:, 0] * obj_data
            class_ids = np.zeros_like(scores, dtype=np.int64)

        mask = scores > OBJ_THRESH
        if np.any(mask):
            all_boxes.append(boxes[mask])
            all_scores.append(scores[mask])
            all_cls.append(class_ids[mask])

    if not all_boxes:
        return None, None, None

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_cls, axis=0).astype(np.int64)

    keep = nms_boxes_fn(boxes, scores)
    return boxes[keep], scores[keep], class_ids[keep]

def nms_boxes_fn(boxes, scores):
    """标准 NMS，输入 xyxy 格式"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    return np.array(keep)


# ======================== 绘图 ========================

def draw_box_corner(draw_img, top, left, right, bottom, length, corner_color):
    cv2.line(draw_img, (top, left), (top + length, left), corner_color, 3)
    cv2.line(draw_img, (top, left), (top, left + length), corner_color, 3)
    cv2.line(draw_img, (right, left), (right - length, left), corner_color, 3)
    cv2.line(draw_img, (right, left), (right, left + length), corner_color, 3)
    cv2.line(draw_img, (top, bottom), (top + length, bottom), corner_color, 3)
    cv2.line(draw_img, (top, bottom), (top, bottom - length), corner_color, 3)
    cv2.line(draw_img, (right, bottom), (right - length, bottom), corner_color, 3)
    cv2.line(draw_img, (right, bottom), (right, bottom - length), corner_color, 3)


def draw(image, boxes, scores, class_ids, ratio, padding):
    """绘图：画框 + 角标 + 类别标签（cat/dog）"""
    if boxes is None:
        return
    for box, score, cls_id in zip(boxes, scores, class_ids):
        top = int((box[0] - padding[0]) / ratio[0])
        left = int((box[1] - padding[1]) / ratio[1])
        right = int((box[2] - padding[0]) / ratio[0])
        bottom = int((box[3] - padding[1]) / ratio[1])

        cv2.rectangle(image, (top, left), (right, bottom), (0, 255, 0), 2)
        draw_box_corner(image, top, left, right, bottom, 15, (0, 255, 0))

        # 类别标签
        label = CLASSES[cls_id] if cls_id < len(CLASSES) else "?"
        cv2.putText(image, f"{label} {score:.2f}", (top, left - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# ======================== 预处理 ========================

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (left, top)


# ======================== 极坐标计算 ========================

# 摄像头视场角（根据实际镜头调整）
CAMERA_FOV_H = 60.0   # 水平 FOV（度）

@dataclass(frozen=True)
class VisionInferenceResult:
    """单帧 RKNN 推理结果。

    frame_seq 与 capture_ts 由 rknnPoolExecutor 在摄像头采集时附加，
    不能在 worker 中用全局变量生成，否则多 worker 完成顺序不同会错帧。
    """
    image: object
    detection: dict


def calc_polar(box, ratio, padding, orig_w, orig_h):
    """
    从检测框计算极坐标和框大小。
    box: [x1, y1, x2, y2] letterbox 坐标系
    返回: dict(theta, r_px, cx, cy, box_w, box_h)
    """
    # 还原到原图坐标
    x1_orig = (box[0] - padding[0]) / ratio[0]
    y1_orig = (box[1] - padding[1]) / ratio[1]
    x2_orig = (box[2] - padding[0]) / ratio[0]
    y2_orig = (box[3] - padding[1]) / ratio[1]

    cx = (x1_orig + x2_orig) / 2.0
    cy = (y1_orig + y2_orig) / 2.0
    bw = x2_orig - x1_orig
    bh = y2_orig - y1_orig

    # 画面中心
    center_x = orig_w / 2.0
    center_y = orig_h / 2.0

    # 偏移
    dx = cx - center_x
    dy = cy - center_y
    r_px = np.sqrt(dx * dx + dy * dy)

    # 水平偏转角（基于 FOV 换算）
    theta_deg = (dx / (orig_w / 2.0)) * (CAMERA_FOV_H / 2.0)

    return {
        'theta': theta_deg,
        'r_px': r_px,
        'cx': cx,
        'cy': cy,
        'box_w': bw,
        'box_h': bh,
    }


# ======================== 主入口 ========================

def myFunc(rknn_lite, IMG):
    """推理 + 画框，返回与当前图像绑定的检测结果。"""
    orig_h, orig_w = IMG.shape[:2]

    IMG2 = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    IMG2, ratio, padding = letterbox(IMG2)
    IMG2 = np.expand_dims(IMG2, 0)

    outputs = rknn_lite.inference(inputs=[IMG2], data_format=['nhwc'])
    boxes, scores, class_ids = yolov8_post_process(outputs)

    if boxes is not None and len(boxes) > 0:
        # 取置信度最高的一个目标
        best_idx = np.argmax(scores)
        best_box = boxes[best_idx]
        best_score = scores[best_idx]
        best_class_id = int(class_ids[best_idx])
        best_class_name = CLASSES[best_class_id] if best_class_id < len(CLASSES) else "pet"

        # 计算极坐标
        polar = calc_polar(best_box, ratio, padding, orig_w, orig_h)
        detection = {
            'theta': polar['theta'],
            'r_px': polar['r_px'],
            'cx': polar['cx'],
            'cy': polar['cy'],
            'box_w': polar['box_w'],
            'box_h': polar['box_h'],
            'conf': float(best_score),
            'detected': True,
            'class_id': best_class_id,
            'class_name': best_class_name,
        }

        # 画所有检测框（框 + 角标 + 类别标签）
        draw(IMG, boxes, scores, class_ids, ratio, padding)
    else:
        detection = {
            'theta': 0.0, 'r_px': 0.0,
            'cx': 0.0, 'cy': 0.0,
            'box_w': 0.0, 'box_h': 0.0,
            'conf': 0.0, 'detected': False,
            'class_id': -1, 'class_name': '',
        }

    return VisionInferenceResult(image=IMG, detection=detection)
