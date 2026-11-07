# PYNQ-Z2 Cat/Dog Vision FPGA Project

This repository contains the FPGA acceleration work for a PYNQ-Z2 vision system.

## Project structure

- `pynq_conv3x3_hls/` — HLS source, testbench, and Vivado packaging scripts for the `conv3x3` accelerator.
- `vision_accel/` — generated PYNQ overlay artifacts (`vision_accel.bit` and `vision_accel.hwh`).
- `merge_pynq_z2_fixed.tcl` — Tcl script used to integrate the accelerator into the PYNQ-Z2 design.
- `outputs/` — exported test results and reports when available.

The PYNQ 2.3 source tree is intentionally excluded because it is a third-party dependency. The board uses PYNQ 2.3 on ARMv7/Linux.

## Verified results

- `conv3x3` hardware output matches the NumPy reference with maximum absolute error 0.
- Hardware average latency: about 5.58 ms.
- Hardware throughput: about 179 FPS for the tested tensor size.
- CPU reference throughput: about 2.18 FPS.
- Measured acceleration: about 82x.

## Runtime pipeline

1. RK3588 runs the YOLOv8 INT8 RKNN model.
2. RK3588 sends one JSON detection result per frame over UDP to PYNQ port 5000.
3. PYNQ receives the detection boxes, processes HDMI frames, and overlays labels and boxes.
4. The PYNQ HDMI input/output path uses the PYNQ-Z2 base video overlay.

## UDP message format

```json
{
  "frame_id": 3,
  "width": 1280,
  "height": 720,
  "timestamp_ms": 0,
  "detections": [
    {
      "class_id": 1,
      "class_name": "dog",
      "score": 0.911,
      "x1": 416,
      "y1": 45,
      "x2": 840,
      "y2": 714
    }
  ]
}
```

Coordinates use the upper-left origin and are already expressed in the declared `width` × `height` coordinate space.

## PYNQ runtime notes

The PYNQ-Z2 CMA allocator is limited. Reuse one HDMI output frame buffer in long-running loops instead of calling `hdmi_out.newframe()` on every frame:

```python
out_frame = hdmi_out.newframe()
for _ in range(300):
    frame = hdmi_in.readframe()
    display = np.array(frame).copy()
    result = receive_latest_result()
    display = draw_result(display, result)
    out_frame[:] = display
    hdmi_out.writeframe(out_frame)
```

