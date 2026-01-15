import time
import threading
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response

# ---------------------------
# Labels (A-Z + 0-9)
# ---------------------------
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# ---------------------------
# TFLite Interpreter loader (CPU or EdgeTPU)
# ---------------------------
def build_interpreter(model_path_cpu, model_path_tpu=None, use_edgetpu=False):
    """
    Coral typically has tflite_runtime. If not, this will fallback to tensorflow on PC.
    """
    try:
        import tflite_runtime.interpreter as tflite
        if use_edgetpu:
            from tflite_runtime.interpreter import load_delegate
            return tflite.Interpreter(
                model_path=model_path_tpu,
                experimental_delegates=[load_delegate("libedgetpu.so.1")]
            )
        return tflite.Interpreter(model_path=model_path_cpu)
    except Exception:
        import tensorflow as tf
        if use_edgetpu:
            raise RuntimeError("EdgeTPU requested but tflite_runtime delegate not available.")
        return tf.lite.Interpreter(model_path=model_path_cpu)

def softmax(x):
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

# ---------------------------
# Segmentation + features (48-D)
# ---------------------------
def segment_hand(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([20, 170, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 170, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def largest_contour(mask, min_area=3000):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    return cnt

def extract_vec48(cnt, frame_w, frame_h):
    x, y, w, h = cv2.boundingRect(cnt)
    x2, y2 = x + w, y + h

    bx = x / frame_w
    by = y / frame_h
    bw = w / frame_w
    bh = h / frame_h

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        cx, cy = x + w / 2, y + h / 2
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

    ncx = (cx - x) / max(w, 1)
    ncy = (cy - y) / max(h, 1)

    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) if hull is not None else 1.0
    solidity = float(area) / float(hull_area + 1e-6)
    extent = float(area) / float((w * h) + 1e-6)
    aspect = float(w) / float(h + 1e-6)
    circularity = (4 * np.pi * area) / (peri * peri + 1e-6)

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    tips = []
    if hull_idx is not None and len(hull_idx) >= 3:
        defects = cv2.convexityDefects(cnt, hull_idx)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = cnt[s][0]
                end = cnt[e][0]
                depth = (d / 256.0)
                for p in (start, end):
                    px, py = p[0], p[1]
                    if py < cy:
                        tips.append((px, py, depth))

    tips_xy = []
    for (px, py, depth) in sorted(tips, key=lambda t: (t[1], -t[2])):
        if all((px - qx) ** 2 + (py - qy) ** 2 > 25 ** 2 for (qx, qy) in tips_xy):
            tips_xy.append((px, py))
        if len(tips_xy) >= 5:
            break

    tip_xy = []
    tip_dist = []
    for i in range(5):
        if i < len(tips_xy):
            tx, ty = tips_xy[i]
            tip_xy.extend([(tx - x) / max(w, 1), (ty - y) / max(h, 1)])
            dist = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2) / (np.sqrt(w * w + h * h) + 1e-6)
            tip_dist.append(dist)
        else:
            tip_xy.extend([-1.0, -1.0])
            tip_dist.append(-1.0)

    hu = cv2.HuMoments(M).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    hull_peri = cv2.arcLength(hull, True) if hull is not None else 0.0
    hull_peri_norm = hull_peri / (frame_w + frame_h + 1e-6)
    area_norm = area / (frame_w * frame_h + 1e-6)
    peri_norm = peri / (frame_w + frame_h + 1e-6)

    tip_count_norm = float(len(tips_xy)) / 5.0
    tip_y_mean = (np.mean([p[1] for p in tips_xy]) / frame_h) if tips_xy else 0.0
    tip_y_min = (min([p[1] for p in tips_xy]) / frame_h) if tips_xy else 0.0
    tip_y_max = (max([p[1] for p in tips_xy]) / frame_h) if tips_xy else 0.0
    tip_x_mean = (np.mean([p[0] for p in tips_xy]) / frame_w) if tips_xy else 0.0

    bx2 = x2 / frame_w
    by2 = y2 / frame_h

    vec = [
        bx, by, bw, bh,
        ncx, ncy,
        *tip_xy,
        *tip_dist,
        area_norm, peri_norm, solidity, extent, aspect, circularity,
        *hu.tolist(),
        hull_peri_norm, tip_count_norm, tip_x_mean, tip_y_mean, tip_y_min, tip_y_max,
        bx2, by2,
        float(w) / frame_w, float(h) / frame_h,
        float(x2 - x) / max(frame_w, 1), float(y2 - y) / max(frame_h, 1),
        float(cx) / frame_w, float(cy) / frame_h
    ]

    vec = np.array(vec, dtype=np.float32)
    if vec.shape[0] != 48:
        vec = vec[:48] if vec.shape[0] > 48 else np.pad(vec, (0, 48 - vec.shape[0]), constant_values=0)
    return vec, (x, y, w, h), tips_xy, (int(cx), int(cy))

def guidance_from_bbox(x, y, w, h, W, H):
    cx = x + w / 2.0
    cy = y + h / 2.0
    dx = (cx - W / 2.0) / W
    dy = (cy - H / 2.0) / H
    size = (w * h) / float(W * H + 1e-9)

    center_thresh = 0.08
    size_min = 0.05
    size_max = 0.25

    moves = []
    if dx < -center_thresh: moves.append("MOVE LEFT")
    elif dx > center_thresh: moves.append("MOVE RIGHT")
    if dy < -center_thresh: moves.append("MOVE UP")
    elif dy > center_thresh: moves.append("MOVE DOWN")
    if size < size_min: moves.append("MOVE CLOSER")
    elif size > size_max: moves.append("MOVE FARTHER")

    ok = (abs(dx) <= center_thresh and abs(dy) <= center_thresh and size_min <= size <= size_max)
    return dx, dy, size, moves, ok

# ---------------------------
# MJPEG server state
# ---------------------------
app = Flask(__name__)
latest_jpeg = None
lock = threading.Lock()

def capture_loop(dev=0, width=640, height=480, fps=30, flip=0,
                 model_cpu="models/mlp48_int8.tflite",
                 model_tpu="models/mlp48_int8_edgetpu.tflite",
                 use_edgetpu=0,
                 min_area=3000,
                 infer_every=1,
                 conf_thresh=0.0):
    global latest_jpeg

    # Camera
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {dev}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))

    print("[CAM] Negotiated:",
          cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
          cap.get(cv2.CAP_PROP_FPS))

    # Model
    interpreter = build_interpreter(model_cpu, model_tpu, bool(use_edgetpu))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_dtype = inp["dtype"]
    out_dtype = out["dtype"]
    in_scale, in_zp = inp.get("quantization", (0.0, 0))
    out_scale, out_zp = out.get("quantization", (0.0, 0))

    def quantize_input(x_float32):
        if in_dtype in (np.int8, np.uint8):
            if in_scale == 0:
                raise RuntimeError("Input quantization scale is 0 (invalid model metadata).")
            return np.round(x_float32  / in_scale + in_zp).astype(in_dtype)
        return x_float32.astype(in_dtype)

    def dequantize_output(y):
        if out_dtype in (np.int8, np.uint8):
            return (y.astype(np.float32) - out_zp) * out_scale
        return y.astype(np.float32)

    print("[MODEL] input dtype:", in_dtype, "quant:", inp.get("quantization"))
    print("[MODEL] output dtype:", out_dtype, "quant:", out.get("quantization"))
    print("[MODEL] output shape:", out.get("shape"))

    pred_hist = deque(maxlen=7)
    last_label = "?"
    last_conf = 0.0

    fps_hist = deque(maxlen=20)
    t_prev = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        if flip:
            frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]
        frame_idx += 1

        # Crosshair center
        cv2.line(frame, (W // 2 - 20, H // 2), (W // 2 + 20, H // 2), (255, 255, 255), 2)
        cv2.line(frame, (W // 2, H // 2 - 20), (W // 2, H // 2 + 20), (255, 255, 255), 2)

        mask = segment_hand(frame)
        cnt = largest_contour(mask, min_area=min_area)

        guide_msg = "No hand"
        pred_text = "Pred: (no hand)"

        if cnt is not None:
            vec48, (x, y, w, h), tips_xy, (cx, cy) = extract_vec48(cnt, W, H)
            dx, dy, size, moves, okpos = guidance_from_bbox(x, y, w, h, W, H)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            for (tx, ty) in tips_xy:
                cv2.circle(frame, (tx, ty), 6, (0, 255, 0), -1)

            if okpos:
                guide_msg = "GOOD POSITION - HOLD STILL"
            else:
                guide_msg = " | ".join(moves) if moves else "HOLD STILL"

            cv2.putText(frame, f"dx={dx:+.2f} dy={dy:+.2f} size={size:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if frame_idx % max(1, infer_every) == 0:
                x_in = vec48.reshape(1, 48).astype(np.float32)
                x_in = quantize_input(x_in)

                interpreter.set_tensor(inp["index"], x_in)
                interpreter.invoke()
                y_out = interpreter.get_tensor(out["index"])
                y_f = dequantize_output(y_out)[0]

                prob = softmax(y_f)
                idx = int(np.argmax(prob))
                conf = float(prob[idx])

                pred_hist.append(idx)
                maj = max(set(pred_hist), key=list(pred_hist).count)

                if 0 <= maj < len(LABELS):
                    last_label = LABELS[maj]
                else:
                    last_label = f"IDX{maj}"
                last_conf = conf

            if last_conf >= conf_thresh:
                pred_text = f"Pred: {last_label} ({last_conf*100:.1f}%)"
            else:
                pred_text = f"Pred: (low conf {last_conf*100:.1f}%)"

        # FPS estimate
        now = time.time()
        fps_now = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        fps_hist.append(fps_now)
        fps_avg = sum(fps_hist) / len(fps_hist)

        cv2.putText(frame, guide_msg, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, pred_text, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok2:
            with lock:
                latest_jpeg = jpg.tobytes()

def gen():
    while True:
        with lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return """
    <html>
      <head><title>Coral Camera + Prediction</title></head>
      <body style">
        <h3>Coral Live View (Guidance + Prediction Overlay)</h3>
        <img src="/video" width="800" />
      </body>
    </html>
    """

@app.route("/video")
def video():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dev", type=int, default=0)
    p.add_argument("--w", type=int, default=640)
    p.add_argument("--h", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--flip", type=int, default=0)
    p.add_argument("--port", type=int, default=5000)

    p.add_argument("--model", type=str, default="models/mlp48_int8.tflite")
    p.add_argument("--model_tpu", type=str, default="models/mlp48_int8_edgetpu.tflite")
    p.add_argument("--edgetpu", type=int, default=0)

    p.add_argument("--min_area", type=int, default=3000)
    p.add_argument("--infer_every", type=int, default=1)
    p.add_argument("--conf_thresh", type=float, default=0.0)

    args = p.parse_args()

    th = threading.Thread(
        target=capture_loop,
        args=(args.dev, args.w, args.h, args.fps, args.flip,
              args.model, args.model_tpu, args.edgetpu,
              args.min_area, args.infer_every, args.conf_thresh),
        daemon=True
    )
    th.start()

    print(f"[HTTP] Open http://<CORAL_IP>:{args.port}/ on your laptop")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
