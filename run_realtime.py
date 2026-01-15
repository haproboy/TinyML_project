import cv2, numpy as np, time
from collections import deque

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# =========================
# CONFIG
# =========================
USE_EDGETPU = False   # Laptop/PC: False | Coral: True

MODEL_CPU = "models/mlp48_int8.tflite"
MODEL_TPU = "models/mlp48_int8_edgetpu.tflite"

# =========================
# LOAD TFLITE (CPU or TPU)
# =========================
if USE_EDGETPU:
    import tflite_runtime.interpreter as tflite
    from tflite_runtime.interpreter import load_delegate
    interpreter = tflite.Interpreter(
        model_path=MODEL_TPU,
        experimental_delegates=[load_delegate("libedgetpu.so.1")]
    )
else:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_CPU)

interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

# quant params (INT8 model)
in_scale, in_zp = inp["quantization"]
out_scale, out_zp = out["quantization"]

def quantize(x_float32):
    return np.round(x_float32 / in_scale + in_zp).astype(np.int8)

def dequantize_logits(y_int8):
    return (y_int8.astype(np.float32) - out_zp) * out_scale

# =========================
# FEATURES (same as capture)
# =========================
def segment_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 170, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower, upper)

    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 170, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 3000:
        return None
    return cnt

def extract_landmark_vector(cnt, frame_w, frame_h):
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
        float(w) / frame_w, float(h) / frame_h, float(x2 - x) / max(frame_w, 1), float(y2 - y) / max(frame_h, 1),
        float(cx) / frame_w, float(cy) / frame_h
    ]
    vec = np.array(vec, dtype=np.float32)
    if vec.shape[0] != 48:
        vec = vec[:48] if vec.shape[0] > 48 else np.pad(vec, (0, 48 - vec.shape[0]), constant_values=0)
    return vec, (x, y, w, h), tips_xy, (int(cx), int(cy))

# =========================
# RUN REALTIME
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pred_hist = deque(maxlen=7)
fps_hist = deque(maxlen=20)
t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]
    mask = segment_hand(frame)
    cnt = largest_contour(mask)

    pred_text = "No hand"
    conf = 0.0

    if cnt is not None:
        vec, (x, y, w, h), tips_xy, (cx, cy) = extract_landmark_vector(cnt, W, H)

        x_in = vec.reshape(1, 48).astype(np.float32)
        x_q = quantize(x_in)

        interpreter.set_tensor(inp["index"], x_q)
        interpreter.invoke()
        y_q = interpreter.get_tensor(out["index"])  # int8 logits
        y_f = dequantize_logits(y_q)[0]

        ex = np.exp(y_f - np.max(y_f))
        prob = ex / (np.sum(ex) + 1e-9)

        idx = int(np.argmax(prob))
        conf = float(prob[idx])
        pred_hist.append(idx)

        maj = max(set(pred_hist), key=list(pred_hist).count)

        # guard: never crash
        if 0 <= maj < len(LABELS):
            pred_text = LABELS[maj]
        else:
            pred_text = "?"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        for (tx, ty) in tips_xy:
            cv2.circle(frame, (tx, ty), 6, (0, 255, 0), -1)

        cv2.putText(frame, f"Pred: {pred_text} ({conf*100:.1f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        cv2.putText(frame, pred_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    now = time.time()
    fps = 1.0 / max(now - t0, 1e-6)
    t0 = now
    fps_hist.append(fps)
    fps_avg = sum(fps_hist) / len(fps_hist)
    cv2.putText(frame, f"FPS: {fps_avg:.1f}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("A-Z + 0-9 Edge Demo", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
