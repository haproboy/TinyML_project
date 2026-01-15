import os, cv2, numpy as np, time

SAVE_DIR = "data_vec"
os.makedirs(SAVE_DIR, exist_ok=True)

# 36 labels: A-Z + 0-9
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# start at '0' so you only collect digits now
label_idx = 26
recording = False

# ====== Hand segmentation (HSV skin-ish) ======
def segment_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 170, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower, upper)

    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 170, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    kernel = np.ones((5,5), np.uint8)
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
    x,y,w,h = cv2.boundingRect(cnt)
    x2,y2 = x+w, y+h

    bx = x / frame_w
    by = y / frame_h
    bw = w / frame_w
    bh = h / frame_h

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        cx, cy = x + w/2, y + h/2
    else:
        cx = M["m10"]/M["m00"]
        cy = M["m01"]/M["m00"]

    ncx = (cx - x) / max(w, 1)
    ncy = (cy - y) / max(h, 1)

    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) if hull is not None else 1.0
    solidity = float(area) / float(hull_area + 1e-6)
    extent = float(area) / float((w*h) + 1e-6)
    aspect = float(w) / float(h + 1e-6)
    circularity = (4*np.pi*area) / (peri*peri + 1e-6)

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    tips = []
    if hull_idx is not None and len(hull_idx) >= 3:
        defects = cv2.convexityDefects(cnt, hull_idx)
        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = cnt[s][0]
                end   = cnt[e][0]
                depth = (d / 256.0)
                for p in (start, end):
                    px, py = p[0], p[1]
                    if py < cy:
                        tips.append((px, py, depth))

    # dedup tips (max 5)
    tips_xy = []
    for (px,py,depth) in sorted(tips, key=lambda t: (t[1], -t[2])):
        if all((px-qx)**2 + (py-qy)**2 > 25**2 for (qx,qy) in tips_xy):
            tips_xy.append((px,py))
        if len(tips_xy) >= 5:
            break

    tip_xy = []
    tip_dist = []
    for i in range(5):
        if i < len(tips_xy):
            tx, ty = tips_xy[i]
            tip_xy.extend([(tx - x) / max(w, 1), (ty - y) / max(h, 1)])
            dist = np.sqrt((tx-cx)**2 + (ty-cy)**2) / (np.sqrt(w*w+h*h) + 1e-6)
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

    tip_count_norm = float(len(tips_xy))/5.0
    tip_y_mean = (np.mean([p[1] for p in tips_xy]) / frame_h) if tips_xy else 0.0
    tip_y_min  = (min([p[1] for p in tips_xy]) / frame_h) if tips_xy else 0.0
    tip_y_max  = (max([p[1] for p in tips_xy]) / frame_h) if tips_xy else 0.0
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
        float(w)/frame_w, float(h)/frame_h, float(x2-x)/max(frame_w,1), float(y2-y)/max(frame_h,1),
        float(cx)/frame_w, float(cy)/frame_h
    ]
    vec = np.array(vec, dtype=np.float32)
    if vec.shape[0] != 48:
        vec = vec[:48] if vec.shape[0] > 48 else np.pad(vec, (0,48-vec.shape[0]), constant_values=0)
    return vec, (x,y,w,h), tips_xy, (int(cx), int(cy))

def ensure_label_dir(label):
    d = os.path.join(SAVE_DIR, label)
    os.makedirs(d, exist_ok=True)
    return d

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Keys: [N]=next label, [R]=toggle record, [Q]=quit")
print("Start label:", LABELS[label_idx])

last_save = 0
save_interval = 0.15  # ~6-7 samples/sec

while True:
    ok, frame = cap.read()
    if not ok:
        break
    H, W = frame.shape[:2]

    mask = segment_hand(frame)
    cnt = largest_contour(mask)

    label = LABELS[label_idx]
    status = f"Label: {label} | REC: {recording}"

    if cnt is not None:
        vec, (x,y,w,h), tips, (cx,cy) = extract_landmark_vector(cnt, W, H)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)
        for (tx,ty) in tips:
            cv2.circle(frame, (tx,ty), 6, (0,255,0), -1)

        if recording and (time.time() - last_save) > save_interval:
            d = ensure_label_dir(label)
            fname = f"{int(time.time()*1000)}.npy"
            np.save(os.path.join(d, fname), vec)
            last_save = time.time()
            cv2.putText(frame, "SAVED", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, status, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, "N-next | R-record | Q-quit", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Capture A-Z + 0-9 (Contour landmarks)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        label_idx = (label_idx + 1) % len(LABELS)
    elif key == ord('r'):
        recording = not recording

cap.release()
cv2.destroyAllWindows()
