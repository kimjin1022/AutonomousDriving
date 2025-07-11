# scripts/step15_render_test_gif.py
# 목적: testing 한 시퀀스만 SORT+CMC(카메라 모션 보정)으로 추적 → GIF로 시각 점검
# 입력: step13에서 만든 det 캐시 (outputs/dets_yolo_test/XXXX.txt)

import os, glob, cv2
from collections import deque
from step7_sort_cmc_minimal import SORT_CMC, estimate_h  # Step7 재사용

# ===== 손댈만한 기본값 =====
TEST_ROOT = "/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/testing"
DET_DIR   = "/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/dets_yolo_test"
OUT_GIF   = "/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/vis/test_seq{seq}_sortcmc.gif"

SEQ = "0000"          # ← 여기만 바꾸면 다른 시퀀스 보기 쉬움
FPS = 10
TRAIL_LEN = 20
IOU_THRES = 0.35
MAX_AGE   = 20
MIN_HITS  = 3
DT        = 0.1
ALPHA     = 0.6       # CMC와 칼만 예측 가중 평균 비율(높을수록 CMC 영향↑)
INCLUDE_CYCLIST = False  # 제출 기준이면 False(차/보행자만)

def color_from_id(tid):
    palette = [
        (255, 99, 71), (30, 144, 255), (60, 179, 113), (238, 130, 238),
        (255, 215, 0), (255, 140, 0), (0, 206, 209), (199, 21, 133),
        (123, 104, 238), (72, 209, 204), (244, 164, 96), (154, 205, 50),
    ]
    return palette[tid % len(palette)]

def load_det_cache(det_txt):
    import csv
    from collections import defaultdict
    frames = defaultdict(list)
    with open(det_txt) as f:
        for row in csv.reader(f):
            if not row: continue
            fr = int(row[0]); cls_id = int(row[1]); score = float(row[2])
            l, t, r, b = map(float, row[3:7])
            frames[fr].append([cls_id, score, [l, t, r, b]])
    return frames

def draw_frame(img, outs, trails, trail_len=20):
    for tid, cls, score, (l,t,r,b) in outs:
        l,t,r,b = map(int, [l,t,r,b])
        c = color_from_id(tid)
        cv2.rectangle(img, (l,t), (r,b), c, 2)
        cv2.putText(img, f"{tid}", (l, max(12, t-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        cx, cy = (l+r)//2, (t+b)//2
        trails.setdefault(tid, deque(maxlen=trail_len)).append((cx, cy))
    for tid, q in list(trails.items()):
        if len(q) >= 2:
            c = color_from_id(tid)
            for i in range(1, len(q)):
                cv2.line(img, q[i-1], q[i], c, 2)
    return img

def save_gif(frames_rgb, path, fps=10, loop=0):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio
        imageio.mimsave(path, frames_rgb, duration=1.0/fps, loop=loop)
    except Exception:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames_rgb]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000/fps), loop=loop)

def main():
    img_dir = os.path.join(TEST_ROOT, "image_02", SEQ)
    det_txt = os.path.join(DET_DIR, f"{SEQ}.txt")
    out_gif = OUT_GIF.format(seq=SEQ)

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert img_files, f"No images in {img_dir}"
    assert os.path.isfile(det_txt), f"Missing det cache: {det_txt}"

    first = cv2.imread(img_files[0])
    H, W = first.shape[:2]

    dets = load_det_cache(det_txt)
    tracker = SORT_CMC(
        iou_thres=IOU_THRES, max_age=MAX_AGE, min_hits=MIN_HITS,
        dt=DT, alpha=ALPHA, img_size=(H, W)
    )
    allowed = {0,1} if not INCLUDE_CYCLIST else {0,1,2}

    frames_rgb, trails = [], {}
    prev = None
    for fr, p in enumerate(img_files):
        curr = cv2.imread(p)
        Hmat = estimate_h(prev, curr) if prev is not None else None

        det_list = [d for d in dets.get(fr, []) if d[0] in allowed]
        outs = tracker.update(det_list, Hmat)
        prev = curr

        vis = draw_frame(curr, outs, trails, trail_len=TRAIL_LEN)
        frames_rgb.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    save_gif(frames_rgb, out_gif, fps=FPS, loop=0)
    print("saved gif:", out_gif)

if __name__ == "__main__":
    main()
