# scripts/step14a_track_one_seq_kitti.py
# 목적: testing 한 시퀀스(또는 일부 프레임 범위)만 SORT+CMC로 추적 → KITTI 제출 포맷 txt로 저장
# 입력 캐시 포맷: frame, class_id, score, left, top, right, bottom  (콤마)
# 제출 포맷(행):
#   frame track_id type truncated occluded alpha l t r b h w l x y z ry score
#   ※ 여기서는 type만 채우고 3D/가림 관련은 -1로 둡니다.

import os, glob, csv, cv2
from collections import defaultdict
from step7_sort_cmc_minimal import SORT_CMC, estimate_h  # Step7 재사용

# ======== 손댈만한 기본값들 ========
TEST_ROOT = "/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/testing"
DET_DIR   = "/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/dets_yolo_test"
OUT_DIR   = "/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/tracks_test_kitti"

SEQ            = "0000"  # ← 여기만 바꾸면 원하는 시퀀스 하나만 처리
FRAME_START    = None    # 예: 0  (한 프레임만이면 START=END 같은 값)
FRAME_END      = None    # 예: 0
INCLUDE_CYCLIST = False  # 제출엔 기본적으로 Car/Ped만

# 트래커 하이퍼파라미터(바꿔보기 쉬움)
IOU_THRES = 0.35
MAX_AGE   = 20
MIN_HITS  = 3
DT        = 0.1
ALPHA     = 0.6

# 클래스 ID → KITTI type 이름
ID_TO_TYPE = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def clamp_box(l, t, r, b, W, H):
    l = max(0, min(float(l), W - 1)); r = max(0, min(float(r), W - 1))
    t = max(0, min(float(t), H - 1)); b = max(0, min(float(b), H - 1))
    if r < l: l, r = r, l
    if b < t: t, b = b, t
    return [l, t, r, b]

def load_det_cache(det_txt):
    frames = defaultdict(list)
    with open(det_txt) as f:
        for row in csv.reader(f):
            if not row: continue
            fr = int(row[0]); cls_id = int(row[1]); score = float(row[2])
            l, t, r, b = map(float, row[3:7])
            frames[fr].append([cls_id, score, [l, t, r, b]])
    return frames

def track_one_sequence(seq, frame_start=None, frame_end=None):
    img_dir = os.path.join(TEST_ROOT, "image_02", seq)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert img_files, f"No images in {img_dir}"

    det_txt = os.path.join(DET_DIR, f"{seq}.txt")
    assert os.path.isfile(det_txt), f"Det cache not found: {det_txt}"

    out_txt = os.path.join(OUT_DIR, f"{seq}.txt")
    ensure_dir(os.path.dirname(out_txt) or ".")

    # 프레임 범위 설정
    first_idx, last_idx = 0, len(img_files) - 1
    if frame_start is not None: first_idx = max(0, int(frame_start))
    if frame_end   is not None: last_idx  = min(len(img_files)-1, int(frame_end))
    assert first_idx <= last_idx, "FRAME_START must be <= FRAME_END"

    H, W = cv2.imread(img_files[0]).shape[:2]
    dets = load_det_cache(det_txt)
    tracker = SORT_CMC(
        iou_thres=IOU_THRES, max_age=MAX_AGE, min_hits=MIN_HITS,
        dt=DT, alpha=ALPHA, img_size=(H, W)
    )
    allowed = {0, 1} if not INCLUDE_CYCLIST else {0, 1, 2}

    prev = None
    with open(out_txt, "w") as f:
        writer = csv.writer(f, delimiter=' ')
        for fr in range(first_idx, last_idx + 1):
            curr = cv2.imread(img_files[fr])
            Hmat = estimate_h(prev, curr) if prev is not None else None

            det_list = [d for d in dets.get(fr, []) if d[0] in allowed]
            outs = tracker.update(det_list, Hmat)
            prev = curr

            for tid, cls_id, score, (l, t, r, b) in outs:
                if cls_id not in allowed: continue
                l, t, r, b = clamp_box(l, t, r, b, W, H)
                typ = ID_TO_TYPE[cls_id]
                writer.writerow([
                    fr, int(tid), typ,
                    -1, -1, -1,
                    f"{l:.2f}", f"{t:.2f}", f"{r:.2f}", f"{b:.2f}",
                    -1, -1, -1,
                    -1, -1, -1,
                    -1,
                    f"{float(score):.3f}",
                ])

    print(f"[OK] seq {seq} frames [{first_idx}..{last_idx}] → {out_txt}")

def main():
    ensure_dir(OUT_DIR)
    track_one_sequence(SEQ, FRAME_START, FRAME_END)

if __name__ == "__main__":
    main()