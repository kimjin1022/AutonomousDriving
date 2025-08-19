# run_kitti_sort_gif.py
import os, glob, time
import numpy as np
import cv2
import imageio.v2 as imageio
from ultralytics import YOLO
from tracker import Tracker  # 우리가 만든 tracker.py

# ---------------- 설정 ----------------
SEQ_DIR = "/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training/image_02/0000"
OUT_GIF = "./kitti_0000_sort.gif"

YOLO_WEIGHTS = "yolov8n.pt"
YOLO_CLASSES = [0, 2]   # COCO: person(0), car(2)
CONF_THRES = 0.3

MAX_AGE   = 1
MIN_HITS  = 3
IOU_TH    = 0.15

FPS = 10               # KITTI 10Hz
LIMIT_FRAMES = None    # 예: 150, 전체면 None
DRAW_DETS = True       # 디텍션 박스도 그릴지 여부
DEBUG = True
# -------------------------------------

def yolo_detect(model, frame_bgr):
    """YOLO로 [x1,y1,x2,y2,score] (Nx5) 반환 (사람/차만)"""
    res = model.predict(frame_bgr[:, :, ::-1],
                        classes=YOLO_CLASSES, conf=CONF_THRES, verbose=False)[0]
    dets = []
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        score = float(box.conf[0])
        dets.append([x1, y1, x2, y2, score])
    if len(dets) == 0:
        return np.empty((0, 5), dtype=float)
    return np.array(dets, dtype=float)

def color_for_id(_id):
    """트랙 ID마다 고정 색상 (BGR)"""
    rng = np.random.RandomState(int(_id) * 9973 % (2**32 - 1))
    c = rng.randint(80, 255, size=3).tolist()
    return int(c[0]), int(c[1]), int(c[2])

def draw(frame_bgr, dets, tracks_out):
    """프레임에 디텍션/트랙 결과를 그려 BGR 반환"""
    vis = frame_bgr.copy()

    if DRAW_DETS:
        for d in dets:
            x1, y1, x2, y2, sc = d
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
            cv2.putText(vis, f"det {sc:.2f}", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)

    for t in tracks_out:
        x1, y1, x2, y2, tid = t
        color = color_for_id(tid)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis, f"ID {int(tid)}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis

def main():
    # 이미지 목록
    img_files = sorted(glob.glob(os.path.join(SEQ_DIR, "*.png")))
    if LIMIT_FRAMES is not None:
        img_files = img_files[:LIMIT_FRAMES]
    assert len(img_files) > 0, f"No frames in {SEQ_DIR}"

    # 모델/트래커
    model = YOLO(YOLO_WEIGHTS)
    mot = Tracker(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_TH, debug=DEBUG)

    gif_frames = []
    t0 = time.time()

    for f, path in enumerate(img_files):
        frame = cv2.imread(path)
        dets = yolo_detect(model, frame)  # Nx5

        tracks_out = mot.update(dets)     # Kx5 -> [x1,y1,x2,y2,id]

        if DEBUG:
            print(f"[Frame {f:04d}] dets={len(dets)}  tracks={len(tracks_out)}")

        vis_bgr = draw(frame, dets, tracks_out)

        # GIF용 RGB 프레임으로 변환해 저장 (GIF는 RGB)
        gif_frames.append(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))

    # GIF 저장
    duration = 1.0 / FPS    # 프레임당 시간(초)
    imageio.mimsave(OUT_GIF, gif_frames, duration=duration, loop=0)  # loop=0: 무한 반복

    dt = time.time() - t0
    print(f"\nSaved GIF: {OUT_GIF}")
    print(f"Processed {len(img_files)} frames in {dt:.2f}s ({len(img_files)/dt:.2f} FPS simulated)")

if __name__ == "__main__":
    main()
