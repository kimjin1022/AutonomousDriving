# scripts/step3_detcache_dummy.py
import os, glob, cv2, csv, random
import numpy as np
from collections import defaultdict

# 고정 클래스 매핑
MAP = {
    "Car": "car", "Van": "car", "Truck": "car",
    "Pedestrian": "pedestrian", "Person_sitting": "pedestrian",
    "Cyclist": "cyclist",
}
IGNORE = {"Tram", "Misc", "DontCare"}

CLASS_TO_ID = {"car": 0, "pedestrian": 1, "cyclist": 2}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
COLORS = {0:(255,0,0), 1:(0,0,255), 2:(0,255,0)}  # car/ped/cyc

def parse_and_map(label_path):
    frames = defaultdict(list)
    with open(label_path) as f:
        for ln in f:
            if not ln.strip():
                continue
            p = ln.split()
            typ = p[2]
            if typ in IGNORE: 
                continue
            cls = MAP.get(typ)
            if cls is None:
                continue
            frame = int(p[0]); tid = int(p[1])
            l,t,r,b = map(float, p[6:10])
            frames[frame].append({"id": tid, "cls": cls, "bbox": [l,t,r,b]})
    return frames

def clamp_box(l, t, r, b, W, H):
    l = max(0, min(l, W-1)); r = max(0, min(r, W-1))
    t = max(0, min(t, H-1)); b = max(0, min(b, H-1))
    if r < l: l, r = r, l
    if b < t: t, b = b, t
    return l, t, r, b

def save_dummy_dets(frames, img_dir, out_txt, keep_prob=0.85, noise_px=3.0, seed=0):
    random.seed(seed); np.random.seed(seed)
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not img_files:
        raise FileNotFoundError(f"No images in {img_dir}")
    # 임시로 첫 프레임 크기로 클램프(모든 프레임 동일 해상도)
    h, w = cv2.imread(img_files[0]).shape[:2]

    with open(out_txt, "w", newline="") as f:
        writer = csv.writer(f)
        for frame in sorted(frames.keys()):
            for o in frames[frame]:
                if random.random() > keep_prob:
                    continue  # 드랍로스
                l,t,r,b = o["bbox"]
                # 가우시안 노이즈
                l += np.random.randn()*noise_px
                t += np.random.randn()*noise_px
                r += np.random.randn()*noise_px
                b += np.random.randn()*noise_px
                l,t,r,b = clamp_box(l,t,r,b,w,h)
                cls_id = CLASS_TO_ID[o["cls"]]
                score = float(np.clip(np.random.uniform(0.5, 0.95), 0, 1))
                writer.writerow([frame, cls_id, f"{score:.3f}", 
                                 f"{l:.1f}", f"{t:.1f}", f"{r:.1f}", f"{b:.1f}"])
    print(f"wrote det cache: {out_txt}")

def load_dets_one_frame(det_txt, frame_idx):
    dets = []
    with open(det_txt) as f:
        for row in csv.reader(f):
            if not row: 
                continue
            fr = int(row[0])
            if fr != frame_idx: 
                continue
            cls_id = int(row[1]); score = float(row[2])
            l,t,r,b = map(float, row[3:7])
            dets.append((cls_id, score, [l,t,r,b]))
    return dets

def draw_dets(img, dets):
    for cls_id, score, (l,t,r,b) in dets:
        l,t,r,b = map(int, [l,t,r,b])
        color = COLORS[cls_id]
        cv2.rectangle(img, (l,t), (r,b), color, 2)
        cv2.putText(img, f"{ID_TO_CLASS[cls_id]} {score:.2f}", 
                    (l, max(10, t-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

def main(
    kitti_root="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training",
    seq="0000",
    out_det_dir="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/dets_dummy",
    out_vis="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/vis/step3_seq0000_f0000.jpg",
    frame_idx=0
):
    img_dir    = os.path.join(kitti_root, "image_02", seq)
    label_path = os.path.join(kitti_root, "label_02", f"{seq}.txt")
    assert os.path.isdir(img_dir), f"Not found: {img_dir}"
    assert os.path.isfile(label_path), f"Not found: {label_path}"

    frames = parse_and_map(label_path)
    out_txt = os.path.join(out_det_dir, f"{seq}.txt")
    save_dummy_dets(frames, img_dir, out_txt)

    # 로드 + 시각화
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    img = cv2.imread(img_files[frame_idx])
    dets = load_dets_one_frame(out_txt, frame_idx)
    vis = draw_dets(img.copy(), dets)
    os.makedirs(os.path.dirname(out_vis) or ".", exist_ok=True)
    cv2.imwrite(out_vis, vis)
    print(f"frame {frame_idx} dets: {len(dets)}")
    print("saved vis:", out_vis)

if __name__ == "__main__":
    main()
