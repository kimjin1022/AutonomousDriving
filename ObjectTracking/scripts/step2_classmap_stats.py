import os, glob, cv2
from collections import defaultdict, Counter

MAP = {
    "Car":"car", "Van":"car", "Truck":"car",
    "Pedestrian":"pedestrian", "Person_sitting":"pedestrian",
    "Cyclist":"cyclist",
}
IGNORE = {"Tram", "Misc", "DontCare"}

COLORS = {"car":(255,0,0), "pedestrian":(0,0,255), "cyclist":(0,255,0)}

def parse_and_map(label_path):
    frames = defaultdict(list)
    raw_counts = Counter()
    with open(label_path) as f:
        for ln in f:
            if not ln.strip(): 
                continue
            p = ln.split()
            frame   = int(p[0]); tid = int(p[1]); typ = p[2]
            raw_counts[typ] += 1
            if typ in IGNORE: 
                continue
            cls = MAP.get(typ, None)
            if cls is None:
                continue
            l,t,r,b = map(float, p[6:10])
            frames[frame].append({"id":tid, "cls":cls, "bbox":[l,t,r,b]})
    return frames, raw_counts

def draw(img, objs):
    for o in objs:
        l,t,r,b = map(int, o["bbox"])
        c = COLORS[o["cls"]]
        cv2.rectangle(img, (l,t), (r,b), c, 2)
        cv2.putText(img, f"{o['id']}:{o['cls']}", (l, max(10, t-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return img

def main(kitti_root, seq="0000", frame_idx=0, out_path="outputs/vis/step2_seq0000_f0000.jpg"):
    img_dir    = os.path.join(kitti_root, "image_02", seq)
    label_path = os.path.join(kitti_root, "label_02", f"{seq}.txt")
    assert os.path.isdir(img_dir), f"Not found: {img_dir}"
    assert os.path.isfile(label_path), f"Not found: {label_path}"

    frames, raw_counts = parse_and_map(label_path)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    # 통계
    kept_counts = Counter(o["cls"] for fr in frames.values() for o in fr)
    uniq_ids = {o["id"] for fr in frames.values() for o in fr}

    print(f"[SEQ {seq}] images={len(img_files)} mapped_frames={len(frames)}")
    print("[RAW]   top-5 types:", raw_counts.most_common(5))
    print("[KEPT]  per-class   :", dict(kept_counts))
    print(f"[KEPT]  unique track IDs: {len(uniq_ids)}")

    # 시각화 저장
    img = cv2.imread(img_files[frame_idx])
    vis = draw(img.copy(), frames.get(frame_idx, []))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)
    print("saved:", out_path)

if __name__ == "__main__":
    main(
        kitti_root="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training",
        seq="0000",
        frame_idx=0,
        out_path="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/vis/step2_seq0000_f0000.jpg",
    )
