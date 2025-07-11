# scripts/step1_vis_gt.py

import os, glob, cv2
from collections import defaultdict

def read_labels(label_path):
    frames = defaultdict(list)
    with open(label_path) as f:
        for line in f:
            if not line.strip():
                continue
            p = line.strip().split()
            frame   = int(p[0])
            trackId = int(p[1])
            objType = p[2]
            l, t, r, b = map(float, p[6:10])  # KITTI: l,t,r,b
            frames[frame].append({"id": trackId, "type": objType, "bbox": [l, t, r, b]})
    return frames

def draw_boxes(img, objs):
    palette = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    for o in objs:
        l,t,r,b = map(int, o["bbox"])
        color = palette[o["id"] % len(palette)]
        cv2.rectangle(img, (l,t), (r,b), color, 2)
        cv2.putText(img, f"{o['id']}:{o['type']}", (l, max(10, t-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

def main(kitti_root, seq="0000", frame_idx=0, out_path="outputs/vis/gt_seq0000_f0000.jpg"):
    img_dir    = os.path.join(kitti_root, "image_02", seq)
    label_path = os.path.join(kitti_root, "label_02", f"{seq}.txt")
    assert os.path.isdir(img_dir), f"Not found: {img_dir}"
    assert os.path.isfile(label_path), f"Not found: {label_path}"

    labels = read_labels(label_path)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    print(f"[SEQ {seq}] images={len(img_files)}  labeled_frames={len(labels)}")

    img = cv2.imread(img_files[frame_idx])
    objs = labels.get(frame_idx, [])
    vis = draw_boxes(img.copy(), objs)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)

    uniq_ids = {o["id"] for arr in labels.values() for o in arr}
    print(f"unique track IDs: {len(uniq_ids)}")
    print(f"saved: {out_path}")

if __name__ == "__main__":
    # ↓↓↓ 여기를 네 환경에 맞게 바꿔 실행
    main(
        kitti_root="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training",
        seq="0000",
        frame_idx=0,
        out_path="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/vis/gt_seq0000_f0000.jpg",
    )
