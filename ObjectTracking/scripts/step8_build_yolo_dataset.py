# scripts/step8_build_yolo_dataset.py
# 목적: KITTI Tracking 라벨을 YOLO 포맷으로 변환하고, 학습/검증 split을 만들기
# 출력 구조:
#   <out_root>/
#     images/train/*.png      # 원본 이미지로 심볼릭 링크(복사 X)
#     images/val/*.png
#     labels/train/*.txt      # YOLO 포맷: cls x y w h (정규화)
#     labels/val/*.txt
#     kitti_yolo.yaml         # YOLO 학습용 데이터셋 YAML

import os, glob, cv2
from collections import defaultdict, Counter

# ----- 경로 기본값 (네가 준 경로) -----
KITTI_ROOT = "/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training"
OUT_ROOT   = "/home/jinjinjara1022/AutonomousDriving/ObjectTracking/datasets/kitti_yolo"

# ----- 클래스 매핑 -----
MAP = {
    "Car":"car", "Van":"car", "Truck":"car",
    "Pedestrian":"pedestrian", "Person_sitting":"pedestrian",
    "Person":"pedestrian",          
    "Cyclist":"cyclist",
}
IGNORE = {"Tram", "Misc", "DontCare"}
CLASS_TO_ID = {"car":0, "pedestrian":1, "cyclist":2}

# ----- 유틸 -----
def read_seq_labels(label_path):
    """KITTI Tracking 시퀀스 라벨을 frame->list 로 파싱"""
    frames = defaultdict(list)
    raw_counts = Counter()
    with open(label_path) as f:
        for ln in f:
            if not ln.strip(): continue
            p = ln.split()
            frame = int(p[0]); obj_type = p[2]
            raw_counts[obj_type] += 1
            if obj_type in IGNORE: 
                continue
            cls = MAP.get(obj_type)
            if cls is None: 
                continue
            l, t, r, b = map(float, p[6:10])
            frames[frame].append((cls, [l,t,r,b]))
    return frames, raw_counts

def kitti_to_yolo(ltrb, W, H):
    l,t,r,b = ltrb
    # 화면 클램프 & 비정상 박스 보정
    l = max(0, min(l, W-1)); r = max(0, min(r, W-1))
    t = max(0, min(t, H-1)); b = max(0, min(b, H-1))
    if r < l: l, r = r, l
    if b < t: t, b = b, t
    w = max(1.0, r - l); h = max(1.0, b - t)
    cx = l + w/2.0; cy = t + h/2.0
    # 정규화
    return cx / W, cy / H, w / W, h / H

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def symlink_force(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)

def main():
    img_root = os.path.join(KITTI_ROOT, "image_02")
    lbl_root = os.path.join(KITTI_ROOT, "label_02")
    assert os.path.isdir(img_root), f"Not found: {img_root}"
    assert os.path.isdir(lbl_root), f"Not found: {lbl_root}"

    # 시퀀스 리스트
    seqs = sorted([d for d in os.listdir(img_root) if d.isdigit()])
    assert seqs, "No sequences found."

    # 간단 split (예: 앞 80% train, 뒤 20% val)
    n = len(seqs)
    train_seqs = seqs[: int(n*0.8)]
    val_seqs   = seqs[int(n*0.8):]

    # 출력 디렉토리 준비
    paths = {
        "images/train": os.path.join(OUT_ROOT, "images", "train"),
        "images/val"  : os.path.join(OUT_ROOT, "images", "val"),
        "labels/train": os.path.join(OUT_ROOT, "labels", "train"),
        "labels/val"  : os.path.join(OUT_ROOT, "labels", "val"),
    }
    for p in paths.values(): ensure_dir(p)

    kept_counts = Counter(); raw_counts_total = Counter()
    num_img_train = num_img_val = 0

    def process_split(split_name, split_seqs):
        nonlocal kept_counts, raw_counts_total, num_img_train, num_img_val
        for seq in split_seqs:
            seq_img_dir = os.path.join(img_root, seq)
            seq_lbl_path = os.path.join(lbl_root, f"{seq}.txt")
            frames, raw_counts = read_seq_labels(seq_lbl_path)
            raw_counts_total.update(raw_counts)

            # 이미지 목록
            img_files = sorted(glob.glob(os.path.join(seq_img_dir, "*.png")))
            assert img_files, f"No images in {seq_img_dir}"

            # 해상도(모두 동일)
            H, W = cv2.imread(img_files[0]).shape[:2]

            for fr, img_path in enumerate(img_files):
                # 대상 파일명: <seq>_<frame6>.png / .txt
                base = f"{seq}_{fr:06d}"
                if split_name == "train":
                    out_img = os.path.join(paths["images/train"], f"{base}.png")
                    out_lbl = os.path.join(paths["labels/train"], f"{base}.txt")
                else:
                    out_img = os.path.join(paths["images/val"], f"{base}.png")
                    out_lbl = os.path.join(paths["labels/val"], f"{base}.txt")

                # 이미지: 링크(복사 X)
                symlink_force(img_path, out_img)

                # 라벨: 해당 프레임 객체들을 YOLO로 기록
                objs = frames.get(fr, [])
                lines = []
                for cls, ltrb in objs:
                    cx, cy, w, h = kitti_to_yolo(ltrb, W, H)
                    # 너무 작은 박스는 옵션으로 제거(픽셀기준) — 원하면 주석 해제
                    # if w*W < 4 or h*H < 4: 
                    #     continue
                    lines.append(f"{CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    kept_counts[cls] += 1

                # 비어 있어도 빈 파일 생성(YOLO는 빈 라벨도 허용)
                with open(out_lbl, "w") as f:
                    f.write("\n".join(lines))

                if split_name == "train": num_img_train += 1
                else: num_img_val += 1

    process_split("train", train_seqs)
    process_split("val", val_seqs)

    # 데이터셋 YAML 생성
    yaml_path = os.path.join(OUT_ROOT, "kitti_yolo.yaml")
    with open(yaml_path, "w") as f:
        f.write(
f"""# auto-generated
path: {OUT_ROOT}
train: images/train
val: images/val

nc: 3
names: [car, pedestrian, cyclist]
"""
        )

    # 로그
    print("=== DONE ===")
    print(f"train seqs: {train_seqs}")
    print(f"val   seqs: {val_seqs}")
    print(f"images: train={num_img_train}, val={num_img_val}")
    print(f"[RAW top]   : {dict(raw_counts_total)}")
    print(f"[KEPT per-class]: {dict(kept_counts)}")
    print("yaml:", yaml_path)

if __name__ == "__main__":
    main()
