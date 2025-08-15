#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 05 — Cache KITTI Tracking GT to per-frame JSON (camera only)

Outputs (default):
  /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step05_gt_cache/
    seq_0000/
      calib.json
      index.json
      frame_000000.json
      frame_000001.json
      ...

Usage:
  # 전체 시퀀스 처리
  python step05_cache_gt_json.py

  # 특정 시퀀스만
  python step05_cache_gt_json.py --seq 0

  # split 파일로 처리 목록 지정
  python step05_cache_gt_json.py --use_split /home/.../outputs/step02_splits/train.txt

Options:
  --profile {coarse,full}   # 기본: coarse (Van/Truck/Tram->Car, Person_sitting->Pedestrian)
  --drop_misc               # Misc 제외
  --with_empty              # GT 없는 프레임도 빈 objects로 JSON 생성(기본 on)
"""

import os
import argparse
from pathlib import Path
import json
from datetime import datetime

# ----- Default paths -----
DEF_KITTI_ROOT = "/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking"
DEF_OUT_ROOT   = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step05_gt_cache"

# ----- Class profiles -----
KITTI_ALL = {
    "Car", "Van", "Truck", "Pedestrian", "Person_sitting",
    "Cyclist", "Tram", "Misc", "DontCare"
}
VALID_FULL = sorted(list(KITTI_ALL - {"DontCare"}))

COARSE_MAP = {
    "Van": "Car",
    "Truck": "Car",
    "Tram": "Car",
    "Person_sitting": "Pedestrian",
    # Car, Pedestrian, Cyclist, Misc 그대로
}
COARSE_ALLOWED = ["Car", "Pedestrian", "Cyclist", "Misc"]  # DontCare 제외

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def list_seq_ids(img_root: str):
    if not os.path.isdir(img_root): 
        return []
    seqs = []
    for name in os.listdir(img_root):
        full = os.path.join(img_root, name)
        if os.path.isdir(full) and name.isdigit():
            seqs.append(name.zfill(4))
    return sorted(seqs)

def read_P2(calib_path: str):
    if not os.path.isfile(calib_path):
        return None
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                vals = line.split(":", 1)[1].strip().split()
                arr = [float(v) for v in vals]
                if len(arr) != 12:
                    return None
                return [arr[0:4], arr[4:8], arr[8:12]]
    return None

def K_from_P2(P2):
    # P2 is 3x4; K is the left 3x3
    if P2 is None: 
        return None
    return [
        [P2[0][0], P2[0][1], P2[0][2]],
        [P2[1][0], P2[1][1], P2[1][2]],
        [P2[2][0], P2[2][1], P2[2][2]],
    ]

def parse_label_file_to_frames(label_path: str):
    """
    Returns: dict[frame_index] -> list of raw records (without class mapping).
    If label file missing, returns {}.
    """
    frames = {}
    if not os.path.isfile(label_path):
        return frames
    with open(label_path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 17:
                continue
            frame      = int(vals[0])
            track_id   = int(vals[1])
            obj_type   = vals[2]
            if obj_type == "DontCare":
                # skip here; we never cache DontCare
                continue
            truncated  = float(vals[3])
            occluded   = int(vals[4])
            alpha      = float(vals[5])
            l, t, r, b = map(float, vals[6:10])
            h, w, l3   = map(float, vals[10:13])
            x, y, z    = map(float, vals[13:16])
            ry         = float(vals[16])
            rec = {
                "frame": frame,
                "id": track_id,
                "class_raw": obj_type,
                "bbox": [l, t, r, b],
                "dim":  [h, w, l3],
                "loc":  [x, y, z],
                "ry": ry,
                "truncated": truncated,
                "occluded": occluded,
                "alpha": alpha,
            }
            frames.setdefault(frame, []).append(rec)
    return frames

def map_class(cls_raw: str, profile: str):
    if profile == "coarse":
        return COARSE_MAP.get(cls_raw, cls_raw)
    return cls_raw  # full

def collect_frame_list(img_dir: str):
    # Return list of available frame indices based on .png files
    if not os.path.isdir(img_dir): 
        return []
    idxs = []
    for name in os.listdir(img_dir):
        if name.endswith(".png") and name[:-4].isdigit():
            idxs.append(int(name[:-4]))
    return sorted(idxs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", default=DEF_KITTI_ROOT,
                    help="KITTI Tracking root containing training/ and testing/")
    ap.add_argument("--out_root", default=DEF_OUT_ROOT,
                    help="Output root for JSON cache")
    ap.add_argument("--profile", choices=["coarse", "full"], default="coarse",
                    help="Class mapping profile")
    ap.add_argument("--drop_misc", action="store_true",
                    help="Drop 'Misc' class from cache")
    ap.add_argument("--with_empty", action="store_true", default=True,
                    help="Also generate JSON for frames without GT objects")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--seq", type=int, default=None,
                       help="Process a single sequence id (e.g., 0)")
    group.add_argument("--use_split", type=str, default=None,
                       help="Text file containing sequence ids (one per line)")
    args = ap.parse_args()

    train_root = os.path.join(args.kitti_root, "training")
    img_root   = os.path.join(train_root, "image_02")
    lab_root   = os.path.join(train_root, "label_02")
    calib_root = os.path.join(train_root, "calib")

    assert os.path.isdir(img_root),   f"[ERR] not found: {img_root}"
    assert os.path.isdir(lab_root),   f"[ERR] not found: {lab_root}"
    assert os.path.isdir(calib_root), f"[ERR] not found: {calib_root}"
    ensure_dir(args.out_root)

    # Determine which classes are allowed for this profile
    if args.profile == "full":
        allowed_classes = [c for c in VALID_FULL if not (args.drop_misc and c == "Misc")]
    else:
        allowed_classes = [c for c in COARSE_ALLOWED if not (args.drop_misc and c == "Misc")]

    # Build target sequence list
    if args.seq is not None:
        seq_ids = [f"{args.seq:04d}"]
    elif args.use_split:
        with open(args.use_split, "r") as f:
            seq_ids = [ln.strip().zfill(4) for ln in f if ln.strip()]
    else:
        seq_ids = list_seq_ids(img_root)
        if not seq_ids:
            raise RuntimeError("[ERR] no sequences found under training/image_02")

    print(f"[INFO] sequences to process: {len(seq_ids)} (profile={args.profile}, drop_misc={args.drop_misc})")

    for sid in seq_ids:
        seq_out = os.path.join(args.out_root, f"seq_{sid}")
        ensure_dir(seq_out)

        # 1) calib.json
        P2 = read_P2(os.path.join(calib_root, f"{sid}.txt"))
        K  = K_from_P2(P2) if P2 is not None else None
        calib_obj = {
            "sequence": sid,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "P2": P2,
            "K":  K,
            "profile": args.profile,
        }
        with open(os.path.join(seq_out, "calib.json"), "w") as f:
            json.dump(calib_obj, f, indent=2)

        # 2) labels -> frames dict
        frames_raw = parse_label_file_to_frames(os.path.join(lab_root, f"{sid}.txt"))

        # 3) image frame indices
        seq_img_dir = os.path.join(img_root, sid)
        frame_indices = collect_frame_list(seq_img_dir)
        if not frame_indices:
            print(f"[WARN] no images in {seq_img_dir}, skip sequence")
            continue

        # 4) per-frame JSON
        written = []
        for fi in frame_indices:
            raw_list = frames_raw.get(fi, [])
            objects = []
            for rec in raw_list:
                cls_mapped = map_class(rec["class_raw"], args.profile)
                if cls_mapped not in allowed_classes:
                    continue
                obj = {
                    "id": rec["id"],
                    "class": cls_mapped,
                    "class_raw": rec["class_raw"],
                    "bbox": rec["bbox"],   # [l,t,r,b]
                    "dim":  rec["dim"],    # [h,w,l]
                    "loc":  rec["loc"],    # [x,y,z] (camera coords)
                    "ry":   rec["ry"],
                    "truncated": rec["truncated"],
                    "occluded":  rec["occluded"],
                    "alpha":     rec["alpha"],
                }
                objects.append(obj)

            if not objects and not args.with_empty:
                # skip empty frame
                continue

            frame_obj = {
                "sequence": sid,
                "frame_index": fi,
                "image_path": os.path.join(seq_img_dir, f"{fi:06d}.png"),
                "objects": objects
            }
            out_path = os.path.join(seq_out, f"frame_{fi:06d}.json")
            with open(out_path, "w") as f:
                json.dump(frame_obj, f, indent=2)
            written.append(f"frame_{fi:06d}.json")

        # 5) index.json
        index_obj = {
            "sequence": sid,
            "num_frames": len(frame_indices),
            "json_frames": written,
            "image_dir": seq_img_dir,
            "calib_file": os.path.join(calib_root, f"{sid}.txt"),
            "profile": args.profile,
            "allowed_classes": allowed_classes,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(seq_out, "index.json"), "w") as f:
            json.dump(index_obj, f, indent=2)

        print(f"[OK] seq {sid}: cached {len(written)} frames → {seq_out}")

if __name__ == "__main__":
    main()
