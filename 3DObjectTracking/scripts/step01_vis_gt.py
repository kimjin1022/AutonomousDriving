#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 01 — KITTI Tracking GT 2D visualization (full / coarse class profiles)

Classes in KITTI Tracking:
  Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare

Profiles:
  - full   : show all valid classes (exclude DontCare)
  - coarse : {Van,Truck,Tram}->Car, {Person_sitting}->Pedestrian (DontCare excluded)

Examples:
  python step01_vis_gt.py --seq 0 --nframes 10 --gif
  python step01_vis_gt.py --seq 0 --profile coarse --nframes 20
  python step01_vis_gt.py --seq 0 --include "Car,Van,Truck" --nframes 30
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

# -------- Class profiles --------
KITTI_ALL = {
    "Car", "Van", "Truck", "Pedestrian", "Person_sitting",
    "Cyclist", "Tram", "Misc", "DontCare"
}
VALID_IN_FULL = KITTI_ALL - {"DontCare"}

COARSE_MAP = {
    "Van": "Car",
    "Truck": "Car",
    "Tram": "Car",
    "Person_sitting": "Pedestrian",
    # Cyclist, Pedestrian, Car, Misc 그대로 유지
}
COARSE_ALLOWED = {"Car", "Pedestrian", "Cyclist", "Misc"}  # DontCare 제외

# BGR colors
CLASS_COLOR = {
    "Car": (0, 255, 0),
    "Van": (0, 200, 0),
    "Truck": (0, 150, 0),
    "Pedestrian": (255, 0, 0),
    "Person_sitting": (200, 0, 0),
    "Cyclist": (0, 165, 255),
    "Tram": (0, 255, 255),
    "Misc": (200, 200, 200),
    "DontCare": (128, 128, 128),
}

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_labels_by_frame(label_file: str):
    by_frame = {}
    if not os.path.isfile(label_file):
        return by_frame
    with open(label_file, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 17:
                continue
            frame = int(vals[0])
            rec = {
                "frame": frame,
                "track_id": int(vals[1]),
                "type": vals[2],
                "truncated": float(vals[3]),
                "occluded": int(vals[4]),
                "alpha": float(vals[5]),
                "bbox": list(map(float, vals[6:10])),   # l t r b
                "dim":  list(map(float, vals[10:13])),  # h w l
                "loc":  list(map(float, vals[13:16])),  # x y z
                "ry":   float(vals[16]),
            }
            by_frame.setdefault(frame, []).append(rec)
    return by_frame

def map_class(kitti_type: str, profile: str) -> str | None:
    """Return mapped class by profile, or None to drop."""
    if kitti_type == "DontCare":
        return None
    if profile == "coarse":
        return COARSE_MAP.get(kitti_type, kitti_type)
    # full
    return kitti_type

def draw_bboxes(img, dets, allowed_set, profile: str, show_orig: bool, thickness=2):
    for d in dets:
        orig = d["type"]
        mapped = map_class(orig, profile)
        if mapped is None:
            continue
        if allowed_set and mapped not in allowed_set:
            continue

        l, t, r, b = map(int, d["bbox"])
        color = CLASS_COLOR.get(mapped, (180, 180, 180))
        cv2.rectangle(img, (l, t), (r, b), color, thickness)

        if show_orig and mapped != orig:
            tag = f'{mapped}({orig}) id:{d["track_id"]}'
        else:
            tag = f'{mapped} id:{d["track_id"]}'
        cv2.putText(img, tag, (l, max(0, t - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

def parse_include(arg: str | None, profile: str):
    """Return allowed_set by --include or profile default."""
    if arg:
        # 사용자 지정 클래스(콤마 구분)
        user = {c.strip() for c in arg.split(",") if c.strip()}
        return user
    # profile 기본 허용
    return VALID_IN_FULL if profile == "full" else COARSE_ALLOWED

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training",
                        help="KITTI Tracking root (training)")
    parser.add_argument("--out",  default="/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step01_vis_gt",
                        help="output dir for PNG/GIF")
    parser.add_argument("--seq", type=int, default=0, help="sequence id (e.g., 0)")
    parser.add_argument("--start", type=int, default=0, help="start frame index")
    parser.add_argument("--nframes", type=int, default=1, help="number of frames to dump")
    parser.add_argument("--stride", type=int, default=1, help="frame step")
    parser.add_argument("--gif", action="store_true", help="also save GIF")
    parser.add_argument("--profile", choices=["full", "coarse"], default="full",
                        help="class handling profile")
    parser.add_argument("--include", type=str, default=None,
                        help='comma-separated class filter (after mapping), e.g. "Car,Truck"')
    parser.add_argument("--show_orig", action="store_true",
                        help="show original class when profile=coarse and mapping applied")
    args = parser.parse_args()

    seq_str = f"{args.seq:04d}"
    img_dir = os.path.join(args.root, "image_02", seq_str)
    label_file = os.path.join(args.root, "label_02", f"{seq_str}.txt")
    calib_file = os.path.join(args.root, "calib", f"{seq_str}.txt")

    assert os.path.isdir(img_dir), f"[ERR] image dir not found: {img_dir}"
    assert os.path.isfile(calib_file), f"[ERR] calib file not found: {calib_file}"

    if not os.path.isfile(label_file):
        print(f"[WARN] label file not found (maybe test split): {label_file}")

    labels_by_frame = read_labels_by_frame(label_file) if os.path.isfile(label_file) else {}
    out_dir = os.path.join(args.out, f"seq_{seq_str}")
    ensure_dir(out_dir)

    allowed_set = parse_include(args.include, args.profile)

    frames_saved = []
    f = args.start
    remaining = args.nframes
    while remaining > 0:
        img_path = os.path.join(img_dir, f"{f:06d}.png")
        if not os.path.isfile(img_path):
            if remaining == args.nframes:
                raise FileNotFoundError(f"[ERR] first frame not found: {img_path}")
            else:
                print(f"[INFO] reached end at frame {f-args.stride}, stop.")
                break

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"[ERR] failed to read image: {img_path}")

        dets = labels_by_frame.get(f, [])
        vis = img.copy()
        vis = draw_bboxes(vis, dets, allowed_set, args.profile, args.show_orig, thickness=2)

        save_path = os.path.join(out_dir, f"frame_{f:06d}.png")
        cv2.imwrite(save_path, vis)
        print(f"[OK] saved PNG: {save_path}")
        frames_saved.append(save_path)

        f += args.stride
        remaining -= 1

    # optional GIF
    if args.gif:
        if imageio is None:
            print("[WARN] imageio not installed; skip GIF. Run: pip install imageio")
        elif not frames_saved:
            print("[WARN] no frames saved; skip GIF")
        else:
            gif_dir = args.out
            Path(gif_dir).mkdir(parents=True, exist_ok=True)
            gif_path = os.path.join(gif_dir, f"sanity_seq_{seq_str}.gif")
            imgs_rgb = []
            for p in frames_saved:
                bgr = cv2.imread(p)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                imgs_rgb.append(rgb)
            if imgs_rgb:
                imageio.mimsave(gif_path, imgs_rgb, duration=0.08)
                print(f"[OK] saved GIF: {gif_path}")

    # print P2 (reference)
    try:
        with open(calib_file, "r") as fcal:
            for line in fcal:
                if line.startswith("P2:"):
                    print("[INFO] P2:", line.strip())
                    break
    except Exception:
        pass

if __name__ == "__main__":
    main()
