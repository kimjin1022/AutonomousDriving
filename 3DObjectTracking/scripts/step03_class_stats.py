#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 03 — Compute class/size statistics & height-prior for KITTI Tracking (training)

Outputs:
  - class_size_priors.yaml
  - class_size_stats.csv

Usage:
  python step03_class_stats.py \
    --kitti_root /home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking \
    --out        /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step03_stats \
    --profile coarse \
    --exclude_misc

Notes:
  - k factor estimates z ≈ k * fy * H / h_pixels  (per-class median k)
  - Filters can be tuned via CLI options.
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import csv
from datetime import datetime

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
    # Car, Pedestrian, Cyclist, Misc stay the same
}
COARSE_ALLOWED = ["Car", "Pedestrian", "Cyclist", "Misc"]  # DontCare excluded

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

def read_calib_fy(calib_path: str):
    """Read fy from P2 (assumes line starts with 'P2:')."""
    if not os.path.isfile(calib_path):
        return None
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                vals = line.split(":", 1)[1].strip().split()
                if len(vals) >= 6:
                    # P2 is 3x4 row-major: [fx,0,cx,Tx, 0,fy,cy,Ty, 0,0,1,0]
                    try:
                        fy = float(vals[5])
                        return fy
                    except Exception:
                        return None
    return None

def parse_label_line(line: str):
    """
    Return dict or None.
    Columns:
      0: frame, 1: track_id, 2: type, 3: truncated, 4: occluded, 5: alpha,
      6..9: bbox l t r b, 10..12: h w l, 13..15: x y z, 16: ry
    """
    vals = line.strip().split()
    if len(vals) < 17:
        return None
    try:
        frame = int(vals[0])
        obj_type = vals[2]
        truncated = float(vals[3])
        occluded = int(vals[4])
        l, t, r, b = map(float, vals[6:10])
        h, w, l3 = map(float, vals[10:13])
        x, y, z = map(float, vals[13:16])
        return {
            "frame": frame,
            "type": obj_type,
            "truncated": truncated,
            "occluded": occluded,
            "bbox": (l, t, r, b),
            "dim": (h, w, l3),
            "loc": (x, y, z),
        }
    except Exception:
        return None

def map_class(cls: str, profile: str):
    if cls == "DontCare":
        return None
    if profile == "coarse":
        return COARSE_MAP.get(cls, cls)
    return cls  # full

def robust_stats(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "min":  float(np.min(arr)),
        "p25":  float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75":  float(np.percentile(arr, 75)),
        "max":  float(np.max(arr)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root",
                    default="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking",
                    help="KITTI Tracking root containing training/ and testing/")
    ap.add_argument("--out",
                    default="/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step03_stats",
                    help="output dir to save priors (YAML) and CSV stats")
    ap.add_argument("--profile", choices=["coarse", "full"], default="coarse",
                    help="class profile for aggregation")
    ap.add_argument("--exclude_misc", action="store_true",
                    help="exclude 'Misc' from priors/statistics")
    ap.add_argument("--max_trunc", type=float, default=0.5,
                    help="exclude if truncated > max_trunc")
    ap.add_argument("--max_occ", type=int, default=2,
                    help="exclude if occluded   > max_occ")
    ap.add_argument("--min_hp", type=float, default=8.0,
                    help="exclude if bbox pixel height < min_hp")
    ap.add_argument("--min_z", type=float, default=1.0,
                    help="exclude if z < min_z (meters)")
    ap.add_argument("--max_z", type=float, default=120.0,
                    help="exclude if z > max_z (meters)")
    args = ap.parse_args()

    train_root = os.path.join(args.kitti_root, "training")
    img_root   = os.path.join(train_root, "image_02")
    lab_root   = os.path.join(train_root, "label_02")
    calib_root = os.path.join(train_root, "calib")

    assert os.path.isdir(img_root),   f"[ERR] not found: {img_root}"
    assert os.path.isdir(lab_root),   f"[ERR] not found: {lab_root}"
    assert os.path.isdir(calib_root), f"[ERR] not found: {calib_root}"

    ensure_dir(args.out)

    # Which classes to keep for stats
    if args.profile == "full":
        allowed_classes = [c for c in VALID_FULL if not (args.exclude_misc and c == "Misc")]
    else:
        allowed_classes = [c for c in COARSE_ALLOWED if not (args.exclude_misc and c == "Misc")]

    # Accumulators
    per_class_dims_h = defaultdict(list)
    per_class_dims_w = defaultdict(list)
    per_class_dims_l = defaultdict(list)
    per_class_hp     = defaultdict(list)  # bbox pixel heights
    per_class_z      = defaultdict(list)  # distances
    per_class_k      = defaultdict(list)  # k_i = (z * hp) / (fy * h_real)

    fy_list = []

    # Iterate sequences
    seq_ids = list_seq_ids(img_root)
    if not seq_ids:
        raise RuntimeError("[ERR] no training sequences found under image_02")

    for sid in seq_ids:
        calib_path = os.path.join(calib_root, f"{sid}.txt")
        label_path = os.path.join(lab_root, f"{sid}.txt")
        fy = read_calib_fy(calib_path)
        if fy is None:
            print(f"[WARN] cannot read fy from {calib_path}, skip this seq")
            continue
        fy_list.append(fy)

        if not os.path.isfile(label_path):
            # no labels (unlikely for training) -> skip
            print(f"[WARN] label not found: {label_path}, skip")
            continue

        with open(label_path, "r") as f:
            for line in f:
                rec = parse_label_line(line)
                if rec is None:
                    continue

                cls_raw = rec["type"]
                mapped = map_class(cls_raw, args.profile)
                if mapped is None:
                    continue
                if mapped not in allowed_classes:
                    continue

                trunc = rec["truncated"]
                occ   = rec["occluded"]
                (l, t, r, b) = rec["bbox"]
                (h_dim, w_dim, l_dim) = rec["dim"]
                (_, _, z) = rec["loc"]

                hp = max(0.0, b - t)

                # Filters
                if trunc > args.max_trunc:  # too truncated
                    continue
                if occ > args.max_occ:      # too occluded
                    continue
                if hp < args.min_hp:
                    continue
                if not (args.min_z <= z <= args.max_z):
                    continue

                # Collect
                per_class_dims_h[mapped].append(h_dim)
                per_class_dims_w[mapped].append(w_dim)
                per_class_dims_l[mapped].append(l_dim)
                per_class_hp[mapped].append(hp)
                per_class_z[mapped].append(z)

                # Per-instance k using instance's true height h_dim
                k_i = (z * hp) / (fy * max(h_dim, 1e-6))
                per_class_k[mapped].append(k_i)

    # Aggregate fy
    fy_stats = robust_stats(fy_list)

    # Build class stats
    rows = []
    priors = {}
    for cls in allowed_classes:
        H_stats = robust_stats(per_class_dims_h[cls])
        W_stats = robust_stats(per_class_dims_w[cls])
        L_stats = robust_stats(per_class_dims_l[cls])
        hp_stats = robust_stats(per_class_hp[cls])
        z_stats  = robust_stats(per_class_z[cls])
        k_stats  = robust_stats(per_class_k[cls])

        # row for CSV
        rows.append({
            "class": cls,
            "count": H_stats["count"] if H_stats else 0,
            "H_mean": H_stats["mean"] if H_stats else "",
            "H_med":  H_stats["median"] if H_stats else "",
            "W_mean": W_stats["mean"] if W_stats else "",
            "W_med":  W_stats["median"] if W_stats else "",
            "L_mean": L_stats["mean"] if L_stats else "",
            "L_med":  L_stats["median"] if L_stats else "",
            "hp_med": hp_stats["median"] if hp_stats else "",
            "z_med":  z_stats["median"]  if z_stats  else "",
            "k_med":  k_stats["median"]  if k_stats  else "",
        })

        # priors for YAML (use medians as robust priors)
        if H_stats and W_stats and L_stats:
            priors[cls] = {
                "H": round(H_stats["median"], 3),
                "W": round(W_stats["median"], 3),
                "L": round(L_stats["median"], 3),
                "count": H_stats["count"],
                "k_median": round(k_stats["median"], 4) if k_stats else None,
            }
        else:
            priors[cls] = {
                "H": None, "W": None, "L": None, "count": 0, "k_median": None
            }

    # Save CSV
    csv_path = os.path.join(args.out, "class_size_stats.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","count","H_mean","H_med","W_mean","W_med","L_mean","L_med","hp_med","z_med","k_med"])
        for r in rows:
            w.writerow([
                r["class"], r["count"], r["H_mean"], r["H_med"], r["W_mean"], r["W_med"],
                r["L_mean"], r["L_med"], r["hp_med"], r["z_med"], r["k_med"]
            ])
    print(f"[OK] saved CSV: {csv_path}")

    # Save YAML (no external dependency)
    yaml_path = os.path.join(args.out, "class_size_priors.yaml")
    with open(yaml_path, "w") as f:
        f.write("# Auto-generated class size priors and height-based distance scale (k)\n")
        f.write(f"generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"profile: {args.profile}\n")
        f.write(f"filters:\n")
        f.write(f"  max_trunc: {args.max_trunc}\n")
        f.write(f"  max_occ: {args.max_occ}\n")
        f.write(f"  min_hp: {args.min_hp}\n")
        f.write(f"  min_z: {args.min_z}\n")
        f.write(f"  max_z: {args.max_z}\n")
        f.write("camera:\n")
        if fy_stats:
            f.write(f"  fy_median: {round(fy_stats['median'],3)}\n")
            f.write(f"  fy_mean: {round(fy_stats['mean'],3)}\n")
            f.write(f"  fy_std: {round(fy_stats['std'],3)}\n")
        else:
            f.write("  fy_median: null\n  fy_mean: null\n  fy_std: null\n")
        f.write("priors:\n")
        for cls in allowed_classes:
            p = priors.get(cls, {})
            f.write(f"  {cls}:\n")
            f.write(f"    H: {p.get('H')}\n")
            f.write(f"    W: {p.get('W')}\n")
            f.write(f"    L: {p.get('L')}\n")
            f.write(f"    count: {p.get('count')}\n")
            f.write(f"    k_median: {p.get('k_median')}\n")
        f.write("notes:\n")
        f.write("  # Distance from pixel height: z ≈ k_median * fy * H / h_pixels\n")
        f.write("  # Use per-class priors (H,W,L) and k_median from above.\n")
    print(f"[OK] saved YAML: {yaml_path}")

    # Console preview
    print("---- PRIORS (median) ----")
    for cls in allowed_classes:
        p = priors[cls]
        print(f"{cls:12s}  H={p['H']}, W={p['W']}, L={p['L']}, k_med={p['k_median']}, N={p['count']}")

if __name__ == "__main__":
    main()
