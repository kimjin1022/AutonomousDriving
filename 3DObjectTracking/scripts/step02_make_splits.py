#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 02 — Create Train/Val splits for KITTI Tracking

- Scans training sequences under: <kitti_root>/training/image_02/<seq>/
- Counts frames and labeled objects (excluding DontCare)
- Saves:
    train.txt, val.txt, test.txt (if testing exists), stats.csv
- Split methods:
    rand (default, reproducible with --seed)
    evenodd (even seq -> train, odd seq -> val)

Usage:
  python step02_make_splits.py \
    --kitti_root /home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking \
    --out        /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step02_splits \
    --ratio 0.8 --method rand --seed 42
"""

import os
import argparse
from pathlib import Path
import random
import csv
from typing import Dict, List, Tuple

VALID_EXCEPT_DONTCARE = {
    "Car", "Van", "Truck", "Pedestrian", "Person_sitting",
    "Cyclist", "Tram", "Misc"
}

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def list_seq_ids(img_root: str) -> List[str]:
    """Return sorted zero-padded sequence ids under image_02 root."""
    if not os.path.isdir(img_root):
        return []
    seqs = []
    for name in os.listdir(img_root):
        full = os.path.join(img_root, name)
        if os.path.isdir(full) and name.isdigit():
            seqs.append(name.zfill(4))
    return sorted(seqs)

def count_frames(seq_img_dir: str) -> int:
    if not os.path.isdir(seq_img_dir):
        return 0
    return sum(1 for fn in os.listdir(seq_img_dir) if fn.endswith(".png"))

def parse_labels(label_file: str) -> int:
    """Count valid labeled objects excluding DontCare."""
    if not os.path.isfile(label_file):
        return 0
    n = 0
    with open(label_file, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 3:
                continue
            cls = vals[2]
            if cls in VALID_EXCEPT_DONTCARE:
                n += 1
    return n

def build_stats(kitti_root: str) -> List[Dict]:
    train_img_root = os.path.join(kitti_root, "training", "image_02")
    train_label_root = os.path.join(kitti_root, "training", "label_02")
    seqs = list_seq_ids(train_img_root)
    rows = []
    for sid in seqs:
        nframes = count_frames(os.path.join(train_img_root, sid))
        nlabels = parse_labels(os.path.join(train_label_root, f"{sid}.txt"))
        rows.append({"seq": sid, "frames": nframes, "labels": nlabels})
    return rows

def split_seqs(rows: List[Dict], ratio: float, method: str, seed: int) -> Tuple[List[str], List[str]]:
    seq_ids = [r["seq"] for r in rows]
    if method == "evenodd":
        train = [s for s in seq_ids if int(s) % 2 == 0]
        val   = [s for s in seq_ids if int(s) % 2 == 1]
        return train, val
    # rand
    rng = random.Random(seed)
    seq_ids_shuf = seq_ids[:]
    rng.shuffle(seq_ids_shuf)
    k = int(round(len(seq_ids_shuf) * ratio))
    train = sorted(seq_ids_shuf[:k])
    val   = sorted(seq_ids_shuf[k:])
    return train, val

def write_list(path: str, items: List[str]):
    with open(path, "w") as f:
        for it in items:
            f.write(f"{it}\n")

def write_stats_csv(path: str, rows: List[Dict]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "frames", "labels"])
        for r in rows:
            w.writerow([r["seq"], r["frames"], r["labels"]])

def list_test_seqs(kitti_root: str) -> List[str]:
    test_img_root = os.path.join(kitti_root, "testing", "image_02")
    return list_seq_ids(test_img_root)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root",
                    default="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking",
                    help="KITTI Tracking root containing training/ and testing/")
    ap.add_argument("--out",
                    default="/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step02_splits",
                    help="output dir to save split files and stats")
    ap.add_argument("--ratio", type=float, default=0.8, help="train ratio for random split")
    ap.add_argument("--method", choices=["rand", "evenodd"], default="rand",
                    help="split method: random or even/odd by seq id")
    ap.add_argument("--seed", type=int, default=42, help="random seed for rand split")
    args = ap.parse_args()

    train_img_root = os.path.join(args.kitti_root, "training", "image_02")
    train_label_root = os.path.join(args.kitti_root, "training", "label_02")
    calib_root = os.path.join(args.kitti_root, "training", "calib")

    assert os.path.isdir(train_img_root), f"[ERR] not found: {train_img_root}"
    assert os.path.isdir(train_label_root), f"[ERR] not found: {train_label_root}"
    assert os.path.isdir(calib_root), f"[ERR] not found: {calib_root}"

    ensure_dir(args.out)

    # 1) build stats
    rows = build_stats(args.kitti_root)
    if not rows:
        raise RuntimeError("[ERR] No training sequences found under image_02")
    write_stats_csv(os.path.join(args.out, "stats.csv"), rows)
    print(f"[OK] stats.csv saved with {len(rows)} sequences")

    # 2) split
    train_ids, val_ids = split_seqs(rows, args.ratio, args.method, args.seed)
    write_list(os.path.join(args.out, "train.txt"), train_ids)
    write_list(os.path.join(args.out, "val.txt"),   val_ids)
    print(f"[OK] train/val saved: {len(train_ids)} / {len(val_ids)} seqs (method={args.method}, ratio={args.ratio})")

    # 3) (optional) test list if exists
    test_ids = list_test_seqs(args.kitti_root)
    if test_ids:
        write_list(os.path.join(args.out, "test.txt"), test_ids)
        print(f"[OK] test.txt saved: {len(test_ids)} seqs")
    else:
        print("[INFO] testing/image_02 not found or empty; skipped test.txt")

    # 4) quick console preview
    def preview(name, ids):
        s = ", ".join(ids[:10]) + (" ..." if len(ids) > 10 else "")
        return f"{name} ({len(ids)}): {s}"
    print(preview("train", train_ids))
    print(preview("val",   val_ids))

if __name__ == "__main__":
    main()
