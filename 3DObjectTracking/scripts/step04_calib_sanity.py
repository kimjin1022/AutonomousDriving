#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 04 — Calib & Resize sanity check for KITTI Tracking (camera only)

- Read P2 -> extract K (fx, fy, cx, cy)
- If resizing, compute scaled K', P2' using:
    sx = new_w / orig_w,  sy = new_h / orig_h
    K' = [[fx*sx, 0,     cx*sx],
          [0,     fy*sy, cy*sy],
          [0,     0,     1]]
    P2' = S * P2,  where S = [[sx,0,0],[0,sy,0],[0,0,1]]

Outputs (default dirs):
  PNGs: /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step04_calib_sanity/seq_XXXX/
  YAML: /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step04_calib_sanity/calib_seq_XXXX.yaml
  GIF : /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step04_calib_sanity/sanity_seq_XXXX.gif (optional)

Usage examples:
  python step04_calib_sanity.py --seq 0 --frame 0
  python step04_calib_sanity.py --seq 0 --resize_wh 1242 375
  python step04_calib_sanity.py --seq 0 --scale 0.5 0.5 --gif
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

DEF_ROOT = "/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking"
DEF_OUT  = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step04_calib_sanity"

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def read_P2(calib_path: str):
    assert os.path.isfile(calib_path), f"[ERR] calib not found: {calib_path}"
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                vals = line.split(":", 1)[1].strip().split()
                arr = np.array([float(v) for v in vals], dtype=np.float64)
                if arr.size != 12:
                    raise ValueError(f"P2 in {calib_path} malformed.")
                return arr.reshape(3, 4)
    raise ValueError(f"P2 not found in {calib_path}")

def K_from_P2(P2: np.ndarray):
    # For KITTI left color: P2 = K [I | t], so K is the left 3x3
    K = P2[:, :3].copy()
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    return K, float(fx), float(fy), float(cx), float(cy)

def scale_P2(P2: np.ndarray, sx: float, sy: float):
    S = np.array([[sx, 0,  0],
                  [0,  sy, 0],
                  [0,  0,  1]], dtype=np.float64)
    return S @ P2

def overlay_calib_guides(img: np.ndarray, fx, fy, cx, cy, color=(0,255,255)):
    h, w = img.shape[:2]
    vis = img.copy()

    # principal point
    cv2.circle(vis, (int(round(cx)), int(round(cy))), 3, (0,0,255), -1)
    cv2.putText(vis, f"cx,cy=({cx:.1f},{cy:.1f})", (int(cx)+6, int(cy)-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    # simple FOV guide lines from principal point
    # draw rays to four corners
    corners = [(0,0), (w-1,0), (w-1,h-1), (0,h-1)]
    for (u,v) in corners:
        cv2.line(vis, (int(cx), int(cy)), (int(u), int(v)), color, 1, cv2.LINE_AA)

    # text box with fx, fy
    box_text = f"fx={fx:.1f}, fy={fy:.1f}, w={w}, h={h}"
    cv2.putText(vis, box_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(vis, box_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,40,40), 1, cv2.LINE_AA)
    return vis

def save_yaml(path: str, content: dict):
    # lightweight yaml writer
    with open(path, "w") as f:
        def w(line): f.write(line + "\n")
        w(f"# generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for k, v in content.items():
            if isinstance(v, dict):
                w(f"{k}:")
                for kk, vv in v.items():
                    if isinstance(vv, (list, tuple)):
                        vv_str = "[" + ", ".join(f"{x:.6f}" if isinstance(x,(int,float)) else str(x) for x in vv) + "]"
                        w(f"  {kk}: {vv_str}")
                    elif isinstance(vv, float):
                        w(f"  {kk}: {vv:.6f}")
                    else:
                        w(f"  {kk}: {vv}")
            else:
                w(f"{k}: {v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", default=DEF_ROOT, help="KITTI Tracking root (contains training/)")
    ap.add_argument("--out", default=DEF_OUT, help="output dir")
    ap.add_argument("--seq", type=int, default=0, help="sequence id")
    ap.add_argument("--frame", type=int, default=0, help="frame index to visualize")
    ap.add_argument("--scale", type=float, nargs=2, metavar=("SX","SY"),
                    default=None, help="resize scales (sx sy), e.g., 0.5 0.5")
    ap.add_argument("--resize_wh", type=int, nargs=2, metavar=("W","H"),
                    default=None, help="target width/height, e.g., 1242 375")
    ap.add_argument("--gif", action="store_true", help="save 2-frame GIF (orig vs resized)")
    args = ap.parse_args()

    seq = f"{args.seq:04d}"
    train_root = os.path.join(args.kitti_root, "training")
    img_dir = os.path.join(train_root, "image_02", seq)
    calib_path = os.path.join(train_root, "calib", f"{seq}.txt")
    assert os.path.isdir(img_dir), f"[ERR] not found: {img_dir}"
    P2 = read_P2(calib_path)
    K, fx, fy, cx, cy = K_from_P2(P2)

    # read image and its size
    img_path = os.path.join(img_dir, f"{args.frame:06d}.png")
    assert os.path.isfile(img_path), f"[ERR] image not found: {img_path}"
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"[ERR] failed to read {img_path}")
    H0, W0 = img.shape[0], img.shape[1]

    # determine scaling
    sx = sy = 1.0
    newW, newH = W0, H0
    if args.scale is not None:
        sx, sy = float(args.scale[0]), float(args.scale[1])
        newW, newH = int(round(W0 * sx)), int(round(H0 * sy))
    elif args.resize_wh is not None:
        newW, newH = int(args.resize_wh[0]), int(args.resize_wh[1])
        sx, sy = newW / float(W0), newH / float(H0)

    # compute scaled intrinsics & P2
    P2_scaled = scale_P2(P2, sx, sy)
    Kp, fxp, fyp, cxp, cyp = K_from_P2(P2_scaled)

    # overlay and save
    out_dir = os.path.join(args.out, f"seq_{seq}")
    ensure_dir(out_dir)

    vis_orig = overlay_calib_guides(img, fx, fy, cx, cy, color=(0,255,255))
    png_orig = os.path.join(out_dir, f"calib_orig_{args.frame:06d}.png")
    cv2.imwrite(png_orig, vis_orig)
    print(f"[OK] saved PNG: {png_orig}")

    if (sx != 1.0) or (sy != 1.0):
        # resized preview
        img_resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
        vis_resz = overlay_calib_guides(img_resized, fxp, fyp, cxp, cyp, color=(255,0,255))
        png_resz = os.path.join(out_dir, f"calib_resized_{newW}x{newH}_{args.frame:06d}.png")
        cv2.imwrite(png_resz, vis_resz)
        print(f"[OK] saved PNG: {png_resz}")
    else:
        png_resz = None

    # dump YAML with K/P2 before/after
    yaml_path = os.path.join(args.out, f"calib_seq_{seq}.yaml")
    yaml_obj = {
        "sequence": seq,
        "frame": args.frame,
        "image_size_original": {"W": W0, "H": H0},
        "image_size_resized": {"W": newW, "H": newH},
        "scale": {"sx": sx, "sy": sy},
        "intrinsics_original": {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "K": [K[0,0], K[0,1], K[0,2],
                  K[1,0], K[1,1], K[1,2],
                  K[2,0], K[2,1], K[2,2]],
            "P2": [p for p in P2.reshape(-1)]
        },
        "intrinsics_resized": {
            "fx": fxp, "fy": fyp, "cx": cxp, "cy": cyp,
            "K": [Kp[0,0], Kp[0,1], Kp[0,2],
                  Kp[1,0], Kp[1,1], Kp[1,2],
                  Kp[2,0], Kp[2,1], Kp[2,2]],
            "P2": [p for p in P2_scaled.reshape(-1)]
        }
    }
    save_yaml(yaml_path, yaml_obj)
    print(f"[OK] saved YAML: {yaml_path}")

    # optional GIF (orig vs resized)
    if args.gif and ((sx != 1.0) or (sy != 1.0)):
        if imageio is None:
            print("[WARN] imageio not installed; skip GIF. pip install imageio")
        else:
            gif_path = os.path.join(args.out, f"sanity_seq_{seq}.gif")
            frames = [cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB)]
            frames.append(cv2.cvtColor(cv2.resize(vis_resz, (vis_orig.shape[1], vis_orig.shape[0])), cv2.COLOR_BGR2RGB))
            imageio.mimsave(gif_path, frames, duration=0.6)
            print(f"[OK] saved GIF: {gif_path}")

    # console preview
    print("---- ORIGINAL ----")
    print(f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, W={W0}, H={H0}")
    print("P2:\n", P2)
    print("---- RESIZED ----")
    print(f"sx={sx:.4f}, sy={sy:.4f} -> fx'={fxp:.3f}, fy'={fyp:.3f}, cx'={cxp:.3f}, cy'={cyp:.3f}, W'={newW}, H'={newH}")
    print("P2':\n", P2_scaled)

if __name__ == "__main__":
    main()
