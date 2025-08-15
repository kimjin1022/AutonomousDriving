#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 09 — Monocular Depth (MiDaS/DPT) inference & caching for KITTI Tracking

Outputs (defaults):
  /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step09_depth/{model_tag}/
    seq_XXXX/
      frame_XXXXXX_invdepth.npy      # float32, MiDaS inverse-depth (값 클수록 가까움)
      frame_XXXXXX_depth16.png       # 16-bit PNG (로버스트 정규화)
      frame_XXXXXX_viz.png           # 컬러맵 시각화
    index.json
  (opt) sanity_seq_XXXX.gif          # viz 모아 GIF

Usage:
  python step09_depth_midas.py --seq 0 --vis --gif
  python step09_depth_midas.py --use_split /path/to/val.txt --model dpt_hybrid --fp16 --device cuda:0 --vis
  python step09_depth_midas.py --seq 0 --stride 2 --gif --gif_duration 0.2
"""

import os
import argparse
from pathlib import Path
import json
from datetime import datetime

import cv2
import numpy as np
import torch

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

# ---- Fixed defaults ----
DEF_KITTI_ROOT = "/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking"
DEF_OUT_ROOT   = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step09_depth"

# ---- Utils ----
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

def list_frame_indices(seq_img_dir: str):
    if not os.path.isdir(seq_img_dir):
        return []
    idxs = []
    for fn in os.listdir(seq_img_dir):
        if fn.endswith(".png") and fn[:-4].isdigit():
            idxs.append(int(fn[:-4]))
    return sorted(idxs)

def robust_normalize(arr: np.ndarray, p_lo=2.0, p_hi=98.0):
    lo = np.percentile(arr, p_lo)
    hi = np.percentile(arr, p_hi)
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:  # constant image
            return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo + 1e-12)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def colorize(norm01: np.ndarray, cmap=cv2.COLORMAP_INFERNO):
    # norm01 in [0,1]
    u8 = (norm01 * 255.0).round().astype(np.uint8)
    col = cv2.applyColorMap(u8, cmap)
    return col

# ---- Model loader (torch.hub MiDaS) ----
def load_midas(model_name: str, device: str, fp16: bool):
    """
    model_name: dpt_hybrid | dpt_large | midas_small
    """
    hub_repo = "intel-isl/MiDaS"
    name_map = {
        "dpt_hybrid": "DPT_Hybrid",
        "dpt_large":  "DPT_Large",
        "midas_small":"MiDaS_small",
    }
    key = name_map.get(model_name, "DPT_Hybrid")
    model = torch.hub.load(hub_repo, key)
    transforms = torch.hub.load(hub_repo, "transforms")
    if "DPT" in key:
        tfm = transforms.dpt_transform
    else:
        tfm = transforms.small_transform

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    model.eval()
    if fp16 and dev.type == "cuda":
        model.half()
    return model, tfm, dev

def infer_midas(model, tfm, device, bgr_img: np.ndarray, fp16: bool):
    # MiDaS expects RGB
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    inp = tfm(rgb)  # CHW, float
    if fp16 and device.type == "cuda":
        inp = inp.half()
    inp = inp.to(device)
    with torch.no_grad():
        pred = model(inp)
        # upsample to original size
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze(1).squeeze(0)
    inv_depth = pred.detach().float().cpu().numpy()  # higher = closer (inverse-depth-like)
    return inv_depth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", default=DEF_KITTI_ROOT, help="KITTI Tracking root (contains training/)")
    ap.add_argument("--out_root", default=DEF_OUT_ROOT, help="Output root for depth cache")
    # 대상 선택
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seq", type=int, default=None, help="single sequence id, e.g., 0")
    g.add_argument("--use_split", type=str, default=None, help="text file with sequence ids (one per line)")
    # 모델/런타임
    ap.add_argument("--model", choices=["dpt_hybrid","dpt_large","midas_small"], default="dpt_hybrid")
    ap.add_argument("--device", type=str, default=None, help="e.g., cuda:0 or cpu")
    ap.add_argument("--fp16", action="store_true", help="half precision (CUDA only)")
    # 처리 옵션
    ap.add_argument("--stride", type=int, default=1, help="use every N-th frame")
    ap.add_argument("--start", type=int, default=0, help="start frame index")
    ap.add_argument("--limit", type=int, default=None, help="process at most N frames (after start/stride)")
    ap.add_argument("--save_npy", action="store_true", help="save raw inverse-depth as .npy")
    # 시각화/GIF
    ap.add_argument("--vis", action="store_true", help="save colorized PNG")
    ap.add_argument("--gif", action="store_true", help="also save per-seq GIF of viz")
    ap.add_argument("--gif_stride", type=int, default=5)
    ap.add_argument("--gif_max", type=int, default=300)
    ap.add_argument("--gif_duration", type=float, default=0.16, help="sec per frame (smaller=faster)")
    args = ap.parse_args()

    train_root = os.path.join(args.kitti_root, "training")
    img_root   = os.path.join(train_root, "image_02")
    calib_root = os.path.join(train_root, "calib")
    assert os.path.isdir(img_root), f"[ERR] not found: {img_root}"

    # 대상 시퀀스
    if args.seq is not None:
        seq_ids = [f"{args.seq:04d}"]
    elif args.use_split:
        with open(args.use_split, "r") as f:
            seq_ids = [ln.strip().zfill(4) for ln in f if ln.strip()]
    else:
        seq_ids = list_seq_ids(img_root)
    if not seq_ids:
        raise RuntimeError("[ERR] no sequences to process")

    # 모델 로드
    model, tfm, device = load_midas(args.model, args.device, args.fp16)
    tag = args.model

    out_root = os.path.join(args.out_root, tag)
    ensure_dir(out_root)

    print(f"[INFO] sequences={len(seq_ids)}, model={args.model}, device={device}, fp16={args.fp16}")

    for sid in seq_ids:
        seq_img_dir = os.path.join(img_root, sid)
        assert os.path.isdir(seq_img_dir), f"[ERR] not found: {seq_img_dir}"

        frame_idxs = list_frame_indices(seq_img_dir)
        # 프레임 범위/샘플링
        frame_idxs = [fi for i, fi in enumerate(frame_idxs) if (fi >= args.start) and (i % args.stride == 0)]
        if args.limit is not None:
            frame_idxs = frame_idxs[:args.limit]
        if not frame_idxs:
            print(f"[WARN] no frames after filtering in {seq_img_dir}, skip.")
            continue

        seq_out_dir = os.path.join(out_root, f"seq_{sid}")
        ensure_dir(seq_out_dir)

        gif_buf = []

        index = {
            "sequence": sid,
            "image_dir": seq_img_dir,
            "calib_file": os.path.join(calib_root, f"{sid}.txt"),
            "model": args.model,
            "device": str(device),
            "fp16": bool(args.fp16),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frames": []
        }

        for i, fi in enumerate(frame_idxs):
            img_path = os.path.join(seq_img_dir, f"{fi:06d}.png")
            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"[WARN] read fail: {img_path}")
                continue

            inv_depth = infer_midas(model, tfm, device, bgr, args.fp16)  # float32
            # 로버스트 정규화(프레임별)
            norm = robust_normalize(inv_depth, p_lo=2.0, p_hi=98.0)
            depth16 = (norm * 65535.0).round().astype(np.uint16)
            viz = colorize(norm, cmap=cv2.COLORMAP_INFERNO)

            # 저장
            path_npy  = os.path.join(seq_out_dir, f"frame_{fi:06d}_invdepth.npy")
            path_u16  = os.path.join(seq_out_dir, f"frame_{fi:06d}_depth16.png")
            path_viz  = os.path.join(seq_out_dir, f"frame_{fi:06d}_viz.png")
            if args.save_npy:
                np.save(path_npy, inv_depth)
            cv2.imwrite(path_u16, depth16)
            if args.vis:
                cv2.imwrite(path_viz, viz)

            index["frames"].append({
                "frame_index": fi,
                "image_path": img_path,
                "invdepth_npy": (path_npy if args.save_npy else None),
                "depth16_png": path_u16,
                "viz_png": (path_viz if args.vis else None)
            })

            if args.gif and args.vis and imageio is not None:
                # 간단한 합성(원본+깊이)으로 훑어보기
                vis_row = np.concatenate([bgr, viz], axis=1)
                if (i % args.gif_stride == 0) and (len(gif_buf) < args.gif_max):
                    gif_buf.append(cv2.cvtColor(vis_row, cv2.COLOR_BGR2RGB))

        # index 저장
        with open(os.path.join(seq_out_dir, "index.json"), "w") as f:
            json.dump(index, f, indent=2)
        print(f"[OK] seq {sid}: saved {len(index['frames'])} frames → {seq_out_dir}")

        if args.gif and args.vis and imageio is not None and gif_buf:
            gif_path = os.path.join(out_root, f"sanity_seq_{sid}.gif")
            imageio.mimsave(gif_path, gif_buf, duration=args.gif_duration)
            print(f"[OK] saved GIF: {gif_path}")

if __name__ == "__main__":
    main()
