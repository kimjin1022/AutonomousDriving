#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 10 — Height-prior lifting: 2D boxes -> camera 3D (X,Y,Z), dims

Inputs:
  - Tracks (Step07):
      /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step07_track2d/{tracker}/{model_tag}/seq_XXXX/frame_XXXXXX.json
  - Priors (Step03 YAML): class_size_priors.yaml
  - Calib (KITTI): P2 from .../datasets/KITTI_Tracking/training/calib/{seq}.txt
  - (Optional) MiDaS (Step09): index.json & per-frame *_invdepth.npy or *_depth16.png

Outputs:
  - 3D JSON:
      /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step10_lift3d/{tracker}/{model_tag}/seq_XXXX/frame_XXXXXX.json
  - PNG/GIF visualization under outputs/step10_lift3d_vis/...

Usage examples:
  # prior-only 리프팅
  python step10_lift3d_height_prior.py \
    --tracker sort --model_tag best --seq 0 \
    --priors_yaml /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step03_stats/class_size_priors.yaml \
    --anchor bottom --vis --gif

  # MiDaS 융합 사용 (Step09 결과 사용)
  python step10_lift3d_height_prior.py \
    --tracker sort --model_tag best --use_split /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step02_splits/val.txt \
    --priors_yaml /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step03_stats/class_size_priors.yaml \
    --midas_root /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step09_depth/dpt_hybrid \
    --use_midas_align --vis --gif
"""

import os
import re
import argparse
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import cv2

# ---------- Fixed defaults ----------
DEF_DATA_ROOT = "/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking"
DEF_TR_ROOT   = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step07_track2d"
DEF_OUT_JSON  = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step10_lift3d"
DEF_OUT_VIS   = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step10_lift3d_vis"
DEF_PRIORS    = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step03_stats/class_size_priors.yaml"
DEF_MIDAS     = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step09_depth/dpt_hybrid"

KITTI_FULL_SET = {"Car","Van","Truck","Tram","Pedestrian","Person","Cyclist","Misc"}

# ---------- Utils ----------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def list_seq_ids_from_tracks(tr_root_tag: str):
    seqs = [d for d in os.listdir(tr_root_tag) if d.startswith("seq_")]
    return sorted([s.split("_",1)[1] for s in seqs])

def list_frame_jsons(seq_tr_dir: str):
    return sorted([os.path.join(seq_tr_dir, f) for f in os.listdir(seq_tr_dir)
                   if f.startswith("frame_") and f.endswith(".json")])

def read_P2(calib_path: str):
    with open(calib_path, "r") as f:
        for ln in f:
            if ln.startswith("P2:"):
                vals = [float(x) for x in ln.split(":",1)[1].strip().split()]
                assert len(vals)==12, f"Malformed P2 in {calib_path}"
                return np.array(vals, dtype=np.float64).reshape(3,4)
    raise FileNotFoundError(f"P2 not found in {calib_path}")

def K_from_P2(P2):
    K = P2[:, :3].copy()
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    return K, float(fx), float(fy), float(cx), float(cy)

# ---------- Priors YAML (lightweight parser) ----------
def load_priors_yaml(path: str):
    """
    Parse class_size_priors.yaml created in Step03.
    Returns: {cls: {"H": float, "W": float, "L": float, "k_median": float or None, "count":int}}
    """
    priors = {}
    cur_cls = None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[ERR] priors yaml not found: {path}")
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            # class header: 'Car:' (2-space indented)
            m_cls = re.match(r"^\s{0,2}([A-Za-z_]+):\s*$", ln)
            if m_cls and m_cls.group(1) in KITTI_FULL_SET:
                cur_cls = m_cls.group(1)
                priors.setdefault(cur_cls, {"H":None,"W":None,"L":None,"k_median":None,"count":0})
                continue
            if cur_cls:
                m_num = re.match(r"^\s{2}([HWL]|k_median|count):\s*(.+)$", ln)
                if m_num:
                    k = m_num.group(1)
                    v = m_num.group(2).strip()
                    if v.lower() in ("null","none"):
                        priors[cur_cls][k] = None
                    else:
                        try:
                            priors[cur_cls][k] = float(v) if k!="count" else int(float(v))
                        except Exception:
                            pass
    return priors

# ---------- MiDaS helpers ----------
def load_midas_index(seq_id: str, midas_root: str):
    idx = os.path.join(midas_root, f"seq_{seq_id}", "index.json")
    if os.path.isfile(idx):
        with open(idx, "r") as f:
            return json.load(f)
    return None

def load_invdepth_for_frame(midas_index: dict, frame_idx: int):
    # prefer npy; fallback to depth16 png (0..65535) -> [0,1]
    if midas_index is None: return None
    recs = [r for r in midas_index.get("frames", []) if int(r["frame_index"])==int(frame_idx)]
    if not recs: return None
    r = recs[0]
    if r.get("invdepth_npy") and os.path.isfile(r["invdepth_npy"]):
        arr = np.load(r["invdepth_npy"])
        return arr.astype(np.float32)
    png = r.get("depth16_png")
    if png and os.path.isfile(png):
        u16 = cv2.imread(png, cv2.IMREAD_UNCHANGED)
        if u16 is None: return None
        return (u16.astype(np.float32) / 65535.0)
    return None

def sample_invdepth(invdepth: np.ndarray, l,t,r,b, mode="median", region="bottom15"):
    H, W = invdepth.shape[:2]
    l = int(max(0, np.floor(l))); t = int(max(0, np.floor(t)))
    r = int(min(W-1, np.ceil(r))); b = int(min(H-1, np.ceil(b)))
    if r<=l or b<=t: return None
    if region.startswith("bottom"):
        try:
            pct = int(region.replace("bottom",""))
        except Exception:
            pct = 15
        hh = max(1, int(np.round((b - t) * pct / 100.0)))
        y0 = max(t, b - hh)
        patch = invdepth[y0:b, l:r]
    else:
        patch = invdepth[t:b, l:r]
    vals = patch.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return None
    if mode == "mean":
        return float(np.mean(vals))
    else:
        return float(np.median(vals))

# ---------- Lifting ----------
def z_from_height_prior(hp: float, H_real: float, fy: float, k: float):
    hp = max(hp, 1.0)
    return float(k * fy * H_real / hp)

def backproject(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float):
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return float(X), float(Y), float(z)

def draw_overlay(img, det, xyz, dims, method, color=(0,255,255)):
    l,t,r,b = map(int, det["bbox"])
    u = int((l + r)/2)
    v = int(b)  # bottom center
    vis = img.copy()
    cv2.rectangle(vis, (l,t), (r,b), color, 2)
    cv2.circle(vis, (u,v), 3, (0,0,255), -1)
    x,y,z = xyz
    H,W,L = dims
    label = f"{det['class']} id#{det.get('id','-')} z={z:.1f}m x={x:.1f} y={y:.1f}"
    cv2.putText(vis, label, (l, max(0, t-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.putText(vis, f"{method}", (l, min(img.shape[0]-4, b+14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    # Required roots
    ap.add_argument("--kitti_root", default=DEF_DATA_ROOT, help="KITTI Tracking root (contains training/)")
    ap.add_argument("--tr_root", default=DEF_TR_ROOT, help="Step07 root")
    ap.add_argument("--tracker", choices=["sort","bytetrack"], required=True, help="tracker subdir")
    ap.add_argument("--model_tag", required=True, help="detector tag under tracker dir (same as Step07)")
    # Priors & MiDaS
    ap.add_argument("--priors_yaml", default=DEF_PRIORS, help="class_size_priors.yaml from Step03")
    ap.add_argument("--use_midas_align", action="store_true", help="align MiDaS inverse depth and fuse")
    ap.add_argument("--midas_root", default=DEF_MIDAS, help="Step09 depth root (e.g., .../step09_depth/dpt_hybrid)")
    # Selection
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seq", type=int, default=None, help="single seq id")
    g.add_argument("--use_split", type=str, default=None, help="text file with seq ids")
    # Options
    ap.add_argument("--anchor", choices=["bottom","center"], default="bottom", help="pixel anchor for backprojection")
    ap.add_argument("--min_hp", type=float, default=8.0, help="skip boxes with pixel height < min_hp")
    ap.add_argument("--z_clip_min", type=float, default=1.5)
    ap.add_argument("--z_clip_max", type=float, default=120.0)
    ap.add_argument("--classes", type=str, default=None, help="comma-separated class filter (default: all)")
    # Viz/GIF
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--gif_stride", type=int, default=5)
    ap.add_argument("--gif_max", type=int, default=200)
    ap.add_argument("--gif_duration", type=float, default=0.16)
    args = ap.parse_args()

    # Resolve dirs
    train_root = os.path.join(args.kitti_root, "training")
    img_root   = os.path.join(train_root, "image_02")
    calib_root = os.path.join(train_root, "calib")
    assert os.path.isdir(img_root), f"[ERR] not found: {img_root}"
    tr_root_tag = os.path.join(args.tr_root, args.tracker, args.model_tag)
    assert os.path.isdir(tr_root_tag), f"[ERR] track dir not found: {tr_root_tag}"

    # Seq list
    if args.seq is not None:
        seq_ids = [f"{args.seq:04d}"]
    elif args.use_split:
        with open(args.use_split, "r") as f:
            seq_ids = [ln.strip().zfill(4) for ln in f if ln.strip()]
    else:
        seq_ids = list_seq_ids_from_tracks(tr_root_tag)
    if not seq_ids:
        raise RuntimeError("[ERR] no sequences to process")

    # Priors
    pri = load_priors_yaml(args.priors_yaml)
    # Default/fallback k if missing: use global median over available
    k_vals = [v.get("k_median") for v in pri.values() if v.get("k_median") is not None]
    k_fallback = float(np.median(k_vals)) if k_vals else 1.0

    # Class filter
    if args.classes:
        keep_classes = {c.strip() for c in args.classes.split(",") if c.strip()}
    else:
        keep_classes = set(KITTI_FULL_SET)

    # Output dirs
    out_json_root = os.path.join(DEF_OUT_JSON, args.tracker, args.model_tag)
    out_vis_root  = os.path.join(DEF_OUT_VIS,  args.tracker, args.model_tag)
    ensure_dir(out_json_root)
    if args.vis:
        ensure_dir(out_vis_root)

    print(f"[INFO] lift3D: tracker={args.tracker}, model_tag={args.model_tag}, seqs={len(seq_ids)}, anchor={args.anchor}, use_midas={args.use_midas_align}")

    for sid in seq_ids:
        seq_tr_dir = os.path.join(tr_root_tag, f"seq_{sid}")
        assert os.path.isdir(seq_tr_dir), f"[ERR] missing track seq dir: {seq_tr_dir}"
        seq_img_dir = os.path.join(img_root, sid)
        calib_path = os.path.join(calib_root, f"{sid}.txt")
        P2 = read_P2(calib_path)
        K, fx, fy, cx, cy = K_from_P2(P2)

        # MiDaS index (optional)
        midas_idx = load_midas_index(sid, args.midas_root) if args.use_midas_align else None

        # Per-seq MiDaS scale 'a' (z ≈ a / invd), robust median over first ~50 valid samples
        a_scale = None
        if args.use_midas_align and midas_idx is not None:
            samples = []
            frame_files = list_frame_jsons(seq_tr_dir)[:200]  # inspect first up to 200 frames
            for fpath in frame_files:
                fobj = json.load(open(fpath, "r"))
                fi = int(fobj["frame_index"])
                invd = load_invdepth_for_frame(midas_idx, fi)
                if invd is None: continue
                for det in fobj.get("tracks", []):
                    cls = det.get("class","")
                    if cls not in keep_classes: continue
                    l,t,r,b = det["bbox"]
                    hp = max(1.0, float(b - t))
                    if hp < args.min_hp: continue
                    # priors
                    pr = pri.get(cls) or pri.get("Pedestrian" if cls=="Person" else None)
                    if not pr or pr.get("H") in (None,0): continue
                    H_real = float(pr["H"])
                    k_med  = float(pr.get("k_median") if pr.get("k_median") is not None else k_fallback)
                    z_p = z_from_height_prior(hp, H_real, fy, k_med)
                    # invdepth sample at bottom region
                    u = (l + r)/2.0
                    v = b if args.anchor=="bottom" else (t + b)/2.0
                    # small box around anchor
                    u0 = max(0, int(round(u - 3))); v0 = max(0, int(round(v - 6)))
                    u1 = min(invd.shape[1]-1, int(round(u + 3))); v1 = min(invd.shape[0]-1, int(round(v)))
                    if u1<=u0 or v1<=v0: continue
                    inv_s = np.median(invd[v0:v1, u0:u1])
                    if inv_s>0 and np.isfinite(inv_s) and z_p>0:
                        samples.append(z_p * inv_s)
                    if len(samples) >= 50:
                        break
                if len(samples) >= 50:
                    break
            if samples:
                a_scale = float(np.median(samples))
                print(f"[INFO] seq {sid}: MiDaS scale a ≈ {a_scale:.3f} (z ≈ a / invd)")
            else:
                print(f"[WARN] seq {sid}: not enough samples for MiDaS alignment; fallback to prior-only")
                a_scale = None

        # Output dirs per seq
        seq_out_json = os.path.join(out_json_root, f"seq_{sid}")
        ensure_dir(seq_out_json)
        if args.vis:
            seq_out_vis = os.path.join(out_vis_root, f"seq_{sid}")
            ensure_dir(seq_out_vis)
            gif_buf = []

        # Process frames
        frame_files = list_frame_jsons(seq_tr_dir)
        for i, fpath in enumerate(frame_files):
            fobj = json.load(open(fpath, "r"))
            fi = int(fobj["frame_index"])
            img_path = fobj.get("image_path") or os.path.join(seq_img_dir, f"{fi:06d}.png")
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] read fail: {img_path}")
                H_img, W_img = 375, 1242
            else:
                H_img, W_img = img.shape[:2]

            invd = None
            if args.use_midas_align and a_scale is not None:
                invd = load_invdepth_for_frame(midas_idx, fi)

            objs3d = []
            vis_img = img.copy() if img is not None else None

            for det in fobj.get("tracks", []):
                cls = det.get("class","")
                if cls == "Person": cls = "Pedestrian"  # normalize
                if cls not in keep_classes: continue
                l,t,r,b = map(float, det["bbox"])
                hp = max(1.0, b - t)
                if hp < args.min_hp:
                    continue

                pr = pri.get(cls)
                if not pr or pr.get("H") in (None, 0):
                    continue
                H_real = float(pr["H"])
                k_med  = float(pr.get("k_median") if pr.get("k_median") is not None else k_fallback)

                # z via height prior
                z_p = z_from_height_prior(hp, H_real, fy, k_med)

                # optional z via MiDaS
                z_use = z_p
                method = "prior_only"
                if invd is not None and a_scale is not None:
                    inv_s = sample_invdepth(invd, l,t,r,b, mode="median", region="bottom15")
                    if inv_s is not None and inv_s > 0:
                        z_m = float(a_scale / inv_s)
                        # fuse weights (hp-based)
                        w_p = np.clip(hp / 80.0, 0.1, 2.0)         # near(큰 hp) -> prior 가중 ↑
                        w_m = np.clip(80.0 / max(hp,1.0), 0.1, 2.0) # far(작은 hp) -> MiDaS 가중 ↑
                        z_f = (w_p * z_p + w_m * z_m) / (w_p + w_m)
                        z_use = float(z_f)
                        method = f"fused(wp={w_p:.2f},wm={w_m:.2f})"

                # clamp
                z_use = float(np.clip(z_use, args.z_clip_min, args.z_clip_max))

                # anchor pixel
                if args.anchor == "bottom":
                    u = (l + r) / 2.0
                    v = b
                else:
                    u = (l + r) / 2.0
                    v = (t + b) / 2.0

                X,Y,Z = backproject(u, v, z_use, fx, fy, cx, cy)
                dims = (float(H_real),
                        float(pr.get("W") if pr.get("W") is not None else 1.8),
                        float(pr.get("L") if pr.get("L") is not None else 4.0))

                obj3d = {
                    "id": int(det.get("id", -1)),
                    "class": cls,
                    "bbox": [l,t,r,b],
                    "hp": float(hp),
                    "xyz": [X,Y,Z],
                    "dims": {"H":dims[0], "W":dims[1], "L":dims[2]},
                    "z_prior": float(z_p),
                    "z_method": method
                }
                objs3d.append(obj3d)

                # vis
                if args.vis and vis_img is not None:
                    vis_img = draw_overlay(vis_img, det, (X,Y,Z), dims, method, color=(0,255,255))

            # save per-frame 3D JSON
            out_obj = {
                "sequence": sid,
                "frame_index": fi,
                "image_path": img_path,
                "K": {"fx":fx,"fy":fy,"cx":cx,"cy":cy},
                "objects3d": objs3d
            }
            out_path = os.path.join(seq_out_json, f"frame_{fi:06d}.json")
            with open(out_path, "w") as f:
                json.dump(out_obj, f, indent=2)

            if args.vis and vis_img is not None:
                seq_vis_dir = os.path.join(out_vis_root, f"seq_{sid}")
                ensure_dir(seq_vis_dir)
                out_png = os.path.join(seq_vis_dir, f"frame_{fi:06d}.png")
                cv2.imwrite(out_png, vis_img)
                # GIF buf
                if args.gif and ((i % args.gif_stride) == 0):
                    if "imageio" in globals() and imageio is not None:
                        gif_buf.append(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))

        # write index
        idx = {
            "sequence": sid,
            "tracker": args.tracker,
            "model_tag": args.model_tag,
            "priors_yaml": args.priors_yaml,
            "use_midas_align": bool(args.use_midas_align),
            "midas_root": (args.midas_root if args.use_midas_align else None),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(seq_out_json, "index.json"), "w") as f:
            json.dump(idx, f, indent=2)
        print(f"[OK] seq {sid}: lifted → {seq_out_json}")

        # save GIF
        if args.vis and args.gif and 'gif_buf' in locals() and gif_buf:
            gif_path = os.path.join(out_vis_root, f"sanity_seq_{sid}.gif")
            import imageio.v2 as imageio_local  # safe import
            imageio_local.mimsave(gif_path, gif_buf, duration=args.gif_duration)
            print(f"[OK] saved GIF: {gif_path}")

if __name__ == "__main__":
    main()
