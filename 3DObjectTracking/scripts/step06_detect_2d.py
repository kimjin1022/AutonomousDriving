#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 06 — 2D Detection on KITTI Tracking (camera only) with Ultralytics YOLO

- COCO 사전학습 모델(.pt) 또는 KITTI로 파인튜닝된 Ultralytics .pt 모두 지원
- 클래스 이름이 COCO이든 KITTI이든 자동/수동(coarse/full) 매핑 지원

Outputs (defaults):
  JSON: /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step06_det_cache/{model_tag}/seq_XXXX/frame_XXXXXX.json
  IDX : .../seq_XXXX/index.json
  (opt) PNG: /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step06_det_vis/{model_tag}/seq_XXXX/frame_XXXXXX.png
  (opt) GIF: /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step06_det_vis/{model_tag}/sanity_seq_XXXX.gif

Usage examples:
  # COCO 가중치
  python step06_detect_2d.py --seq 0 --weights yolov8n.pt --conf 0.25 --imgsz 1280 --vis --gif

  # KITTI로 파인튜닝된 Ultralytics .pt
  python step06_detect_2d.py --seq 0 --weights /path/to/your_kitti_best.pt --profile coarse --vis --gif

Requirements:
  pip install ultralytics opencv-python imageio
"""

import os
import argparse
from pathlib import Path
import json
from datetime import datetime

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

# ----- Fixed defaults -----
DEF_KITTI_ROOT = "/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking"
DEF_OUT_JSON   = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step06_det_cache"
DEF_OUT_VIS    = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step06_det_vis"

# ----- Colors (BGR) -----
CLASS_COLOR = {
    "Car": (0, 255, 0),
    "Pedestrian": (255, 0, 0),
    "Cyclist": (0, 165, 255),
    "Misc": (200, 200, 200)
}

# ----- Known class sets -----
COCO_SET = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench",
}
KITTI_FULL_SET = {"Car","Van","Truck","Tram","Pedestrian","Person","Cyclist","Misc"}
KITTI_COARSE_ALLOWED = {"Car","Pedestrian","Cyclist","Misc"}

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

# ---------- Mapping ----------
def detect_label_schema(names_obj) -> str:
    """
    names_obj: model.names (list or dict)
    return: "coco" or "kitti"
    """
    if isinstance(names_obj, dict):
        name_list = [str(v) for k, v in sorted(names_obj.items())]
    else:
        name_list = [str(v) for v in names_obj]
    lowers = {n.lower() for n in name_list}
    uppers = set(name_list)

    # if looks like KITTI
    if {"Car","Pedestrian","Cyclist"}.issubset(uppers) or {"car","van","truck","tram"}.issubset(lowers):
        return "kitti"
    # if looks like COCO
    if {"person","car","bicycle"}.issubset(lowers):
        return "coco"
    # fallback: treat as KITTI-like if it has Car/Ped/Cyc at least
    if {"Car","Pedestrian","Cyclist"}.issubset(uppers):
        return "kitti"
    return "coco"

def map_name_to_kitti(name: str, schema: str, profile: str, allow_misc: bool):
    """
    Normalize detector class name -> {Car, Pedestrian, Cyclist, Misc} (coarse)
    or preserve KITTI full classes if profile=='full'.
    """
    n_low = name.lower()
    n_raw = name

    if schema == "kitti":
        # KITTI name space
        if profile == "full":
            # keep full (normalize Person -> Pedestrian)
            if n_raw == "Person":
                return "Pedestrian"
            if n_raw in KITTI_FULL_SET:
                return n_raw
            return "Misc" if allow_misc else None
        else:  # coarse
            if n_raw in {"Car","Van","Truck","Tram"}:
                return "Car"
            if n_raw in {"Pedestrian","Person"}:
                return "Pedestrian"
            if n_raw == "Cyclist":
                return "Cyclist"
            return "Misc" if allow_misc else None

    # schema == "coco"
    if n_low in {"person"}:
        return "Pedestrian"
    if n_low in {"bicycle","motorcycle"}:
        return "Cyclist"
    if n_low in {"car","truck","bus","train"}:
        return "Car"
    # everything else
    return "Misc" if allow_misc else None

def draw_dets(img, dets):
    vis = img.copy()
    for d in dets:
        l, t, r, b = map(int, d["bbox"])
        cls = d["class"]
        score = d["score"]
        color = CLASS_COLOR.get(cls, (180, 180, 180))
        cv2.rectangle(vis, (l, t), (r, b), color, 2)
        cv2.putText(vis, f"{cls} {score:.2f}", (l, max(0, t-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", default=DEF_KITTI_ROOT,
                    help="KITTI Tracking root containing training/")
    ap.add_argument("--out_json_root", default=DEF_OUT_JSON,
                    help="Output root for detection JSON")
    ap.add_argument("--out_vis_root",  default=DEF_OUT_VIS,
                    help="Output root for visualization PNG/GIF")

    # target list
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seq", type=int, default=None, help="Process a single sequence id, e.g., 0")
    g.add_argument("--use_split", type=str, default=None, help="Text file with sequence ids (one per line)")

    # model
    ap.add_argument("--weights", type=str, default="yolov8n.pt", help="Ultralytics weights path/name (.pt)")
    ap.add_argument("--model_tag", type=str, default=None,
                    help="Tag name for out dir (default: inferred from weights filename)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference image size (short side)")
    ap.add_argument("--device", type=str, default=None, help="e.g., '0' or 'cpu'")

    # class mapping & vis
    ap.add_argument("--profile", choices=["coarse","full"], default="coarse",
                    help="KITTI class handling: coarse={Car/Ped/Cyc[/Misc]}, full=KITTI full set")
    ap.add_argument("--allow_misc", action="store_true", help="include unmapped classes as 'Misc'")
    ap.add_argument("--vis", action="store_true", help="save PNG visualizations")
    ap.add_argument("--gif", action="store_true", help="also save a per-seq GIF (needs --vis)")
    ap.add_argument("--gif_stride", type=int, default=5, help="use every Nth vis frame in GIF")
    ap.add_argument("--gif_max", type=int, default=200, help="max frames in GIF")
    args = ap.parse_args()

    # Load model
    if YOLO is None:
        raise RuntimeError("[ERR] Ultralytics not installed. Run: pip install ultralytics")

    # Friendly check for incompatible formats
    ext = os.path.splitext(args.weights)[1].lower()
    if ext not in {".pt", ""}:
        print(f"[WARN] weights extension '{ext}' may not be supported by Ultralytics. "
              f"Use a .pt (YOLOv5/v8) or switch to the external YOLOv3 wrapper.")
    try:
        model = YOLO(args.weights)
    except Exception as e:
        raise RuntimeError(
            f"[ERR] Failed to load weights with Ultralytics: {args.weights}\n"
            f"      If this is YOLOv3 (.weights/.pth) or non-Ultralytics .pt, "
            f"      use step06_detect_2d_kitti_yolov3.py instead.\n"
            f"      Original error: {e}"
        )
    names = model.names  # dict or list
    schema = detect_label_schema(names)
    print(f"[INFO] Detected label schema: {schema}")

    # Model tag (for out dir)
    tag = args.model_tag or os.path.splitext(os.path.basename(args.weights))[0]

    train_root = os.path.join(args.kitti_root, "training")
    img_root   = os.path.join(train_root, "image_02")
    calib_root = os.path.join(train_root, "calib")
    assert os.path.isdir(img_root), f"[ERR] not found: {img_root}"

    # sequence list
    if args.seq is not None:
        seq_ids = [f"{args.seq:04d}"]
    elif args.use_split:
        with open(args.use_split, "r") as f:
            seq_ids = [ln.strip().zfill(4) for ln in f if ln.strip()]
    else:
        seq_ids = list_seq_ids(img_root)
    if not seq_ids:
        raise RuntimeError("[ERR] no sequences to process")

    # Output dirs
    out_json_root = os.path.join(args.out_json_root, tag)
    ensure_dir(out_json_root)
    out_vis_root = os.path.join(args.out_vis_root, tag)
    if args.vis:
        ensure_dir(out_vis_root)

    print(f"[INFO] sequences: {len(seq_ids)}, weights={args.weights}, tag={tag}, conf={args.conf}, imgsz={args.imgsz}, profile={args.profile}")

    for sid in seq_ids:
        seq_img_dir = os.path.join(img_root, sid)
        assert os.path.isdir(seq_img_dir), f"[ERR] not found: {seq_img_dir}"
        seq_json_dir = os.path.join(out_json_root, f"seq_{sid}")
        ensure_dir(seq_json_dir)

        # list frames
        frame_idxs = list_frame_indices(seq_img_dir)
        if not frame_idxs:
            print(f"[WARN] no frames in {seq_img_dir}, skip.")
            continue

        # index meta
        index = {
            "sequence": sid,
            "image_dir": seq_img_dir,
            "calib_file": os.path.join(calib_root, f"{sid}.txt"),
            "model": tag,
            "weights": args.weights,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frames": [],
            "schema": schema,
            "profile": args.profile,
        }

        # vis prep
        seq_vis_dir = os.path.join(out_vis_root, f"seq_{sid}") if args.vis else None
        if args.vis:
            ensure_dir(seq_vis_dir)
            gif_candidates = []  # collected RGB frames for GIF

        for i, fi in enumerate(frame_idxs):
            img_path = os.path.join(seq_img_dir, f"{fi:06d}.png")
            if not os.path.isfile(img_path):
                continue

            res = model.predict(
                source=img_path,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False
            )
            r0 = res[0]
            dets = []
            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                conf = r0.boxes.conf.cpu().numpy()
                cls  = r0.boxes.cls.cpu().numpy().astype(int)

                # get class name by index
                def idx_to_name(ci: int) -> str:
                    if isinstance(names, (list, tuple)):
                        if 0 <= ci < len(names):
                            return str(names[ci])
                        return str(ci)
                    else:
                        return str(names.get(ci, ci))

                for b, s, c in zip(xyxy, conf, cls):
                    l, t, r, btm = [float(x) for x in b]
                    raw_name = idx_to_name(c)
                    mapped = map_name_to_kitti(raw_name, schema=schema, profile=args.profile, allow_misc=args.allow_misc)
                    if mapped is None:
                        continue
                    if args.profile == "coarse" and mapped not in {"Car","Pedestrian","Cyclist","Misc"}:
                        # safety clamp
                        continue
                    dets.append({
                        "bbox": [l, t, r, btm],
                        "score": float(s),
                        "class": mapped,
                        "class_raw": str(raw_name)
                    })

            frame_obj = {
                "sequence": sid,
                "frame_index": fi,
                "image_path": img_path,
                "detections": dets
            }
            out_json = os.path.join(seq_json_dir, f"frame_{fi:06d}.json")
            with open(out_json, "w") as f:
                json.dump(frame_obj, f, indent=2)
            index["frames"].append(os.path.basename(out_json))

            if args.vis:
                img = cv2.imread(img_path)
                vis_img = draw_dets(img, dets)
                out_png = os.path.join(seq_vis_dir, f"frame_{fi:06d}.png")
                cv2.imwrite(out_png, vis_img)
                if args.gif and (i % args.gif_stride == 0) and len(index["frames"]) <= args.gif_max:
                    if imageio is not None:
                        gif_candidates.append(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))

        with open(os.path.join(seq_json_dir, "index.json"), "w") as f:
            json.dump(index, f, indent=2)
        print(f"[OK] seq {sid}: saved {len(index['frames'])} JSONs → {seq_json_dir}")

        if args.vis and args.gif and imageio is not None and gif_candidates:
            gif_path = os.path.join(out_vis_root, f"sanity_seq_{sid}.gif")
            imageio.mimsave(gif_path, gif_candidates, duration=0.08)
            print(f"[OK] saved GIF: {gif_path}")

if __name__ == "__main__":
    main()
