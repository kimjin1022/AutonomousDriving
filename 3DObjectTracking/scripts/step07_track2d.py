#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 07 — 2D Tracking (SORT / ByteTrack-lite) for KITTI Tracking detections

Usage:
  python step07_track2d.py \
    --model_tag best \
    --seq 0 \
    --tracker sort \
    --det_conf 0.25 --iou_thresh 0.3 --max_age 20 --min_hits 3 \
    --vis --gif

  # val split 전체 + ByteTrack-lite (high/low 두 단계)
  python step07_track2d.py \
    --model_tag best \
    --use_split /home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step02_splits/val.txt \
    --tracker bytetrack \
    --conf_high 0.5 --conf_low 0.1 \
    --vis --gif

Inputs (from Step06):
  /home/.../outputs/step06_det_cache/{model_tag}/seq_XXXX/frame_XXXXXX.json
Each JSON has:
  {"sequence": "...", "frame_index": int, "image_path": "...", "detections":[{"bbox":[l,t,r,b],"score":float,"class": "..."}]}

Outputs:
  Tracks per frame in JSON (same bbox format), plus PNG/GIF visualization.

Notes:
  - Tracking is done per class to avoid cross-class ID switches.
  - SORT uses a simple Kalman filter (x,y,w,h, and velocities) + IoU matching.
  - ByteTrack-lite: first match high-score dets, then match remaining tracks with low-score dets (no new tracks from low).
"""

import os, json, argparse, math, glob
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

# ---- Fixed defaults ----
DEF_KITTI_ROOT = "/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training"
DEF_DET_ROOT   = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step06_det_cache"
DEF_OUT_ROOT   = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step07_track2d"
DEF_VIS_ROOT   = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step07_track2d_vis"

# KITTI full set (현재 Step06에서 이 셋으로 맵핑됨)
KITTI_FULL_SET = {"Car","Van","Truck","Tram","Pedestrian","Person","Cyclist","Misc"}

# ---------- Utilities ----------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def list_seq_ids_from_det(det_seq_root: str) -> List[str]:
    # det_seq_root = .../step06_det_cache/{model_tag}
    seq_dirs = sorted([d for d in os.listdir(det_seq_root) if d.startswith("seq_")])
    return [d.split("_",1)[1] for d in seq_dirs]

def list_frames_json(seq_det_dir: str) -> List[str]:
    # returns sorted list of frame_XXXXXX.json
    files = sorted(glob.glob(os.path.join(seq_det_dir, "frame_*.json")))
    return files

def load_det_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [l,t,r,b]
    l = max(a[0], b[0]); t = max(a[1], b[1])
    r = min(a[2], b[2]); btm = min(a[3], b[3])
    iw = max(0.0, r - l); ih = max(0.0, btm - t)
    inter = iw * ih
    area_a = max(0.0, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0]) * (b[3]-b[1]))
    union = area_a + area_b - inter + 1e-6
    return inter / union

def iou_matrix(tracks: List[np.ndarray], dets: List[np.ndarray]) -> np.ndarray:
    if len(tracks)==0 or len(dets)==0:
        return np.zeros((len(tracks), len(dets)), dtype=float)
    M = np.zeros((len(tracks), len(dets)), dtype=float)
    for i, tb in enumerate(tracks):
        for j, db in enumerate(dets):
            M[i,j] = iou_xyxy(tb, db)
    return M

def color_for_id(track_id: int) -> Tuple[int,int,int]:
    np.random.seed(track_id * 97 + 13)
    # BGR
    c = tuple(int(x) for x in np.random.randint(50, 230, size=3))
    return (c[0], c[1], c[2])

# ---------- Kalman Filter (simple) ----------
class KalmanBoxTracker:
    """
    Simple Kalman for (x,y,w,h) with constant velocity (vx,vy,vw,vh)
    State: [x,y,w,h,vx,vy,vw,vh]^T
    Measurement: [x,y,w,h]
    """
    _count = 0

    def __init__(self, bbox_xyxy: np.ndarray, score: float, cls: str):
        # convert xyxy -> x,y,w,h
        x = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
        y = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
        w = max(1.0, bbox_xyxy[2] - bbox_xyxy[0])
        h = max(1.0, bbox_xyxy[3] - bbox_xyxy[1])

        self.x = np.array([x, y, w, h, 0, 0, 0, 0], dtype=float).reshape(8,1)

        self.P = np.eye(8, dtype=float)
        self.P[4:,4:] *= 100.0  # high uncertainty on velocities

        self.F = np.eye(8, dtype=float)
        dt = 1.0
        for i in range(4):
            self.F[i, i+4] = dt

        self.H = np.zeros((4,8), dtype=float)
        self.H[0,0] = 1; self.H[1,1] = 1; self.H[2,2] = 1; self.H[3,3] = 1

        self.Q = np.eye(8, dtype=float) * 0.01
        self.R = np.eye(4, dtype=float) * 1.0

        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.score = float(score)
        self.cls = cls

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_state_xyxy()

    def update(self, bbox_xyxy: np.ndarray, score: float):
        # z = [x,y,w,h]
        z = np.zeros((4,1), dtype=float)
        z[0,0] = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
        z[1,0] = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
        z[2,0] = max(1.0, bbox_xyxy[2] - bbox_xyxy[0])
        z[3,0] = max(1.0, bbox_xyxy[3] - bbox_xyxy[1])

        # Kalman update
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)

    def get_state_xyxy(self) -> np.ndarray:
        x, y, w, h = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
        l = x - w/2.0; t = y - h/2.0; r = x + w/2.0; b = y + h/2.0
        return np.array([l, t, r, b], dtype=float)

# ---------- Multi-object tracker per class ----------
class MultiClassTracker2D:
    def __init__(self, tracker_type: str="sort",
                 iou_thresh: float=0.3, max_age: int=20, min_hits: int=3,
                 conf_det: float=0.25, conf_high: float=0.5, conf_low: float=0.1):
        self.tracker_type = tracker_type
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.conf_det = conf_det
        self.conf_high = conf_high
        self.conf_low  = conf_low

        # per-class holders
        self.tracks_by_cls: Dict[str, List[KalmanBoxTracker]] = {}
        self.next_id_by_cls: Dict[str, int] = {}

    def _assign(self, track_boxes: List[np.ndarray], det_boxes: List[np.ndarray]) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        """
        Return matches (ti, dj), unmatched_tracks, unmatched_dets
        Matching by maximizing IoU with threshold.
        """
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return [], list(range(len(track_boxes))), list(range(len(det_boxes)))

        iou = iou_matrix(track_boxes, det_boxes)
        cost = 1.0 - iou

        if _HAS_SCIPY:
            ti, dj = linear_sum_assignment(cost)
            pairs = []
            unmatched_t = set(range(len(track_boxes)))
            unmatched_d = set(range(len(det_boxes)))
            for i, j in zip(ti, dj):
                if iou[i, j] >= self.iou_thresh:
                    pairs.append((i, j))
                    unmatched_t.discard(i)
                    unmatched_d.discard(j)
            return pairs, sorted(list(unmatched_t)), sorted(list(unmatched_d))
        else:
            # Greedy by IoU
            pairs = []
            used_t = set(); used_d = set()
            cand = []
            for i in range(iou.shape[0]):
                for j in range(iou.shape[1]):
                    cand.append((iou[i,j], i, j))
            cand.sort(reverse=True)
            for val, i, j in cand:
                if val < self.iou_thresh:
                    break
                if (i in used_t) or (j in used_d):
                    continue
                pairs.append((i,j))
                used_t.add(i); used_d.add(j)
            unmatched_t = [i for i in range(iou.shape[0]) if i not in used_t]
            unmatched_d = [j for j in range(iou.shape[1]) if j not in used_d]
            return pairs, unmatched_t, unmatched_d

    def _get_track_boxes(self, tracks: List[KalmanBoxTracker]) -> List[np.ndarray]:
        return [trk.get_state_xyxy() for trk in tracks]

    def step_sort(self, cls_name: str, dets: List[Tuple[np.ndarray, float]]) -> List[Dict]:
        """
        Simple SORT step for a single class.
        dets: list of (bbox_xyxy, score) with score >= conf_det
        """
        tracks = self.tracks_by_cls.setdefault(cls_name, [])

        # 1) Predict all tracks
        for trk in tracks:
            trk.predict()

        # 2) Associate by IoU
        track_boxes = self._get_track_boxes(tracks)
        det_boxes = [d[0] for d in dets]
        det_scores = [d[1] for d in dets]

        matches, unmatched_t, unmatched_d = self._assign(track_boxes, det_boxes)

        # 3) Update matched
        for ti, dj in matches:
            tracks[ti].update(det_boxes[dj], det_scores[dj])

        # 4) Create new tracks for unmatched dets
        new_tracks = []
        for j in unmatched_d:
            trk = KalmanBoxTracker(det_boxes[j], det_scores[j], cls_name)
            new_tracks.append(trk)
        tracks.extend(new_tracks)

        # 5) Manage track ages; remove dead
        alive_tracks = []
        outputs = []
        for i, trk in enumerate(tracks):
            if i in [ti for ti,_ in matches]:
                pass
            else:
                trk.time_since_update += 0  # already incremented in predict()

            # Prepare output if valid
            bbox = trk.get_state_xyxy()
            # Only output tracks that have enough hits or are recently updated
            if (trk.hits >= self.min_hits) or (trk.time_since_update == 0):
                outputs.append({
                    "id": int(trk.id),
                    "class": trk.cls,
                    "bbox": [float(v) for v in bbox],
                    "score": float(trk.score)
                })

            # keep if not too old
            if trk.time_since_update <= self.max_age:
                alive_tracks.append(trk)
        self.tracks_by_cls[cls_name] = alive_tracks
        return outputs

    def step_bytetrack(self, cls_name: str, dets_all: List[Tuple[np.ndarray, float]]) -> List[Dict]:
        """
        ByteTrack-lite step for a single class:
          - Split dets into high (>= conf_high) and low ([conf_low, conf_high))
          - Assoc tracks with high first (create new from high unmatched)
          - Then assoc remaining tracks with low (no new tracks from low)
        """
        tracks = self.tracks_by_cls.setdefault(cls_name, [])

        # Predict
        for trk in tracks:
            trk.predict()

        # Split dets
        high = [(b,s) for (b,s) in dets_all if s >= self.conf_high]
        low  = [(b,s) for (b,s) in dets_all if (self.conf_low <= s < self.conf_high)]

        # Assoc with high
        tb = self._get_track_boxes(tracks)
        db = [d[0] for d in high]
        ds = [d[1] for d in high]
        matches, unmatched_t, unmatched_d = self._assign(tb, db)
        for ti, dj in matches:
            tracks[ti].update(db[dj], ds[dj])

        # Create new from unmatched high
        for j in unmatched_d:
            trk = KalmanBoxTracker(db[j], ds[j], cls_name)
            tracks.append(trk)

        # Assoc remaining tracks with low (no new)
        # Gather remaining track boxes
        remain_idx = [i for i in unmatched_t]  # still unmatched after high
        if remain_idx and len(low) > 0:
            tb2 = [tb[i] for i in remain_idx]
            db2 = [d[0] for d in low]
            ds2 = [d[1] for d in low]
            matches2, unmatched_t2, unmatched_d2 = self._assign(tb2, db2)
            for k, j in matches2:  # k is index in remain_idx
                ti = remain_idx[k]
                tracks[ti].update(db2[j], ds2[j])
            # unmatched low are discarded

        # Output + cleanup
        alive_tracks = []
        outputs = []
        for trk in tracks:
            bbox = trk.get_state_xyxy()
            if (trk.hits >= self.min_hits) or (trk.time_since_update == 0):
                outputs.append({
                    "id": int(trk.id),
                    "class": trk.cls,
                    "bbox": [float(v) for v in bbox],
                    "score": float(trk.score)
                })
            if trk.time_since_update <= self.max_age:
                alive_tracks.append(trk)
        self.tracks_by_cls[cls_name] = alive_tracks
        return outputs

    def step(self, detections: List[Dict]) -> List[Dict]:
        """
        detections: list dict with keys: bbox [l,t,r,b], score, class
        Returns list of track dicts for this frame across all classes.
        """
        outputs = []
        # Group by class
        by_cls: Dict[str, List[Tuple[np.ndarray,float]]] = {}
        for d in detections:
            cls = d["class"]
            if cls not in KITTI_FULL_SET:
                continue
            s = float(d.get("score", 0.0))
            if self.tracker_type == "sort":
                if s < self.conf_det:
                    continue
            else:  # bytetrack
                if s < self.conf_low:
                    continue
            box = np.array(d["bbox"], dtype=float)
            by_cls.setdefault(cls, []).append((box, s))

        # For each class, run tracker
        for cls_name, dets in by_cls.items():
            if self.tracker_type == "sort":
                out = self.step_sort(cls_name, dets)
            else:
                out = self.step_bytetrack(cls_name, dets)
            outputs.extend(out)

        # Also advance classes with no detections (keep predicting & aging)
        for cls_name in KITTI_FULL_SET:
            if cls_name not in by_cls:
                _ = self.tracks_by_cls.setdefault(cls_name, [])
                # Predict + cleanup so that old tracks die out
                preds = []
                for trk in self.tracks_by_cls[cls_name]:
                    trk.predict()
                alive = []
                for trk in self.tracks_by_cls[cls_name]:
                    bbox = trk.get_state_xyxy()
                    if (trk.hits >= self.min_hits) or (trk.time_since_update == 0):
                        outputs.append({
                            "id": int(trk.id),
                            "class": trk.cls,
                            "bbox": [float(v) for v in bbox],
                            "score": float(trk.score)
                        })
                    if trk.time_since_update <= self.max_age:
                        alive.append(trk)
                self.tracks_by_cls[cls_name] = alive

        return outputs

# ---------- Visualization ----------
def draw_tracks(img, tracks: List[Dict], thickness=2):
    vis = img.copy()
    for tr in tracks:
        l,t,r,b = map(int, tr["bbox"])
        tid = tr["id"]
        cls = tr["class"]
        color = color_for_id(tid)
        cv2.rectangle(vis, (l,t), (r,b), color, thickness)
        label = f"{cls} #{tid}"
        cv2.putText(vis, label, (l, max(0, t-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return vis

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_train_root", default=DEF_KITTI_ROOT,
                    help=".../datasets/KITTI_Tracking/training (for images)")
    ap.add_argument("--det_root", default=DEF_DET_ROOT,
                    help="root of detection JSON cache (from Step06)")
    ap.add_argument("--model_tag", required=True,
                    help="subdir under det_root to read, e.g., 'best' (from Step06)")
    ap.add_argument("--out_root", default=DEF_OUT_ROOT,
                    help="output root for track JSON")
    ap.add_argument("--out_vis_root", default=DEF_VIS_ROOT,
                    help="output root for visualization (PNG/GIF)")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seq", type=int, default=None, help="single seq id, e.g., 0")
    g.add_argument("--use_split", type=str, default=None, help="text file with seq ids")

    ap.add_argument("--tracker", choices=["sort","bytetrack"], default="sort")
    ap.add_argument("--iou_thresh", type=float, default=0.3)
    ap.add_argument("--max_age", type=int, default=20)
    ap.add_argument("--min_hits", type=int, default=3)

    # SORT
    ap.add_argument("--det_conf", type=float, default=0.25, help="min det conf for SORT")

    # ByteTrack-lite
    ap.add_argument("--conf_high", type=float, default=0.5, help="high conf threshold")
    ap.add_argument("--conf_low",  type=float, default=0.1, help="low conf threshold")

    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--gif_stride", type=int, default=5)
    ap.add_argument("--gif_max", type=int, default=200)

    args = ap.parse_args()

    # resolve dirs
    det_root_tag = os.path.join(args.det_root, args.model_tag)
    assert os.path.isdir(det_root_tag), f"[ERR] det dir not found: {det_root_tag}"

    img_root = os.path.join(args.kitti_train_root, "image_02")
    assert os.path.isdir(img_root), f"[ERR] image root not found: {img_root}"

    out_json_root = os.path.join(args.out_root, args.tracker, args.model_tag)
    out_vis_root  = os.path.join(args.out_vis_root, args.tracker, args.model_tag)
    ensure_dir(out_json_root)
    if args.vis:
        ensure_dir(out_vis_root)

    # seq list
    if args.seq is not None:
        seq_ids = [f"{args.seq:04d}"]
    elif args.use_split:
        with open(args.use_split, "r") as f:
            seq_ids = [ln.strip().zfill(4) for ln in f if ln.strip()]
    else:
        seq_ids = list_seq_ids_from_det(det_root_tag)
    if not seq_ids:
        raise RuntimeError("[ERR] no sequences found to process")

    print(f"[INFO] tracker={args.tracker}, det_tag={args.model_tag}, seqs={len(seq_ids)}")

    # tracker instance
    mot = MultiClassTracker2D(
        tracker_type=args.tracker,
        iou_thresh=args.iou_thresh,
        max_age=args.max_age,
        min_hits=args.min_hits,
        conf_det=args.det_conf,
        conf_high=args.conf_high,
        conf_low=args.conf_low
    )

    for sid in seq_ids:
        seq_det_dir = os.path.join(det_root_tag, f"seq_{sid}")
        assert os.path.isdir(seq_det_dir), f"[ERR] det seq dir not found: {seq_det_dir}"

        frame_jsons = list_frames_json(seq_det_dir)
        if not frame_jsons:
            print(f"[WARN] no frame jsons in {seq_det_dir}, skip.")
            continue

        # output dirs
        seq_out_json = os.path.join(out_json_root, f"seq_{sid}")
        ensure_dir(seq_out_json)

        if args.vis:
            seq_vis_dir = os.path.join(out_vis_root, f"seq_{sid}")
            ensure_dir(seq_vis_dir)
            gif_buf = []

        index = {
            "sequence": sid,
            "det_source": seq_det_dir,
            "frames": [],
            "tracker": args.tracker,
            "params": {
                "iou_thresh": args.iou_thresh,
                "max_age": args.max_age,
                "min_hits": args.min_hits,
                "det_conf": args.det_conf,
                "conf_high": args.conf_high,
                "conf_low": args.conf_low,
            }
        }

        for i, fjson in enumerate(frame_jsons):
            data = load_det_json(fjson)
            frame_idx = int(data["frame_index"])
            img_path  = data["image_path"]
            dets_in   = data.get("detections", [])

            # step
            tracks_out = mot.step(dets_in)

            # save per-frame tracks
            out_obj = {
                "sequence": sid,
                "frame_index": frame_idx,
                "image_path": img_path,
                "tracks": tracks_out
            }
            out_path = os.path.join(seq_out_json, f"frame_{frame_idx:06d}.json")
            with open(out_path, "w") as f:
                json.dump(out_obj, f, indent=2)
            index["frames"].append(os.path.basename(out_path))

            # vis
            if args.vis:
                img = cv2.imread(img_path)
                if img is None:
                    # fallback: try KITTI train image dir
                    img_path2 = os.path.join(img_root, sid, f"{frame_idx:06d}.png")
                    img = cv2.imread(img_path2)
                vis_img = draw_tracks(img, tracks_out)
                out_png = os.path.join(seq_vis_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(out_png, vis_img)

                if args.gif and (i % args.gif_stride == 0) and len(gif_buf) < args.gif_max and imageio is not None:
                    gif_buf.append(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))

        # save index
        with open(os.path.join(seq_out_json, "index.json"), "w") as f:
            json.dump(index, f, indent=2)
        print(f"[OK] seq {sid}: tracked {len(index['frames'])} frames → {seq_out_json}")

        if args.vis and args.gif and imageio is not None and gif_buf:
            gif_path = os.path.join(out_vis_root, f"sanity_seq_{sid}.gif")
            imageio.mimsave(gif_path, gif_buf, duration=0.1) 
            print(f"[OK] saved GIF: {gif_path}")

if __name__ == "__main__":
    main()
