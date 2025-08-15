#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 08 — 2D Evaluation (MOTA/MOTP/IDF1 [+ optional HOTA via trackeval])

Inputs:
  GT (Step05):  /.../outputs/step05_gt_cache/seq_XXXX/frame_XXXXXX.json
  TR (Step07):  /.../outputs/step07_track2d/{tracker}/{model_tag}/seq_XXXX/frame_XXXXXX.json

Outputs:
  /.../outputs/step08_metrics_2d/{tracker}/{model_tag}/
    - summary.json        # overall + per-class metrics
    - metrics_per_seq.csv # per-sequence table

Install (optional):
  pip install scipy       # Hungarian
  pip install trackeval   # to compute HOTA with --use_trackeval
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import csv
import math
import numpy as np

# ---------- Optional deps ----------
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import trackeval
    _HAS_TRACKEVAL = True
except Exception:
    _HAS_TRACKEVAL = False

# ---------- Defaults ----------
DEF_GT_ROOT   = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step05_gt_cache"
DEF_TR_ROOT   = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step07_track2d"
DEF_OUT_ROOT  = "/home/jinjinjara1022/AutonomousDriving/3DObjectTracking/outputs/step08_metrics_2d"

KITTI_FULL_SET = {"Car","Van","Truck","Tram","Pedestrian","Person","Cyclist","Misc"}

# ---------- IO utils ----------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def list_seq_ids_from_tracks(tr_root_tag: str):
    # tr_root_tag = .../step07_track2d/{tracker}/{model_tag}
    seq_dirs = sorted([d for d in os.listdir(tr_root_tag) if d.startswith("seq_")])
    return [d.split("_",1)[1] for d in seq_dirs]

def list_frame_jsons(seq_tr_dir: str):
    return sorted([os.path.join(seq_tr_dir, f) for f in os.listdir(seq_tr_dir)
                   if f.startswith("frame_") and f.endswith(".json")])

# ---------- Box utils ----------
def iou_xyxy(a, b) -> float:
    l = max(a[0], b[0]); t = max(a[1], b[1])
    r = min(a[2], b[2]); btm = min(a[3], b[3])
    iw = max(0.0, r - l); ih = max(0.0, btm - t)
    inter = iw * ih
    ua = max(0.0, (a[2]-a[0])*(a[3]-a[1]))
    ub = max(0.0, (b[2]-b[0])*(b[3]-b[1]))
    union = ua + ub - inter + 1e-6
    return inter / union

def build_iou_matrix(gt_boxes, tr_boxes):
    if len(gt_boxes)==0 or len(tr_boxes)==0:
        return np.zeros((len(gt_boxes), len(tr_boxes)), dtype=float)
    M = np.zeros((len(gt_boxes), len(tr_boxes)), dtype=float)
    for i, g in enumerate(gt_boxes):
        for j, d in enumerate(tr_boxes):
            M[i,j] = iou_xyxy(g, d)
    return M
def match_by_iou(gt_boxes, tr_boxes, thr=0.5):
    """Return (matches, unmatched_gt, unmatched_tr, iou_matrix)"""
    # 항상 iou 행렬을 먼저 만들고 빈 경우 처리
    iou = build_iou_matrix(gt_boxes, tr_boxes)
    if len(gt_boxes) == 0 or len(tr_boxes) == 0 or iou.size == 0:
        return [], list(range(len(gt_boxes))), list(range(len(tr_boxes))), iou

    cost = 1.0 - iou
    matches = []
    if _HAS_SCIPY:
        gi, tj = linear_sum_assignment(cost)
        for i, j in zip(gi, tj):
            if iou[i, j] >= thr:
                matches.append((i, j))
    else:
        cand = []
        for i in range(iou.shape[0]):
            for j in range(iou.shape[1]):
                cand.append((iou[i, j], i, j))
        cand.sort(reverse=True)
        used_g, used_t = set(), set()
        for val, i, j in cand:
            if val < thr:
                break
            if i in used_g or j in used_t:
                continue
            matches.append((i, j))
            used_g.add(i); used_t.add(j)

    matched_g = {i for i, _ in matches}
    matched_t = {j for _, j in matches}
    um_g = [i for i in range(len(gt_boxes)) if i not in matched_g]
    um_t = [j for j in range(len(tr_boxes)) if j not in matched_t]
    return matches, um_g, um_t, iou


# ---------- Metrics aggregators ----------
class SeqMetrics:
    def __init__(self, classes, iou_thr):
        self.classes = classes
        self.iou_thr = iou_thr
        # per-class counts
        self.tp = Counter()
        self.fp = Counter()
        self.fn = Counter()
        self.idsw = Counter()
        self.iou_sum = Counter()
        self.match_cnt = Counter()
        self.gt_total = Counter()
        self.tr_total = Counter()
        # for IDF1 contingency (per class)
        self.contingency = {c: defaultdict(int) for c in classes}  # key: (gt_id, tr_id) -> count
        # track->prev_gt for IDSW
        self.prev_gt_for_tr = {c: {} for c in classes}

    def add_frame(self, cls, gt_list, tr_list, matches, iou_mat):
        # counts
        self.tp[cls] += len(matches)
        self.fp[cls] += (len(tr_list) - len(matches))
        self.fn[cls] += (len(gt_list) - len(matches))
        self.gt_total[cls] += len(gt_list)
        self.tr_total[cls] += len(tr_list)
        # IoU sum
        for gi, tj in matches:
            self.iou_sum[cls] += float(iou_mat[gi, tj])
            self.match_cnt[cls] += 1
            gt_id = gt_list[gi]["id"]
            tr_id = tr_list[tj]["id"]
            self.contingency[cls][(gt_id, tr_id)] += 1
            # ID switches
            prev = self.prev_gt_for_tr[cls].get(tr_id, None)
            if prev is not None and prev != gt_id:
                # consider switch only if track was matched in consecutive presence
                self.idsw[cls] += 1
            self.prev_gt_for_tr[cls][tr_id] = gt_id

    def _idf1_for_class(self, cls):
        # Build bipartite matching on counts to get IDTP
        # Rows: gt ids; Cols: tr ids
        # Collect unique ids
        pairs = list(self.contingency[cls].items())  # [((gt,tr), count), ...]
        if not pairs:
            return 0, 0, 0, 0.0
        gt_ids = sorted({gt for (gt, _), _ in pairs})
        tr_ids = sorted({tr for (_, tr), _ in pairs})
        gi = {g:i for i,g in enumerate(gt_ids)}
        tj = {t:j for j,t in enumerate(tr_ids)}
        M = np.zeros((len(gt_ids), len(tr_ids)), dtype=float)
        for (g,t), c in pairs:
            M[gi[g], tj[t]] = float(c)
        if _HAS_SCIPY:
            # maximize counts
            maxv = M.max() if M.size>0 else 0.0
            cost = (maxv - M)
            row_ind, col_ind = linear_sum_assignment(cost)
            idtp = float(sum(M[row_ind[k], col_ind[k]] for k in range(len(row_ind))))
        else:
            # greedy on counts
            cand = []
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    cand.append((M[i,j], i, j))
            cand.sort(reverse=True)
            used_r=set(); used_c=set(); idtp=0.0
            for v,i,j in cand:
                if v<=0: break
                if i in used_r or j in used_c: continue
                idtp += v; used_r.add(i); used_c.add(j)

        total_tr = float(self.tr_total[cls])
        total_gt = float(self.gt_total[cls])
        idfp = max(0.0, total_tr - idtp)
        idfn = max(0.0, total_gt - idtp)
        denom = 2.0*idtp + idfp + idfn
        idf1 = (2.0*idtp/denom) if denom>0 else 0.0
        return idtp, idfp, idfn, idf1

    def summarize(self):
        per_cls = {}
        overall = {
            "MOTA": 0.0, "MOTP": 0.0, "IDF1": 0.0,
            "TP": 0, "FP": 0, "FN": 0, "IDSW": 0,
            "GT": 0, "TR": 0, "Matches": 0, "IoU_sum": 0.0
        }
        # per-class compute
        for c in self.classes:
            tp = self.tp[c]; fp = self.fp[c]; fn = self.fn[c]
            idsw = self.idsw[c]
            gt = self.gt_total[c]; tr = self.tr_total[c]
            match_cnt = self.match_cnt[c]; iou_sum = self.iou_sum[c]
            mota = 1.0 - ((fp + fn + idsw) / gt) if gt>0 else 0.0
            motp = (iou_sum / match_cnt) if match_cnt>0 else 0.0
            idtp, idfp, idfn, idf1 = self._idf1_for_class(c)
            per_cls[c] = {
                "MOTA": round(mota,4), "MOTP": round(motp,4), "IDF1": round(idf1,4),
                "TP": int(tp), "FP": int(fp), "FN": int(fn), "IDSW": int(idsw),
                "GT": int(gt), "TR": int(tr), "Matches": int(match_cnt)
            }
            # overall accumulate raw
            overall["TP"] += int(tp); overall["FP"] += int(fp); overall["FN"] += int(fn)
            overall["IDSW"] += int(idsw); overall["GT"] += int(gt); overall["TR"] += int(tr)
            overall["Matches"] += int(match_cnt); overall["IoU_sum"] += float(iou_sum)
        # overall metrics
        if overall["GT"] > 0:
            overall["MOTA"] = round(1.0 - ((overall["FP"] + overall["FN"] + overall["IDSW"]) / overall["GT"]), 4)
        overall["MOTP"] = round((overall["IoU_sum"]/overall["Matches"]) if overall["Matches"]>0 else 0.0, 4)
        # overall IDF1 via pooled contingency
        pooled = SeqMetrics(classes=["_all_"], iou_thr=self.iou_thr)
        # merge contingency
        for c in self.classes:
            for (g,t), cnt in self.contingency[c].items():
                pooled.contingency["_all_"][(("c",c,g), ("t",c,t))] += cnt
            pooled.tr_total["_all_"] += self.tr_total[c]
            pooled.gt_total["_all_"] += self.gt_total[c]
        idtp, idfp, idfn, idf1 = pooled._idf1_for_class("_all_")
        overall["IDF1"] = round(idf1, 4)
        return per_cls, overall

# ---------- TrackEval (optional HOTA) ----------
def maybe_run_trackeval_hota(gt_root, tr_seq_dir_by_sid, out_dir, iou_thr=0.5):
    if not _HAS_TRACKEVAL:
        print("[INFO] trackeval not installed; skip HOTA")
        return None
    # Build minimal in-memory dataset for trackeval's API
    # We will use the 'Raw dataset' interface.
    try:
        from trackeval.datasets._base_dataset import _BaseDataset
        from trackeval.metrics import HOTA as HOTA_Metric
    except Exception as e:
        print("[WARN] trackeval API changed or not available:", e)
        return None

    class RawSeqDataset(_BaseDataset):
        def __init__(self, gt_root, tr_seq_dir_by_sid, iou_thr):
            self.gt_data = {}
            self.tr_data = {}
            self.seq_list = []
            self.output_fol = out_dir
            self.use_super_categories = False
            self.do_preproc = False
            self.iou_type = 'bbox'
            self.class_list = ['all']
            self.ignore_region = False
            self.seq_list = sorted(tr_seq_dir_by_sid.keys())
            self.iou_thresholds = np.array([iou_thr], dtype=float)
            # Load per sequence frames
            for sid in self.seq_list:
                seq = f"seq_{sid}"
                # frames list from tracker dir
                tr_frames = [f for f in os.listdir(tr_seq_dir_by_sid[sid]) if f.startswith("frame_") and f.endswith(".json")]
                tr_frames = sorted(tr_frames)
                gt_seq_dir = os.path.join(gt_root, f"seq_{sid}")
                gt_frames = [f"frame_{int(os.path.splitext(os.path.basename(fp))[0].split('_')[1]):06d}.json" for fp in tr_frames]
                # build arrays
                gt_ids_all=[]; gt_boxes_all=[]; tr_ids_all=[]; tr_boxes_all=[]
                gt_frame_inds=[]; tr_frame_inds=[]
                fidx = 0
                for fn in tr_frames:
                    frame_idx = int(fn.split("_")[1].split(".")[0])
                    # GT
                    gpath = os.path.join(gt_seq_dir, f"frame_{frame_idx:06d}.json")
                    if not os.path.isfile(gpath): 
                        # empty GT
                        gobjs=[]
                    else:
                        gobjs = load_json(gpath).get("objects", [])
                    gt_ids=[]; gt_boxes=[]
                    for g in gobjs:
                        cls = g.get("class","")
                        if cls not in KITTI_FULL_SET: continue
                        gt_ids.append(int(g["id"]))
                        l,t,r,b = g["bbox"]
                        gt_boxes.append([l,t,r,b])
                    # TR
                    tpath = os.path.join(tr_seq_dir_by_sid[sid], fn)
                    tobjs = load_json(tpath).get("tracks", [])
                    tr_ids=[]; tr_boxes=[]
                    for tr in tobjs:
                        cls = tr.get("class","")
                        if cls not in KITTI_FULL_SET: continue
                        tr_ids.append(int(tr["id"]))
                        l,t,r,b = tr["bbox"]
                        tr_boxes.append([l,t,r,b])

                    gt_ids_all.append(np.asarray(gt_ids, dtype=int))
                    gt_boxes_all.append(np.asarray(gt_boxes, dtype=float))
                    tr_ids_all.append(np.asarray(tr_ids, dtype=int))
                    tr_boxes_all.append(np.asarray(tr_boxes, dtype=float))
                    gt_frame_inds.append(fidx); tr_frame_inds.append(fidx)
                    fidx += 1

                self.gt_data[seq] = {'ids': gt_ids_all, 'boxes': gt_boxes_all, 'frame_nums': np.array(gt_frame_inds)}
                self.tr_data[seq] = {'ids': tr_ids_all, 'boxes': tr_boxes_all, 'frame_nums': np.array(tr_frame_inds)}

        def get_raw_seq_data(self, seq):
            return self.gt_data[seq], self.tr_data[seq]

    try:
        dataset = RawSeqDataset(gt_root, tr_seq_dir_by_sid, iou_thr)
        hota_metric = HOTA_Metric()
        hota_res = {}
        for sid in dataset.seq_list:
            seq = f"seq_{sid}"
            gt, tr = dataset.get_raw_seq_data(seq)
            res = hota_metric.eval_sequence({}, gt, tr)
            hota_res[sid] = {k: float(v[0]) if isinstance(v, np.ndarray) else float(v) for k,v in res.items() if isinstance(v,(np.ndarray,float,int))}
        # overall average HOTA
        if hota_res:
            overall = {k: float(np.mean([v.get(k,0.0) for v in hota_res.values()])) for k in hota_res[next(iter(hota_res))].keys()}
        else:
            overall = {}
        out_path = os.path.join(out_dir, "trackeval_hota.json")
        with open(out_path, "w") as f:
            json.dump({"per_seq": hota_res, "overall_mean": overall}, f, indent=2)
        print(f"[OK] trackeval HOTA saved: {out_path}")
        return {"per_seq": hota_res, "overall_mean": overall}
    except Exception as e:
        print("[WARN] trackeval HOTA failed:", e)
        return None

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", default=DEF_GT_ROOT, help="Step05 GT cache root")
    ap.add_argument("--tr_root", default=DEF_TR_ROOT, help="Step07 track root")
    ap.add_argument("--tracker", choices=["sort","bytetrack"], default="sort", help="subdir under tr_root")
    ap.add_argument("--model_tag", required=True, help="subdir under tr_root/tracker")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seq", type=int, default=None, help="single sequence id")
    g.add_argument("--use_split", type=str, default=None, help="text file with seq ids")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    ap.add_argument("--classes", type=str, default=None,
                    help='comma-separated class filter (default: all KITTI_FULL_SET)')
    ap.add_argument("--out_root", default=DEF_OUT_ROOT, help="output root for metrics")
    ap.add_argument("--use_trackeval", action="store_true", help="also compute HOTA if trackeval installed")
    args = ap.parse_args()

    # Resolve dirs
    tr_root_tag = os.path.join(args.tr_root, args.tracker, args.model_tag)
    assert os.path.isdir(tr_root_tag), f"[ERR] track dir not found: {tr_root_tag}"
    assert os.path.isdir(args.gt_root), f"[ERR] gt root not found: {args.gt_root}"

    # Classes
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        for c in classes:
            if c not in KITTI_FULL_SET:
                raise ValueError(f"Unknown class '{c}'. Valid: {sorted(KITTI_FULL_SET)}")
    else:
        classes = sorted(KITTI_FULL_SET)

    # Seq list
    if args.seq is not None:
        seq_ids = [f"{args.seq:04d}"]
    elif args.use_split:
        with open(args.use_split, "r") as f:
            seq_ids = [ln.strip().zfill(4) for ln in f if ln.strip()]
    else:
        seq_ids = list_seq_ids_from_tracks(tr_root_tag)
    if not seq_ids:
        raise RuntimeError("[ERR] no sequences to evaluate")

    # Output dir
    out_dir = os.path.join(args.out_root, args.tracker, args.model_tag)
    ensure_dir(out_dir)

    # Per-seq metrics collector
    per_seq_rows = []

    # For optional HOTA, remember tracker seq dirs
    tr_seq_dir_by_sid = {}

    for sid in seq_ids:
        seq_tr_dir = os.path.join(tr_root_tag, f"seq_{sid}")
        assert os.path.isdir(seq_tr_dir), f"[ERR] missing track seq dir: {seq_tr_dir}"
        tr_seq_dir_by_sid[sid] = seq_tr_dir
        seq_gt_dir = os.path.join(args.gt_root, f"seq_{sid}")
        assert os.path.isdir(seq_gt_dir), f"[ERR] missing GT seq dir: {seq_gt_dir}"

        # Build metrics aggregator
        SM = SeqMetrics(classes=classes, iou_thr=args.iou)

        # Iterate frames present in tracker
        frame_files = list_frame_jsons(seq_tr_dir)
        for fpath in frame_files:
            fobj = load_json(fpath)
            frame_idx = int(fobj["frame_index"])
            tr_objs = fobj.get("tracks", [])

            gpath = os.path.join(seq_gt_dir, f"frame_{frame_idx:06d}.json")
            if os.path.isfile(gpath):
                gt_objs = load_json(gpath).get("objects", [])
            else:
                gt_objs = []

            # Per class matching
            for cls in classes:
                gt_list = [g for g in gt_objs if g.get("class") == cls]
                tr_list = [t for t in tr_objs if t.get("class") == cls]

                gt_boxes = [g["bbox"] for g in gt_list]
                tr_boxes = [t["bbox"] for t in tr_list]

                matches, um_g, um_t, iou_mat = match_by_iou(gt_boxes, tr_boxes, thr=args.iou)
                SM.add_frame(cls, gt_list, tr_list, matches, iou_mat)

        # Summarize seq
        per_cls, overall = SM.summarize()
        row = {
            "sequence": sid,
            "MOTA": overall["MOTA"],
            "MOTP": overall["MOTP"],
            "IDF1": overall["IDF1"],
            "GT": overall["GT"],
            "TR": overall["TR"],
            "TP": overall["TP"],
            "FP": overall["FP"],
            "FN": overall["FN"],
            "IDSW": overall["IDSW"]
        }
        # add few class metrics (Car/Ped/Cyc if present)
        for k in ["Car","Pedestrian","Cyclist"]:
            if k in per_cls:
                row[f"{k}_MOTA"] = per_cls[k]["MOTA"]
                row[f"{k}_IDF1"] = per_cls[k]["IDF1"]
        per_seq_rows.append(row)

    # Write per-seq CSV
    csv_path = os.path.join(out_dir, "metrics_per_seq.csv")
    # Collect all keys
    header = sorted({k for r in per_seq_rows for k in r.keys()}, key=lambda x: (x!="sequence", x))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in per_seq_rows:
            w.writerow(r)
    print(f"[OK] saved per-seq CSV: {csv_path}")

    # Aggregate overall across sequences
    agg = SeqMetrics(classes=classes, iou_thr=args.iou)
    for sid in seq_ids:
        # re-run light pass: we could also re-read and accumulate, but to be light we just sum CSV rows is insufficient for IDF1.
        # So re-read frames quickly:
        seq_tr_dir = os.path.join(tr_root_tag, f"seq_{sid}")
        seq_gt_dir = os.path.join(args.gt_root, f"seq_{sid}")
        frame_files = list_frame_jsons(seq_tr_dir)
        for fpath in frame_files:
            fobj = load_json(fpath)
            frame_idx = int(fobj["frame_index"])
            tr_objs = fobj.get("tracks", [])
            gpath = os.path.join(seq_gt_dir, f"frame_{frame_idx:06d}.json")
            gt_objs = load_json(gpath).get("objects", []) if os.path.isfile(gpath) else []

            for cls in classes:
                gt_list = [g for g in gt_objs if g.get("class") == cls]
                tr_list = [t for t in tr_objs if t.get("class") == cls]
                gt_boxes = [g["bbox"] for g in gt_list]
                tr_boxes = [t["bbox"] for t in tr_list]
                matches, um_g, um_t, iou_mat = match_by_iou(gt_boxes, tr_boxes, thr=args.iou)
                agg.add_frame(cls, gt_list, tr_list, matches, iou_mat)

    per_cls_all, overall_all = agg.summarize()

    # (optional) HOTA via trackeval
    hota_res = None
    if args.use_trackeval:
        hota_res = maybe_run_trackeval_hota(args.gt_root, tr_seq_dir_by_sid, out_dir, iou_thr=args.iou)

    # Write summary.json
    summary = {
        "params": {
            "tracker": args.tracker,
            "model_tag": args.model_tag,
            "iou": args.iou,
            "classes": classes,
            "seqs": seq_ids
        },
        "overall": overall_all,
        "per_class": per_cls_all,
    }
    if hota_res is not None:
        summary["HOTA"] = hota_res
    sum_path = os.path.join(out_dir, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] saved summary: {sum_path}")

if __name__ == "__main__":
    main()
