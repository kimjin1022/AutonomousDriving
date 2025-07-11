# scripts/step12_eval_motmetrics.py
import os, csv, glob, math
from collections import defaultdict, Counter
import numpy as np
import motmetrics as mm
import cv2

# ---------- 경로 기본값 (네 환경) ----------
KITTI_ROOT = "/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training"
PRED_DIR   = "/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/tracks_sort_cmc"  # 예: step7 결과 디렉토리
OUT_DIR    = "/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/eval_mot"

# ---------- 클래스 매핑 ----------
MAP = {
    "Car":"car", "Van":"car", "Truck":"car",
    "Pedestrian":"pedestrian", "Person_sitting":"pedestrian", "Person":"pedestrian",
    "Cyclist":"cyclist",
}
IGNORE = {"Tram", "Misc", "DontCare"}
CLS_TO_ID = {"car":0, "pedestrian":1, "cyclist":2}
ID_TO_CLS = {v:k for k,v in CLS_TO_ID.items()}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ---------- IoU ----------
def iou_ltrb(a, b):
    l1,t1,r1,b1 = a; l2,t2,r2,b2 = b
    inter_l, inter_t = max(l1,l2), max(t1,t2)
    inter_r, inter_b = min(r1,r2), min(b1,b2)
    iw, ih = max(0.0, inter_r-inter_l), max(0.0, inter_b-inter_t)
    inter = iw*ih
    if inter <= 0: return 0.0
    area1 = (r1-l1)*(b1-t1); area2 = (r2-l2)*(b2-t2)
    return inter / max(1e-6, (area1 + area2 - inter))

def to_xywh(l,t,r,b):
    w = max(0.0, r-l); h = max(0.0, b-t)
    return [l, t, w, h]

# ---------- KITTI GT 로드: frame -> [ {id, cls_id, box(l,t,r,b)} ] ----------
def load_kitti_gt_seq(seq):
    label_path = os.path.join(KITTI_ROOT, "label_02", f"{seq}.txt")
    frames = defaultdict(list)
    with open(label_path) as f:
        for ln in f:
            if not ln.strip(): continue
            p = ln.split()
            frame = int(p[0]); tid = int(p[1]); typ = p[2]
            if typ in IGNORE: 
                continue
            cls = MAP.get(typ)
            if cls is None: 
                continue
            l,t,r,b = map(float, p[6:10])
            frames[frame].append({"id":tid, "cls_id": CLS_TO_ID[cls], "box":[l,t,r,b]})
    return frames

# ---------- 예측(트래킹 결과) 로드: frame -> [ {id, cls_id, box} ] ----------
def load_pred_seq(pred_txt):
    frames = defaultdict(list)
    with open(pred_txt) as f:
        for row in csv.reader(f):
            if not row: continue
            fr  = int(row[0]); tid = int(row[1])
            cls = int(row[2]); # score = float(row[3])  # 점수는 매칭엔 직접 사용 안 함
            l,t,r,b = map(float, row[4:8])
            frames[fr].append({"id":tid, "cls_id":cls, "box":[l,t,r,b]})
    return frames

# ---------- 한 시퀀스 평가(클래스 필터 지원) ----------
def evaluate_seq(gt_frames, pr_frames, img_size, iou_thr=0.5, class_filter=None):
    """
    class_filter: None이면 전체(클래스 무시), 정수(0/1/2)이면 해당 클래스만 평가
    """
    acc = mm.MOTAccumulator(auto_id=True)

    H, W = img_size
    max_fr = max(
        [max(gt_frames.keys()) if gt_frames else -1,
         max(pr_frames.keys()) if pr_frames else -1]
    )

    for fr in range(max_fr + 1):
        g_objs = gt_frames.get(fr, [])
        p_objs = pr_frames.get(fr, [])

        if class_filter is not None:
            g_objs = [o for o in g_objs if o["cls_id"] == class_filter]
            p_objs = [o for o in p_objs if o["cls_id"] == class_filter]

        g_ids  = [o["id"] for o in g_objs]
        p_ids  = [o["id"] for o in p_objs]
        g_boxes = [o["box"] for o in g_objs]
        p_boxes = [o["box"] for o in p_objs]

        if len(g_boxes) and len(p_boxes):
            # distance = 1 - IoU; 임계 미만은 NaN(=매칭 불가)
            C = np.zeros((len(g_boxes), len(p_boxes)), dtype=float)
            for i, gb in enumerate(g_boxes):
                for j, pb in enumerate(p_boxes):
                    iou = iou_ltrb(gb, pb)
                    C[i, j] = 1.0 - iou if iou >= iou_thr else np.nan
        else:
            C = np.empty((len(g_boxes), len(p_boxes)))
            C[:] = np.nan

        acc.update(g_ids, p_ids, C)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            'num_frames','mota','motp','idf1','idp','idr',
            'num_switches','num_objects','num_predictions','mostly_tracked','mostly_lost'
        ],
        name='acc'
    )
    return summary  # pandas DataFrame 1행

def main(
    seq="0000",
    pred_dir=PRED_DIR,
    out_dir=OUT_DIR,
    iou_thr=0.5
):
    ensure_dir(out_dir)

    # 어느 시퀀스를 평가할지 결정
    if seq == "all":
        img_root = os.path.join(KITTI_ROOT, "image_02")
        seqs = sorted([d for d in os.listdir(img_root) if d.isdigit()])
    else:
        seqs = [seq]

    all_rows_overall = []
    all_rows_percls = {0:[], 1:[], 2:[]}

    for s in seqs:
        # 로드
        gt_frames = load_kitti_gt_seq(s)
        pred_txt  = os.path.join(pred_dir, f"{s}.txt")
        if not os.path.isfile(pred_txt):
            print(f"[WARN] pred missing: {pred_txt} — skip")
            continue
        pr_frames = load_pred_seq(pred_txt)

        # 이미지 크기(모든 프레임 동일)
        img0 = sorted(glob.glob(os.path.join(KITTI_ROOT, "image_02", s, "*.png")))[0]
        H, W = cv2.imread(img0).shape[:2]

        print(f"[EVAL] seq {s} | GT frames={len(gt_frames)} | PR frames={len(pr_frames)} | size={W}x{H}")

        # 전체(클래스 무시) 평가
        sum_overall = evaluate_seq(gt_frames, pr_frames, (H,W), iou_thr=iou_thr, class_filter=None)
        sum_overall.index = [f"{s}_all"]
        all_rows_overall.append(sum_overall)

        # 클래스별 평가
        for cid in [0,1,2]:
            sum_c = evaluate_seq(gt_frames, pr_frames, (H,W), iou_thr=iou_thr, class_filter=cid)
            sum_c.index = [f"{s}_{ID_TO_CLS[cid]}"]
            all_rows_percls[cid].append(sum_c)

    # ----- 결과 합치기 & 저장 -----
    import pandas as pd

    if all_rows_overall:
        tbl_overall = pd.concat(all_rows_overall, axis=0)
        csv_overall = os.path.join(out_dir, "summary_overall.csv")
        tbl_overall.to_csv(csv_overall)
        print("\n=== Overall (per-sequence, class-agnostic) ===")
        print(tbl_overall[['mota','idf1','motp','idp','idr','num_switches','num_objects','num_predictions']])
        print("saved:", csv_overall)

    for cid in [0,1,2]:
        rows = all_rows_percls[cid]
        if not rows: 
            continue
        tbl = pd.concat(rows, axis=0)
        csv_p = os.path.join(out_dir, f"summary_{ID_TO_CLS[cid]}.csv")
        tbl.to_csv(csv_p)
        print(f"\n=== Class: {ID_TO_CLS[cid]} (per-sequence) ===")
        print(tbl[['mota','idf1','motp','idp','idr','num_switches']])
        print("saved:", csv_p)

    # 전체 평균(시퀀스 평균)도 레포트
    def print_mean(tbl, title):
        if tbl is None: return
        mean_row = tbl.mean(numeric_only=True)
        print(f"\n>>> Mean ({title}) — MOTA={mean_row['mota']:.3f}, IDF1={mean_row['idf1']:.3f}, MOTP={mean_row['motp']:.3f}, IDSW={mean_row['num_switches']:.1f}")

    try:
        print_mean(tbl_overall, "overall")
        for cid in [0,1,2]:
            rows = all_rows_percls[cid]
            if rows:
                tbl = __import__('pandas').concat(rows, axis=0)
                print_mean(tbl, ID_TO_CLS[cid])
    except Exception:
        pass

if __name__ == "__main__":
    # 예) SORT+CMC 결과 디렉토리 평가
    # main(seq="0000", pred_dir=PRED_DIR, out_dir=OUT_DIR, iou_thr=0.5)

    # 바꾸고 싶으면 아래를 수정해서 실행
    main(
        seq="0000",
        pred_dir="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/tracks_sort_cmc",
        out_dir="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/eval_mot/sortcmc_iou50",
        iou_thr=0.5
    )
