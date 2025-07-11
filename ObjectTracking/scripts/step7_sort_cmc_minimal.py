# scripts/step7_sort_cmc_minimal.py
# SORT(칼만+헝가리안)에 CMC(호모그래피) 보정 추가
# - 매 프레임 ORB 특징점으로 H(prev->curr) 추정 (RANSAC)
# - track의 마지막 관측 박스를 H로 워핑
# - KF 예측 박스와 CMC 워핑 박스를 alpha로 블렌딩해 매칭 IoU 계산

import os, csv, glob, cv2, numpy as np
from collections import defaultdict

# -------------------------
# 공용 로더/변환/IoU
# -------------------------
def load_dets(det_txt):
    frames = defaultdict(list)
    with open(det_txt) as f:
        for row in csv.reader(f):
            if not row: continue
            fr = int(row[0]); cls_id = int(row[1]); score = float(row[2])
            l, t, r, b = map(float, row[3:7])
            frames[fr].append([cls_id, score, [l, t, r, b]])
    return frames

def iou(a, b):
    l1,t1,r1,b1 = a; l2,t2,r2,b2 = b
    inter_l, inter_t = max(l1,l2), max(t1,t2)
    inter_r, inter_b = min(r1,r2), min(b1,b2)
    iw, ih = max(0.0, inter_r-inter_l), max(0.0, inter_b-inter_t)
    inter = iw*ih
    if inter <= 0: return 0.0
    area1 = (r1-l1)*(b1-t1); area2 = (r2-l2)*(b2-t2)
    return inter / (area1 + area2 - inter + 1e-6)

def to_cxcywh(l,t,r,b):
    w = max(0.0, r-l); h = max(0.0, b-t)
    return np.array([l + w/2.0, t + h/2.0, w, h], dtype=float)

def to_ltrb(cx,cy,w,h):
    l = cx - w/2.0; t = cy - h/2.0
    r = cx + w/2.0; b = cy + h/2.0
    return [l,t,r,b]

def clamp_box(l,t,r,b,W,H):
    l = max(0, min(l, W-1)); r = max(0, min(r, W-1))
    t = max(0, min(t, H-1)); b = max(0, min(b, H-1))
    if r < l: l, r = r, l
    if b < t: t, b = b, t
    return [l,t,r,b]

# -------------------------
# CMC: ORB + RANSAC Homography
# -------------------------
def estimate_h(prev_bgr, curr_bgr, max_feats=1500, ratio=0.75, ransac_thr=5.0):
    """prev→curr 호모그래피 추정. 실패하면 단위행렬 반환."""
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_feats)
    k1, d1 = orb.detectAndCompute(prev, None)
    k2, d2 = orb.detectAndCompute(curr, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.eye(3, dtype=float)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 8:
        return np.eye(3, dtype=float)

    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thr)
    if H is None:
        H = np.eye(3, dtype=float)
    return H

def warp_box_with_H(box_ltrb, Hmat, W, Himg):
    """박스 4꼭짓점을 H로 워핑한 뒤 바운딩 박스로 환산."""
    if box_ltrb is None or Hmat is None:
        return box_ltrb
    l,t,r,b = box_ltrb
    pts = np.array([[l,t],[r,t],[r,b],[l,b]], dtype=np.float32).reshape(-1,1,2)
    pts_h = cv2.perspectiveTransform(pts, Hmat).reshape(-1,2)
    xs, ys = pts_h[:,0], pts_h[:,1]
    l2, t2, r2, b2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return clamp_box(l2, t2, r2, b2, W, Himg)

def blend_boxes(box_a, box_b, alpha=0.6):
    """cxcywh로 변환 후 보간(박스 평균). alpha=1이면 box_a, 0이면 box_b."""
    ca = to_cxcywh(*box_a); cb = to_cxcywh(*box_b)
    c = alpha*ca + (1.0-alpha)*cb
    return to_ltrb(*c)

# -------------------------
# 칼만 박스 (cx,cy,vx,vy,w,h)
# -------------------------
class KalmanBox:
    def __init__(self, dt=0.1, q_pos=1.0, q_vel=10.0, q_size=0.1, r_pos=10.0, r_size=5.0):
        self.x = np.zeros((6,1), dtype=float)
        self.P = np.eye(6)*1000.0
        self.F = np.array([
            [1,0,dt,0, 0,0],
            [0,1,0, dt,0,0],
            [0,0,1, 0, 0,0],
            [0,0,0, 1, 0,0],
            [0,0,0, 0, 1,0],
            [0,0,0, 0, 0,1],
        ], dtype=float)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_size, q_size]).astype(float)
        self.H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ], dtype=float)
        self.R = np.diag([r_pos, r_pos, r_size, r_size]).astype(float)
        self.I = np.eye(6, dtype=float)

    def initiate(self, cx, cy, w, h):
        self.x[:] = np.array([[cx],[cy],[0.0],[0.0],[w],[h]])
        self.P = np.eye(6)*10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas):
        z = meas.reshape(4,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def get_bbox(self):
        cx, cy, vx, vy, w, h = self.x.flatten()
        return to_ltrb(cx, cy, max(1.0,w), max(1.0,h))

# -------------------------
# SORT + CMC
# -------------------------
class SORT_CMC:
    def __init__(self, iou_thres=0.3, max_age=15, min_hits=3, dt=0.1, alpha=0.6, img_size=None):
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.min_hits = min_hits
        self.dt = dt
        self.alpha = alpha  # CMC vs KF blending
        self.next_id = 0
        self.tracks = []    # {"id","cls","kf","hits","age","time_since_update","score","last_obs"}
        self.img_size = img_size  # (H, W)

    def _hungarian(self, cost):
        try:
            from scipy.optimize import linear_sum_assignment
            r, c = linear_sum_assignment(cost)
            return np.array(list(zip(r,c)))
        except Exception:
            # fallback: greedy
            pairs = []
            for i in range(cost.shape[0]):
                for j in range(cost.shape[1]):
                    pairs.append((-cost[i,j], i, j))
            pairs.sort()
            used_r, used_c, match = set(), set(), []
            for _, i, j in pairs:
                if i in used_r or j in used_c: continue
                used_r.add(i); used_c.add(j); match.append((i,j))
            return np.array(match)

    def update(self, dets, Hmat):
        """
        dets: list of [cls, score, [l,t,r,b]]
        Hmat: prev->curr homography (np.ndarray 3x3)
        return: list of [tid, cls, score, [l,t,r,b]] (confirmed)
        """
        Himg, Wimg = self.img_size

        # 1) 예측
        for tr in self.tracks:
            tr["kf"].predict()
            tr["age"] += 1
            tr["time_since_update"] += 1

        outputs = []
        if len(self.tracks)==0 and len(dets)==0:
            return outputs

        # 2) 클래스별 매칭: cost = 1 - IoU( blended(KF_pred, CMC_warp) , det )
        det_idx_by_cls = defaultdict(list)
        for di, (cls, score, box) in enumerate(dets):
            det_idx_by_cls[cls].append(di)

        tr_idx_by_cls = defaultdict(list)
        for ti, tr in enumerate(self.tracks):
            tr_idx_by_cls[tr["cls"]].append(ti)

        matched_t, matched_d = set(), set()
        for cls, det_idx_list in det_idx_by_cls.items():
            tr_idx_list = tr_idx_by_cls.get(cls, [])
            if not tr_idx_list: 
                continue
            cost = np.ones((len(tr_idx_list), len(det_idx_list)), dtype=float)
            for i, ti in enumerate(tr_idx_list):
                tr = self.tracks[ti]
                pred_box = tr["kf"].get_bbox()
                cmc_box  = warp_box_with_H(tr["last_obs"], Hmat, Wimg, Himg) if tr["last_obs"] is not None else pred_box
                blended  = blend_boxes(cmc_box, pred_box, alpha=self.alpha)
                for j, di in enumerate(det_idx_list):
                    _, _, b = dets[di]
                    I = iou(blended, b)
                    cost[i,j] = 1.0 - I
            matches = self._hungarian(cost)
            for i, j in matches:
                ti = tr_idx_list[i]; di = det_idx_list[j]
                I = 1.0 - cost[i, j]
                if I < self.iou_thres:
                    continue
                cls_id, score, box = dets[di]
                cx, cy, w, h = to_cxcywh(*box)
                self.tracks[ti]["kf"].update(np.array([cx,cy,w,h], dtype=float))
                self.tracks[ti]["time_since_update"] = 0
                self.tracks[ti]["hits"] += 1
                self.tracks[ti]["score"] = 0.7*self.tracks[ti]["score"] + 0.3*score
                self.tracks[ti]["last_obs"] = box
                matched_t.add(ti); matched_d.add(di)

        # 3) 미매칭 detection → 신규 트랙
        for di, (cls, score, box) in enumerate(dets):
            if di in matched_d: continue
            kf = KalmanBox(dt=self.dt)
            cx, cy, w, h = to_cxcywh(*box)
            kf.initiate(cx, cy, w, h)
            tr = {"id": self.next_id, "cls": cls, "kf": kf,
                  "hits": 1, "age": 0, "time_since_update": 0,
                  "score": float(score), "last_obs": box}
            self.next_id += 1
            self.tracks.append(tr)

        # 4) 오래된 트랙 제거
        self.tracks = [t for t in self.tracks if t["time_since_update"] <= self.max_age]

        # 5) 출력(확정 트랙만)
        for tr in self.tracks:
            if tr["hits"] >= self.min_hits or tr["time_since_update"] == 0:
                box = tr["kf"].get_bbox()
                # 화면 밖으로 튄 값 보정
                box = clamp_box(*box, W=Wimg, H=Himg)
                outputs.append([tr["id"], tr["cls"], tr["score"], box])
        return outputs

# -------------------------
# 시각화(한 프레임 확인)
# -------------------------
def color_from_id(tid):
    palette = [
        (255, 99, 71), (30, 144, 255), (60, 179, 113), (238, 130, 238),
        (255, 215, 0), (255, 140, 0), (0, 206, 209), (199, 21, 133),
        (123, 104, 238), (72, 209, 204), (244, 164, 96), (154, 205, 50),
    ]
    return palette[tid % len(palette)]

def draw_once(img, outs):
    for tid, cls, score, (l,t,r,b) in outs:
        l,t,r,b = map(int, [l,t,r,b])
        c = color_from_id(tid)
        cv2.rectangle(img, (l,t), (r,b), c, 2)
        cv2.putText(img, f"{tid}", (l, max(12, t-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    return img

# -------------------------
# 메인: H 추정 + SORT_CMC 업데이트
# -------------------------
def main(
    kitti_root="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training",
    seq="0000",
    det_txt="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/dets_dummy/0000.txt",
    out_track_txt="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/tracks_sort_cmc/0000.txt",
    out_vis="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/vis/step7_sortcmc_seq0000_f0010.jpg",
    frame_vis_idx=10,
    # 하이퍼파라미터
    iou_thres=0.3, max_age=15, min_hits=3, dt=0.1, alpha=0.6
):
    os.makedirs(os.path.dirname(out_track_txt) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_vis) or ".", exist_ok=True)

    # 이미지 시퀀스
    img_dir = os.path.join(kitti_root, "image_02", seq)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert img_files, f"No images in {img_dir}"
    Himg, Wimg = cv2.imread(img_files[0]).shape[:2]

    # 로드
    dets = load_dets(det_txt)
    max_frame = max(dets.keys()) if dets else -1

    tracker = SORT_CMC(iou_thres=iou_thres, max_age=max_age, min_hits=min_hits,
                       dt=dt, alpha=alpha, img_size=(Himg, Wimg))

    with open(out_track_txt, "w", newline="") as f:
        writer = csv.writer(f)

        prev_img = None
        for fr in range(max_frame+1):
            curr_img = cv2.imread(img_files[fr])
            if prev_img is None:
                Hmat = np.eye(3, dtype=float)  # 첫 프레임은 보정 없음
            else:
                Hmat = estimate_h(prev_img, curr_img)

            outs = tracker.update(dets.get(fr, []), Hmat)

            for tid, cls, score, (l,t,r,b) in outs:
                writer.writerow([fr, tid, cls, f"{score:.3f}",
                                 f"{l:.1f}", f"{t:.1f}", f"{r:.1f}", f"{b:.1f}"])

            if fr == frame_vis_idx:
                vis = draw_once(curr_img.copy(), outs)
                cv2.imwrite(out_vis, vis)

            prev_img = curr_img

    print("wrote tracks:", out_track_txt)
    print("saved vis:", out_vis)

if __name__ == "__main__":
    main()
