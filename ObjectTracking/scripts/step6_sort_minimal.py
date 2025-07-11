# scripts/step6_sort_minimal.py
import os, csv, glob, cv2, numpy as np
from collections import defaultdict

# -------------------------
# 0) 공용 유틸
# -------------------------
def load_dets(det_txt):
    frames = defaultdict(list)
    with open(det_txt) as f:
        for row in csv.reader(f):
            if not row: continue
            fr = int(row[0]); cls_id = int(row[1]); score = float(row[2])
            l,t,r,b = map(float, row[3:7])
            frames[fr].append([cls_id, score, [l,t,r,b]])
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

# -------------------------
# 1) 칼만 필터 (상태: [cx, cy, vx, vy, w, h])
#    - 중심은 등속(속도 포함), w,h는 느리게 변한다고 가정(상태에 그대로 유지)
# -------------------------
class KalmanBox:
    def __init__(self, dt=0.1, q_pos=1.0, q_vel=10.0, q_size=0.1, r_pos=10.0, r_size=5.0):
        # 상태 6x1
        self.x = np.zeros((6,1), dtype=float)
        # 공분산 6x6
        self.P = np.eye(6)*1000.0

        # 상태전이 F
        self.F = np.array([
            [1,0,dt,0, 0,0],
            [0,1,0, dt,0,0],
            [0,0,1, 0, 0,0],
            [0,0,0, 1, 0,0],
            [0,0,0, 0, 1,0],
            [0,0,0, 0, 0,1],
        ], dtype=float)

        # 프로세스 잡음 Q
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_size, q_size]).astype(float)

        # 관측 z = [cx, cy, w, h]
        self.H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ], dtype=float)

        # 관측 잡음 R
        self.R = np.diag([r_pos, r_pos, r_size, r_size]).astype(float)

        self.I = np.eye(6, dtype=float)

    def initiate(self, cx, cy, w, h):
        self.x[:] = np.array([[cx],[cy],[0.0],[0.0],[w],[h]], dtype=float)
        self.P = np.eye(6)*10.0  # 초기 불확실성 낮춤

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas):
        # meas: [cx, cy, w, h]
        z = meas.reshape(4,1)
        y = z - (self.H @ self.x)                     # 혁신
        S = self.H @ self.P @ self.H.T + self.R       # 혁신 공분산
        K = self.P @ self.H.T @ np.linalg.inv(S)      # 칼만 이득
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def get_bbox(self):
        cx, cy, vx, vy, w, h = self.x.flatten()
        return to_ltrb(cx, cy, max(1.0, w), max(1.0, h))

# -------------------------
# 2) Minimal SORT 트래커
#    - IoU cost로 헝가리안 할당(없으면 greedy)
#    - max_age: 미업데이트 허용 프레임 수
#    - min_hits: 확정 전 최소 히트(초기 흔들림 억제)
# -------------------------
class SORT:
    def __init__(self, iou_thres=0.3, max_age=15, min_hits=3, dt=0.1):
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.min_hits = min_hits
        self.dt = dt
        self.next_id = 0
        # 트랙: dict("id","cls","kf","hits","age","time_since_update","score")
        self.tracks = []

    def _hungarian(self, cost):
        try:
            from scipy.optimize import linear_sum_assignment
            r, c = linear_sum_assignment(cost)
            return np.array(list(zip(r,c)))
        except Exception:
            # 폴백: greedy (큰 IoU 우선)
            pairs = []
            for i in range(cost.shape[0]):
                for j in range(cost.shape[1]):
                    pairs.append(( -cost[i,j], i, j ))  # cost↑ 나쁨 → -cost로 정렬
            pairs.sort()
            used_r, used_c, match = set(), set(), []
            for _, i, j in pairs:
                if i in used_r or j in used_c: continue
                used_r.add(i); used_c.add(j); match.append((i,j))
            return np.array(match)

    def update(self, dets):
        """
        dets: list of [cls_id, score, [l,t,r,b]]
        return current frame outputs: list of [tid, cls, score, [l,t,r,b]] for confirmed tracks
        """
        # 1) 모든 트랙 예측
        for tr in self.tracks:
            tr["kf"].predict()
            tr["age"] += 1
            tr["time_since_update"] += 1

        # 2) 클래스별로 매칭 (클래스 섞으면 ID가 엉킬 수 있음)
        outputs = []
        if len(self.tracks)==0 and len(dets)==0:
            return outputs

        # 트랙/디텍션 인덱스 수집
        det_indices_by_cls = defaultdict(list)
        for di, (cls, score, box) in enumerate(dets):
            det_indices_by_cls[cls].append(di)

        track_indices_by_cls = defaultdict(list)
        for ti, tr in enumerate(self.tracks):
            track_indices_by_cls[tr["cls"]].append(ti)

        matched_t = set(); matched_d = set()

        for cls, det_idx_list in det_indices_by_cls.items():
            tr_idx_list = track_indices_by_cls.get(cls, [])
            if not tr_idx_list:
                continue
            # cost = 1 - IoU(예측 bbox vs det bbox)
            cost = np.ones((len(tr_idx_list), len(det_idx_list)), dtype=float)
            for i, ti in enumerate(tr_idx_list):
                pred_box = self.tracks[ti]["kf"].get_bbox()
                for j, di in enumerate(det_idx_list):
                    _, _, b = dets[di]
                    I = iou(pred_box, b)
                    cost[i, j] = 1.0 - I

            # 헝가리안
            matches = self._hungarian(cost)
            for i, j in matches:
                ti = tr_idx_list[i]; di = det_idx_list[j]
                I = 1.0 - cost[i, j]
                if I < self.iou_thres:
                    continue  # 임계 미만은 매칭 취소
                # 업데이트
                cls_id, score, box = dets[di]
                cx, cy, w, h = to_cxcywh(*box)
                self.tracks[ti]["kf"].update(np.array([cx,cy,w,h], dtype=float))
                self.tracks[ti]["time_since_update"] = 0
                self.tracks[ti]["hits"] += 1
                self.tracks[ti]["score"] = 0.7*self.tracks[ti]["score"] + 0.3*score
                matched_t.add(ti); matched_d.add(di)

        # 3) unmatched detections → 신규 트랙
        for di, (cls, score, box) in enumerate(dets):
            if di in matched_d: continue
            kf = KalmanBox(dt=self.dt)
            cx, cy, w, h = to_cxcywh(*box)
            kf.initiate(cx, cy, w, h)
            tr = {
                "id": self.next_id, "cls": cls, "kf": kf,
                "hits": 1, "age": 0, "time_since_update": 0,
                "score": float(score),
            }
            self.next_id += 1
            self.tracks.append(tr)

        # 4) 오래된 트랙 제거
        self.tracks = [t for t in self.tracks if t["time_since_update"] <= self.max_age]

        # 5) 출력(확정 트랙만)
        for tr in self.tracks:
            if tr["hits"] >= self.min_hits or tr["time_since_update"] == 0:
                box = tr["kf"].get_bbox()
                outputs.append([tr["id"], tr["cls"], tr["score"], box])
        return outputs

# -------------------------
# 3) 시각화(확인용 1프레임)
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
# 4) 메인: 캐시 → SORT → 결과txt + 1프레임 이미지
# -------------------------
def main(
    kitti_root="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training",
    seq="0000",
    det_txt="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/dets_dummy/0000.txt",
    out_track_txt="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/tracks_sort_min/0000.txt",
    out_vis="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/vis/step6_sort_seq0000_f0010.jpg",
    frame_vis_idx=10,
    iou_thres=0.3, max_age=15, min_hits=3, dt=0.1
):
    os.makedirs(os.path.dirname(out_track_txt) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_vis) or ".", exist_ok=True)

    # 이미지 목록
    img_dir = os.path.join(kitti_root, "image_02", seq)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert img_files, f"No images in {img_dir}"

    # detection 로드 & 트래커 준비
    dets = load_dets(det_txt)
    max_frame = max(dets.keys()) if dets else -1
    tracker = SORT(iou_thres=iou_thres, max_age=max_age, min_hits=min_hits, dt=dt)

    with open(out_track_txt, "w", newline="") as f:
        writer = csv.writer(f)
        for fr in range(max_frame+1):
            outs = tracker.update(dets.get(fr, []))
            for tid, cls, score, (l,t,r,b) in outs:
                writer.writerow([fr, tid, cls, f"{score:.3f}", f"{l:.1f}", f"{t:.1f}", f"{r:.1f}", f"{b:.1f}"])
            if fr == frame_vis_idx:
                img = cv2.imread(img_files[fr])
                vis = draw_once(img, outs)
                cv2.imwrite(out_vis, vis)

    print("wrote tracks:", out_track_txt)
    print("saved vis:", out_vis)

if __name__ == "__main__":
    main()
