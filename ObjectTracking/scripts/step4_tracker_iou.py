# scripts/step4_tracker_iou.py
import os, csv, glob, cv2
from collections import defaultdict

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
    # a,b: [l,t,r,b]
    l1,t1,r1,b1 = a; l2,t2,r2,b2 = b
    inter_l, inter_t = max(l1,l2), max(t1,t2)
    inter_r, inter_b = min(r1,r2), min(b1,b2)
    iw, ih = max(0.0, inter_r-inter_l), max(0.0, inter_b-inter_t)
    inter = iw*ih
    if inter <= 0: return 0.0
    area1 = (r1-l1)*(b1-t1); area2 = (r2-l2)*(b2-t2)
    return inter / (area1 + area2 - inter + 1e-6)

class IOUTracker:
    def __init__(self, iou_thres=0.3, max_age=15):
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.next_id = 0
        # track = {"id":int, "cls":int, "bbox":[l,t,r,b], "age":int, "score":float}
        self.tracks = []

    def _match(self, dets):
        # greedy: 모든 (track, det) iou 계산 후 큰 순서로 매칭
        pairs = []
        for ti, trk in enumerate(self.tracks):
            for di, (cls, score, box) in enumerate(dets):
                if cls != trk["cls"]: 
                    continue
                i = iou(trk["bbox"], box)
                if i >= self.iou_thres:
                    pairs.append((i, ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        used_t, used_d = set(), set()
        matches = []
        for _iou, ti, di in pairs:
            if ti in used_t or di in used_d: 
                continue
            used_t.add(ti); used_d.add(di)
            matches.append((ti, di))
        unmatched_t = [i for i in range(len(self.tracks)) if i not in used_t]
        unmatched_d = [i for i in range(len(dets)) if i not in used_d]
        return matches, unmatched_t, unmatched_d

    def update(self, dets):
        # dets: list of [cls_id, score, [l,t,r,b]]
        outputs = []  # current frame outputs
        if not self.tracks:
            # 첫 프레임: 전부 신규 트랙
            for cls, score, box in dets:
                tid = self.next_id; self.next_id += 1
                self.tracks.append({"id":tid, "cls":cls, "bbox":box, "age":0, "score":score})
                outputs.append([tid, cls, score, box])
        else:
            matches, unmatched_t, unmatched_d = self._match(dets)
            # 매칭된 트랙 업데이트
            for ti, di in matches:
                cls, score, box = dets[di]
                self.tracks[ti]["bbox"] = box
                self.tracks[ti]["score"] = 0.7*self.tracks[ti]["score"] + 0.3*score
                self.tracks[ti]["age"] = 0
                outputs.append([self.tracks[ti]["id"], cls, self.tracks[ti]["score"], box])
            # 매칭 실패 트랙은 age 증가
            for ti in unmatched_t:
                self.tracks[ti]["age"] += 1
            # 새로운 트랙 생성
            for di in unmatched_d:
                cls, score, box = dets[di]
                tid = self.next_id; self.next_id += 1
                self.tracks.append({"id":tid, "cls":cls, "bbox":box, "age":0, "score":score})
                outputs.append([tid, cls, score, box])
            # 오래된 트랙 제거
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        return outputs  # list of [track_id, cls_id, score, [l,t,r,b]]

def draw_tracks(img, outs):
    COLORS = {0:(255,0,0), 1:(0,0,255), 2:(0,255,0)}  # car/ped/cyc
    for tid, cls, score, (l,t,r,b) in outs:
        l,t,r,b = map(int, [l,t,r,b])
        cv2.rectangle(img, (l,t), (r,b), COLORS[cls], 2)
        cv2.putText(img, f"{tid}", (l, max(10, t-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2, cv2.LINE_AA)
    return img

def main(
    kitti_root="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training",
    seq="0000",
    det_txt="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/dets_dummy/0000.txt",
    out_track_txt="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/tracks_iou/0000.txt",
    out_vis="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/vis/step4_track_seq0000_f0010.jpg",
    frame_vis_idx=10,
    iou_thres=0.3, max_age=15
):
    os.makedirs(os.path.dirname(out_track_txt) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_vis) or ".", exist_ok=True)

    # 이미지 목록
    img_dir = os.path.join(kitti_root, "image_02", seq)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert img_files, f"No images in {img_dir}"

    # detection 로드
    dets = load_dets(det_txt)
    max_frame = max(dets.keys()) if dets else -1

    # 트래커
    tracker = IOUTracker(iou_thres=iou_thres, max_age=max_age)

    # 결과 저장
    with open(out_track_txt, "w", newline="") as f:
        writer = csv.writer(f)
        for fr in range(max_frame+1):
            outs = tracker.update(dets.get(fr, []))  # 현재 프레임 업데이트
            for tid, cls, score, (l,t,r,b) in outs:
                writer.writerow([fr, tid, cls, f"{score:.3f}", f"{l:.1f}", f"{t:.1f}", f"{r:.1f}", f"{b:.1f}"])
            # 시각화 프레임이면 저장
            if fr == frame_vis_idx:
                img = cv2.imread(img_files[fr])
                vis = draw_tracks(img.copy(), outs)
                cv2.imwrite(out_vis, vis)

    print("wrote tracks:", out_track_txt)
    print("saved vis:", out_vis)

if __name__ == "__main__":
    main()
