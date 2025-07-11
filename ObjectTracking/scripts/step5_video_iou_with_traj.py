# scripts/step5_gif_iou_with_traj.py
import os, csv, glob, cv2
from collections import defaultdict, deque

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

class IOUTracker:
    def __init__(self, iou_thres=0.3, max_age=15):
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.next_id = 0
        self.tracks = []  # {"id","cls","bbox","age","score"}

    def _match(self, dets):
        pairs = []
        for ti, trk in enumerate(self.tracks):
            for di, (cls, score, box) in enumerate(dets):
                if cls != trk["cls"]: continue
                I = iou(trk["bbox"], box)
                if I >= self.iou_thres: pairs.append((I, ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])
        used_t, used_d = set(), set(); matches = []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d: continue
            used_t.add(ti); used_d.add(di); matches.append((ti, di))
        unmatched_t = [i for i in range(len(self.tracks)) if i not in used_t]
        unmatched_d = [i for i in range(len(dets)) if i not in used_d]
        return matches, unmatched_t, unmatched_d

    def update(self, dets):
        outs = []
        if not self.tracks:
            for cls, score, box in dets:
                tid = self.next_id; self.next_id += 1
                self.tracks.append({"id":tid,"cls":cls,"bbox":box,"age":0,"score":score})
                outs.append([tid, cls, score, box])
        else:
            matches, ut, ud = self._match(dets)
            for ti, di in matches:
                cls, score, box = dets[di]
                tr = self.tracks[ti]
                tr["bbox"] = box
                tr["score"] = 0.7*tr["score"] + 0.3*score
                tr["age"] = 0
                outs.append([tr["id"], cls, tr["score"], box])
            for ti in ut: self.tracks[ti]["age"] += 1
            for di in ud:
                cls, score, box = dets[di]
                tid = self.next_id; self.next_id += 1
                self.tracks.append({"id":tid,"cls":cls,"bbox":box,"age":0,"score":score})
                outs.append([tid, cls, score, box])
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        return outs

def color_from_id(tid):
    palette = [
        (255, 99, 71), (30, 144, 255), (60, 179, 113), (238, 130, 238),
        (255, 215, 0), (255, 140, 0), (0, 206, 209), (199, 21, 133),
        (123, 104, 238), (72, 209, 204), (244, 164, 96), (154, 205, 50),
    ]
    return palette[tid % len(palette)]

def draw_frame(img, outs, trails, trail_len=20):
    for tid, cls, score, (l,t,r,b) in outs:
        l,t,r,b = map(int, [l,t,r,b])
        c = color_from_id(tid)
        cv2.rectangle(img, (l,t), (r,b), c, 2)
        cv2.putText(img, f"{tid}", (l, max(12, t-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        cx, cy = int((l+r)/2), int((t+b)/2)
        trails.setdefault(tid, deque(maxlen=trail_len)).append((cx, cy))
    for tid, q in list(trails.items()):
        if len(q) >= 2:
            c = color_from_id(tid)
            for i in range(1, len(q)):
                cv2.line(img, q[i-1], q[i], c, 2)
    return img

def save_gif(frames_rgb, path, fps=10, loop=0):
    """frames_rgb: list of RGB numpy arrays"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio
        # duration: 1/fps seconds per frame
        imageio.mimsave(path, frames_rgb, duration=1.0/fps, loop=loop)
    except Exception:
        # Pillow fallback
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames_rgb]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000/fps), loop=loop)

def main(
    kitti_root="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training",
    seq="0000",
    det_txt="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/dets_dummy/0000.txt",
    out_gif="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/vis/seq0000_iou_traj.gif",
    iou_thres=0.3, max_age=15, fps=10, trail_len=20
):
    img_dir = os.path.join(kitti_root, "image_02", seq)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert img_files, f"No images in {img_dir}"

    dets = load_dets(det_txt)
    tracker = IOUTracker(iou_thres=iou_thres, max_age=max_age)
    trails = {}

    frames_rgb = []  # GIF로 내보낼 RGB 프레임 목록
    for fr, img_path in enumerate(img_files):
        img = cv2.imread(img_path)                # BGR
        outs = tracker.update(dets.get(fr, []))
        vis = draw_frame(img, outs, trails, trail_len=trail_len)
        frames_rgb.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))  # BGR→RGB

    save_gif(frames_rgb, out_gif, fps=fps, loop=0)
    print("saved gif:", out_gif)

if __name__ == "__main__":
    main()
