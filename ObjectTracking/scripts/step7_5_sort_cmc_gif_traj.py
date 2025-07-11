# scripts/step7_5_sort_cmc_gif_traj.py
# 목적: Step 7의 SORT_CMC(칼만+헝가리안+카메라모션보정)으로 시퀀스를 돌려
#       ID 라벨 + 궤적을 그리며 GIF로 저장
# 의존: 같은 폴더에 step7_sort_cmc_minimal.py (SORT_CMC, load_dets, estimate_h)

import os, glob, cv2
from collections import deque
from step7_sort_cmc_minimal import SORT_CMC, load_dets, estimate_h  # 재사용

def color_from_id(tid):
    """ID 고정 색상 팔레트 (파일 간 독립 사용)"""
    palette = [
        (255, 99, 71), (30, 144, 255), (60, 179, 113), (238, 130, 238),
        (255, 215, 0), (255, 140, 0), (0, 206, 209), (199, 21, 133),
        (123, 104, 238), (72, 209, 204), (244, 164, 96), (154, 205, 50),
    ]
    return palette[tid % len(palette)]

def draw_frame(img, outs, trails, trail_len=20):
    """
    outs: [ [track_id, cls_id, score, [l,t,r,b]], ... ]
    trails: {track_id: deque([(cx,cy), ...])}
    """
    for tid, cls, score, (l, t, r, b) in outs:
        l, t, r, b = map(int, [l, t, r, b])
        c = color_from_id(tid)
        # 박스 + ID
        cv2.rectangle(img, (l, t), (r, b), c, 2)
        cv2.putText(img, f"{tid}", (l, max(12, t - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        # 궤적 포인트(박스 중심)
        cx, cy = (l + r) // 2, (t + b) // 2
        trails.setdefault(tid, deque(maxlen=trail_len)).append((cx, cy))

    # 궤적 선 연결
    for tid, q in list(trails.items()):
        if len(q) >= 2:
            c = color_from_id(tid)
            for i in range(1, len(q)):
                cv2.line(img, q[i - 1], q[i], c, 2)
    return img

def save_gif(frames_rgb, path, fps=10, loop=0):
    """frames_rgb: RGB numpy 배열 리스트 → GIF 저장"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio
        imageio.mimsave(path, frames_rgb, duration=1.0 / fps, loop=loop)
    except Exception:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames_rgb]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=loop)

def main(
    # 경로 기본값: 네가 준 경로 사용
    kitti_root="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training",
    seq="0000",
    det_txt="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/dets_dummy/0000.txt",
    out_gif="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/vis/seq0000_sortcmc_traj.gif",
    # SORT_CMC 하이퍼파라미터
    iou_thres=0.3, max_age=15, min_hits=3, dt=0.1, alpha=0.6,
    # 렌더링 파라미터
    fps=10, trail_len=20
):
    # 1) 이미지 시퀀스 로드
    img_dir = os.path.join(kitti_root, "image_02", seq)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert img_files, f"No images in {img_dir}"

    # 2) det 캐시 로드 & 트래커 초기화 (이미지 크기 전달)
    first = cv2.imread(img_files[0])
    Himg, Wimg = first.shape[:2]
    dets = load_dets(det_txt)
    tracker = SORT_CMC(iou_thres=iou_thres, max_age=max_age, min_hits=min_hits,
                       dt=dt, alpha=alpha, img_size=(Himg, Wimg))

    # 3) 프레임 루프: H(prev→curr) 추정 → 트래커 업데이트 → 그리기
    frames_rgb, trails = [], {}
    prev_img = None
    for fr, img_path in enumerate(img_files):
        curr_img = cv2.imread(img_path)  # BGR
        Hmat = estimate_h(prev_img, curr_img) if prev_img is not None else None
        outs = tracker.update(dets.get(fr, []), Hmat if Hmat is not None else (None))
        vis = draw_frame(curr_img, outs, trails, trail_len)
        frames_rgb.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))  # GIF는 RGB
        prev_img = curr_img

        # 메모리/속도 줄이고 싶으면 샘플링:
        # if fr % 2: continue  # 2배 빠르게

    # 4) GIF 저장
    save_gif(frames_rgb, out_gif, fps=fps, loop=0)
    print("saved gif:", out_gif)

if __name__ == "__main__":
    main()
