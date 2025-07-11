# scripts/step10_infer_to_cache.py
# 목적: YOLO 가중치로 KITTI Tracking 이미지에 추론 → det 캐시(txt) 생성
# 포맷: frame, class_id, score, left, top, right, bottom  (콤마 구분)

import os, glob, csv, cv2
from ultralytics import YOLO

def infer_seq(model, img_dir, out_txt, imgsz=1280, conf=0.25, iou=0.6,
              device=0, vis_frame_idx=None, out_vis=None):
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert img_files, f"No images in {img_dir}"

    with open(out_txt, "w", newline="") as f:
        writer = csv.writer(f)
        # stream=True로 메모리 절약, 결과는 원본 해상도 좌표(xyxy)
        results = model.predict(img_files, imgsz=imgsz, conf=conf, iou=iou,
                                device=device, stream=True, verbose=False)
        for frame_idx, res in enumerate(results):
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy  = boxes.xyxy.cpu().numpy()          # (N,4) in pixels
            confs = boxes.conf.cpu().numpy()          # (N,)
            clss  = boxes.cls.cpu().numpy().astype(int)  # (N,)
            for (l,t,r,b), s, c in zip(xyxy, confs, clss):
                writer.writerow([frame_idx, int(c), f"{float(s):.3f}",
                                 f"{float(l):.1f}", f"{float(t):.1f}",
                                 f"{float(r):.1f}", f"{float(b):.1f}"])
            # 선택: 특정 프레임 시각화 저장
            if vis_frame_idx is not None and frame_idx == vis_frame_idx and out_vis:
                img = cv2.imread(res.path)  # BGR
                for (l,t,r,b), s, c in zip(xyxy, confs, clss):
                    cv2.rectangle(img, (int(l),int(t)), (int(r),int(b)), (0,255,0), 2)
                    cv2.putText(img, f"{c}:{s:.2f}", (int(l), max(12, int(t)-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
                os.makedirs(os.path.dirname(out_vis) or ".", exist_ok=True)
                cv2.imwrite(out_vis, img)

def main(
    # 경로 기본값 (네 환경)
    kitti_root="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/training",
    seq="0000",   # "0000" 또는 "all"
    weights="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/scripts/runs/detect/kitti_y8n_1280/weights/best.pt",
    out_dir="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/dets_yolo",
    # 추론 하이퍼파라미터
    imgsz=1280, conf=0.25, iou=0.6, device=0,
    # 시각화 옵션
    vis_frame_idx=0
):
    model = YOLO(weights)

    img_root = os.path.join(kitti_root, "image_02")
    seqs = sorted([d for d in os.listdir(img_root) if d.isdigit()]) if seq=="all" else [seq]

    for s in seqs:
        img_dir = os.path.join(img_root, s)
        out_txt = os.path.join(out_dir, f"{s}.txt")
        out_vis = os.path.join(os.path.dirname(out_dir), "vis", f"step10_{s}_f{vis_frame_idx:04d}.jpg")
        print(f"[SEQ {s}] infer → {out_txt}")
        infer_seq(model, img_dir, out_txt, imgsz=imgsz, conf=conf, iou=iou,
                  device=device, vis_frame_idx=vis_frame_idx, out_vis=out_vis)
        print(f"saved: {out_txt}")
        print(f"vis  : {out_vis}")

if __name__ == "__main__":
    main()
