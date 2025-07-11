# scripts/step13_infer_testing_to_cache.py
# 목적: KITTI_Tracking/testing 에 대해 YOLO 추론 → det 캐시(txt) 생성
# 참고: step10_infer_to_cache.py의 main을 재사용

import os
import step10_infer_to_cache as s10

def main(
    kitti_root="/home/jinjinjara1022/AutonomousDriving/datasets/KITTI_Tracking/testing",
    weights="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/scripts/runs/detect/kitti_y8n_1280/weights/best.pt",
    out_dir="/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/dets_yolo_test",
    imgsz=1280, conf=0.25, iou=0.6, device=1, vis_frame_idx=0
):
    # step10 재사용: seq='all' + testing 루트만 전달
    s10.main(
        kitti_root=kitti_root,
        seq="all",
        weights=weights,
        out_dir=out_dir,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        vis_frame_idx=vis_frame_idx
    )

if __name__ == "__main__":
    main()
