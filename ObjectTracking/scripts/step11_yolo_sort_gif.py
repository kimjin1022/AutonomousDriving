# step11_yolo_sort_gif.py
import step6_5_sort_gif_traj as sort_gif
import step7_5_sort_cmc_gif_traj as sort_cmc_gif
import os

def main():
    seq = "0000"
    det_txt = f"/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/dets_yolo/{seq}.txt"
    vis_dir = "/home/jinjinjara1022/AutonomousDriving/ObjectTracking/outputs/vis"

    # 1) SORT 궤적 GIF
    print(f"[SORT] seq {seq} → GIF 생성 중...")
    sort_gif.main(
        seq=seq,
        det_txt=det_txt,
        fps=10,
        iou_thres=0.35,
        max_age=20,
        min_hits=3,
        dt=0.1,
        trail_len=20
    )
    print(f"[SORT] 완료 → {os.path.join(vis_dir, f'seq{seq}_sort_traj.gif')}")

    # 2) SORT+CMC 궤적 GIF
    print(f"[SORT+CMC] seq {seq} → GIF 생성 중...")
    sort_cmc_gif.main(
        seq=seq,
        det_txt=det_txt,
        fps=10,
        iou_thres=0.35,
        max_age=20,
        min_hits=3,
        dt=0.1,
        alpha=0.6,   # 카메라 보정 비율
        trail_len=20
    )
    print(f"[SORT+CMC] 완료 → {os.path.join(vis_dir, f'seq{seq}_sortcmc_traj.gif')}")

if __name__ == "__main__":
    main()
