from ultralytics import YOLO

DATA = "/home/jinjinjara1022/AutonomousDriving/ObjectTracking/datasets/kitti_yolo/kitti_yolo.yaml"

def main(model="yolov8n.pt", name="kitti_y8n_1280", imgsz=1280, epochs=5, batch=16, device=1):
    m = YOLO(model)  # 프리트레인 로드
    m.train(
        data=DATA, imgsz=imgsz, epochs=epochs, batch=batch,
        device=device, workers=8, name=name
    )

if __name__ == "__main__":
    main()
