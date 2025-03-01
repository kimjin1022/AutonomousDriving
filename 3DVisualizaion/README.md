# KITTI 3D Visualization

KITTI Object Detection 데이터셋을 활용해  
**LiDAR 포인트 클라우드와 3D 객체 라벨을 직관적으로 이해**

---

## 진행 내용
- **LiDAR 데이터 이해**
  - 거리·고도·밀도에 따른 포인트 분포 확인
  - ROI 설정과 다운샘플링으로 가시성 개선
- **3D 라벨 구조 파악**
  - `dims[h,w,l]`, `loc[x,y,z]`, `ry` 의미 및 좌표계 정리
- **다양한 시점에서 시각화**
  - Image + 2D GT
  - LiDAR 멀티뷰 (3D, BEV, Side, Front)
  - 3D GT Boxes (LiDAR 공간, 두 시점)
  - Image + Projected 3D Boxes

---

## 결과 예시
### Image + 2D GT
![image_2d_gt](assets/Image_2D.png)

### LiDAR Multi-View
![lidar_views](assets/Lider_views.png)

### 3D GT Boxes (Two Views)
![3d_boxes_views](assets/GTBoxes_Lider_2.png)

### Image + Projected 3D Boxes
![image_3d_proj](assets/Image_3D.png)

---

## 데이터셋
- **KITTI Object Detection**
  - `image_2/` : RGB 이미지
  - `label_2/` : 2D & 3D 라벨
  - `velodyne/` : LiDAR 포인트 클라우드 (`.bin` : x,y,z,intensity)
  - `calib/` : 카메라-라이다 캘리브레이션 (`P2`, `R0_rect`, `Tr_velo_to_cam`)


---

## 메모
- 이 시각화를 통해 **라이다 데이터 구조, 3D 라벨 의미**을 직관적으로 이해할 수 있음  
- 향후 3D Detection / BEV 실험에 활용
