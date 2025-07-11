# KITTI 2D Multi-Object Tracking (YOLO + SORT/CMC)

KITTI Tracking 데이터셋으로 **검출 → 추적 → 평가 → 테스트**

---

## 진행 내용
- **데이터 준비**
  - KITTI 라벨 파싱, 클래스 매핑  
    *(Car/Van/Truck→car, Pedestrian/Person/Person_sitting→pedestrian, Cyclist)*
  - YOLO 학습용으로 **이미지별 라벨 변환**(KITTI→YOLO)
- **검출(Detection)**
  - Ultralytics YOLO 학습/추론 → **검출 캐시(txt)** 생성  
    *(포맷: `frame, class_id, score, left, top, right, bottom`)*
- **추적(Tracking)**
  - IOU 미니멀 → **SORT(칼만+헝가리안)** → **SORT+CMC(호모그래피 보정)**
  - 프레임별 **ID/박스/궤적** 시각화(GIF)
- **평가/테스트**
  - Val: **MOTA / IDF1** (motmetrics)  
  - Test: **KITTI 제출 포맷(txt)** 생성

---

## 결과 예시
> 아래 파일들을 `assets/` 폴더에 두면 바로 보입니다.

### GT 1프레임 시각화
![gt_vis](ObjectTracking/outputs/vis/step1_gt_seq0000_f0000.jpg)

### YOLO 검출 1프레임
![yolo_det](ObjectTracking/outputs/vis/step2_seq0000_f0000.jpg)

### iou
![sort_demo](ObjectTracking/outputs/vis/step5_seq0000_iou_traj.gif)

### SORT + CMC (카메라 모션 보정)
![sortcmc_demo](ObjectTracking/outputs/vis/step7_5_seq0000_sortcmc_traj.gif)

### 테스트 시퀀스 GIF (시각 점검용)
![test_demo](ObjectTracking/outputs/vis/test_seq0000_sortcmc.gif)

---

## 데이터셋
- **KITTI Tracking**
  - `training/` : 학습/평가
  - `testing/`  : 제출 포맷 생성(정답 없음)
- YOLO 학습용 변환 스크립트

---

## 코드 (역할 요약)
- **데이터/시각화**
  - `step1_vis_gt.py` : GT 로딩 + 1프레임 박스 표시  
  - `step2_classmap_stats.py` : 클래스 매핑/필터 + 통계
- **검출 캐시**
  - `step3_detcache_dummy.py` : 더미 검출 캐시(파이프라인 점검용)  
  - `step10_infer_to_cache.py` : YOLO 추론 → **검출 캐시(txt)**
- **트래커**
  - `step4_tracker_iou.py` : IOU만으로 매칭하는 미니멀 트래커  
  - `step6_sort_minimal.py` : **SORT(칼만+헝가리안)**  
  - `step7_sort_cmc_minimal.py` : **SORT + CMC(호모그래피 보정)**
- **시각화(GIF)**
  - `step5_gif_iou_with_traj.py` : IOU 트래커 GIF  
  - `step6_5_sort_gif_traj.py` : SORT GIF  
  - `step7_5_sort_cmc_gif_traj.py` : SORT+CMC GIF  
  - `step15_render_test_gif.py` : 테스트 시퀀스 GIF
- **학습/변환/평가**
  - `step8_build_yolo_dataset.py` : KITTI→YOLO 변환(+split)  
  - `step9_train_yolo.py` : YOLO 학습  
  - `step12_eval_motmetrics.py` : **MOTA/IDF1** 평가
- **테스트 제출**
  - `step13_infer_testing_to_cache.py` : 테스트셋 검출 캐시 생성  
  - `step14a_track_one_seq_kitti.py` : (한 시퀀스/구간) 추적 → **KITTI 제출 포맷(txt)**

---

## 폴더 구조 (핵심)

- ├─ scripts/ # 위 스크립트들
- ├─ outputs/
- │ ├─ dets_dummy/ # 더미 검출 캐시
- │ ├─ dets_yolo/ # YOLO 검출 캐시 (train/val)
- │ ├─ dets_yolo_test/ # YOLO 검출 캐시 (testing)
- │ ├─ tracks_sort_min/ # SORT 결과(txt)
- │ ├─ tracks_sort_cmc/ # SORT+CMC 결과(txt)
- │ ├─ tracks_test_kitti/ # 테스트 제출 포맷(txt)
- │ └─ vis/ # 이미지/GIF 결과
- └─ datasets/
- └─ kitti_yolo/ # YOLO 학습 변환 결과

---