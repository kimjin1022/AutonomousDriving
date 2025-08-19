# tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from track import Track   # 바로 전에 만든 7차 상태 칼만필터 트랙 클래스 사용

# ---------------- IoU 유틸 ----------------
def iou_batch(bb_test, bb_gt):
    """
    bb_test: Nx4, bb_gt: Mx4 (x1,y1,x2,y2)
    return: NxM IoU matrix
    """
    if bb_test.size == 0 or bb_gt.size == 0:
        return np.zeros((bb_test.shape[0], bb_gt.shape[0]), dtype=np.float32)

    xx1 = np.maximum(bb_test[:, None, 0], bb_gt[None, :, 0])
    yy1 = np.maximum(bb_test[:, None, 1], bb_gt[None, :, 1])
    xx2 = np.minimum(bb_test[:, None, 2], bb_gt[None, :, 2])
    yy2 = np.minimum(bb_test[:, None, 3], bb_gt[None, :, 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_a = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_b = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / (union + 1e-6)).astype(np.float32)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    detections: Nx4, trackers: Mx4  (둘 다 bbox)
    return:
      matches: Kx2 (det_idx, trk_idx)
      unmatched_dets: list[int]
      unmatched_trks: list[int]
    """
    if trackers.size == 0:
        return np.empty((0, 2), dtype=int), np.arange(detections.shape[0]), np.empty((0), dtype=int)

    iou_mat = iou_batch(detections, trackers)

    # IoU를 비용으로 바꾸고, 임계 미만은 큰 비용으로 게이팅
    cost = 1.0 - iou_mat
    cost[iou_mat < iou_threshold] = 1e6

    det_idx, trk_idx = linear_sum_assignment(cost)

    matches = []
    matched_det, matched_trk = set(), set()
    for d, t in zip(det_idx, trk_idx):
        if cost[d, t] > 1e5:   # 게이트 통과 실패(실질적으로 매칭 안 함)
            continue
        matches.append([d, t])
        matched_det.add(d); matched_trk.add(t)

    unmatched_dets = [d for d in range(detections.shape[0]) if d not in matched_det]
    unmatched_trks = [t for t in range(trackers.shape[0]) if t not in matched_trk]

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches, dtype=int)

    return matches, np.array(unmatched_dets, dtype=int), np.array(unmatched_trks, dtype=int)


# ---------------- 메인 Tracker ----------------
class Tracker:
    """
    SORT 스타일의 멀티오브젝트 트래커 관리자
    - max_age: 연속으로 관측이 끊겨도 유지할 최대 프레임 수
    - min_hits: 안정적으로 출력하기 위한 최소 hit 수
    - iou_threshold: 매칭 임계치
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, debug=False):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.debug = debug

        self.tracks = []      # List[Track]
        self.frame_count = 0  # 처리한 프레임 수

    def update(self, dets):
        """
        dets: Nx4 또는 Nx5([x1,y1,x2,y2,(score)]) np.array
        return: tracked boxes [x1,y1,x2,y2,id] (time_since_update==0인 것만)
        """
        self.frame_count += 1

        # dets 정리: 스코어가 있으면 제거, bbox만 사용
        dets = np.asarray(dets, dtype=float)
        if dets.size == 0:
            det_bboxes = np.empty((0, 4), dtype=float)
        else:
            det_bboxes = dets[:, :4] if dets.shape[1] >= 4 else dets.reshape(-1, 4)

        # 1) 모든 트랙 예측
        trk_bboxes = []
        dead_idx = []
        for i, trk in enumerate(self.tracks):
            pred = trk.predict()     # [x1,y1,x2,y2] 반환
            if np.any(np.isnan(pred)):
                dead_idx.append(i)
            else:
                trk_bboxes.append(pred)
        # NaN 트랙 제거
        for i in reversed(dead_idx):
            self.tracks.pop(i)
        trk_bboxes = np.array(trk_bboxes, dtype=float) if len(trk_bboxes) else np.empty((0, 4), dtype=float)

        # 2) 연계(Association): IoU + 헝가리안
        if det_bboxes.size == 0 and trk_bboxes.size == 0:
            matches = np.empty((0, 2), dtype=int)
            unmatched_dets = np.empty((0), dtype=int)
            unmatched_trks = np.empty((0), dtype=int)
        else:
            matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
                det_bboxes, trk_bboxes, self.iou_threshold
            )

        # 3) 매칭된 트랙 업데이트
        for m in matches:
            det_idx, trk_idx = int(m[0]), int(m[1])
            self.tracks[trk_idx].update(det_bboxes[det_idx])

        # 4) 매칭 안 된 detection → 새 트랙 생성
        for di in unmatched_dets:
            self.tracks.append(Track(det_bboxes[int(di)], max_age=self.max_age))

        # 5) 오래 갱신 안 된 트랙 삭제
        survivors = []
        for trk in self.tracks:
            if trk.time_since_update <= self.max_age:
                survivors.append(trk)
        self.tracks = survivors

        # 6) 출력 만들기 (논문 규칙)
        ret = []
        for trk in self.tracks:
            # 이번 프레임에서 업데이트된 트랙만 (time_since_update == 0)
            if (trk.time_since_update == 0) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                x1, y1, x2, y2 = trk.get_state_bbox()
                ret.append([x1, y1, x2, y2, trk.id])

        return np.array(ret, dtype=float) if len(ret) else np.empty((0, 5), dtype=float)

    # 편의상 현재 트랙 상태를 전부 보고 싶을 때
    def dump_tracks(self):
        infos = []
        for t in self.tracks:
            bb = t.get_state_bbox()
            infos.append({
                "id": t.id,
                "bbox": bb.tolist(),
                "hits": t.hits,
                "age": t.age,
                "miss": t.time_since_update
            })
        return infos
