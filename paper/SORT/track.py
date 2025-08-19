# track.py
import numpy as np
import math

def bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> z=[u,v,s,r]^T"""
    x1, y1, x2, y2 = map(float, bbox)
    w = x2 - x1
    h = y2 - y1
    u = x1 + w / 2.0
    v = y1 + h / 2.0
    s = w * h                      # area
    r = w / (h + 1e-6)             # aspect ratio
    return np.array([[u], [v], [s], [r]], dtype=float)

def x_to_bbox(x):
    """x=[u,v,s,r,du,dv,ds]^T -> [x1,y1,x2,y2]"""
    u, v, s, r = x[0,0], x[1,0], x[2,0], x[3,0]
    w = math.sqrt(max(0.0, s * r))
    h = 0.0 if w == 0.0 else s / (w + 1e-6)
    x1 = u - w / 2.0
    y1 = v - h / 2.0
    x2 = u + w / 2.0
    y2 = v + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=float)

class Track:
    """SORT 논문 스타일의 7차 상태 칼만 트랙"""
    _next_id = 0

    def __init__(self, init_bbox, max_age=1):
        # 상태: [u, v, s, r, du, dv, ds]^T  (열벡터)
        self.x = np.zeros((7,1), dtype=float)
        self.x[:4] = bbox_to_z(init_bbox)      # 속도항은 0으로 초기화

        # 공분산
        self.P = np.eye(7, dtype=float)
        self.P[4:, 4:] *= 1000.0   # 속도 불확실성 큼
        self.P *= 10.0

        # 상태전이 F (dt=1, r은 상수)
        self.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)

        # 관측행렬 H (u,v,s,r만 관측)
        self.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)

        # 잡음 행렬 (레퍼런스 SORT와 유사한 스케일)
        self.R = np.eye(4, dtype=float)
        self.R[2:, 2:] *= 10.0      # s,r 관측 잡음 큼
        self.Q = np.eye(7, dtype=float)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01

        # 메타
        self.id = Track._next_id; Track._next_id += 1
        self.time_since_update = 0
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.max_age = max_age
        self.history = []

    # ---------- Kalman steps ----------
    def predict(self):
        """다음 프레임 상태 예측"""
        # s + ds가 음수로 내려가는 비정상 방지 (SORT 가드)
        if (self.x[2,0] + self.x[6,0]) <= 0:
            self.x[6,0] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1
        self.history.append(x_to_bbox(self.x))
        return self.history[-1]

    def update(self, meas_bbox):
        """관측 bbox로 상태 보정 (meas_bbox: [x1,y1,x2,y2])"""
        z = bbox_to_z(meas_bbox)
        y = z - (self.H @ self.x)                         # residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain
        self.x = self.x + (K @ y)
        I = np.eye(7)
        self.P = (I - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.history = []

    # ---------- Helpers ----------
    def get_state_bbox(self):
        """현재 상태를 [x1,y1,x2,y2]로 반환"""
        return x_to_bbox(self.x)

    def get_state_x(self):
        """현재 상태 벡터(7x1) 복사본"""
        return self.x.copy()