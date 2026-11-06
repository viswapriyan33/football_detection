import cv2
import numpy as np
from collections import deque


class SpeedTracker:
    def __init__(self, fps=30.0, h_m=1.80, window=8, alpha=0.4, max_kmh=40.0, deadband=2.0):
        # fps: Video frame rate (30.0)
        # h_m: Average player height in meters (1.80m)
        # window: Number of past frames used for velocity smoothing (8 frames)
        # alpha: Exponential smoothing factor (0.4)
        # max_kmh: Max physiological sprint cap (40 km/h, Mbappe top speed ~37 km/h)
        # deadband: Speeds below 2.0 km/h are rounded to 0.0 (prevents jitter while standing)
        self.fps, self.h_m, self.win, self.a, self.mx, self.db = fps, h_m, window, alpha, max_kmh, deadband
        self.prev, self.cum, self.hist, self.sm, self.f = None, np.eye(3, dtype=np.float32), {}, {}, 0

    def _cam(self, frame, boxes):
        """Camera Motion Compensation using Background Optical Flow & Affine RANSAC."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev is not None:
            H, W = g.shape
            # Create a full white mask of the entire frame
            m = np.full((H, W), 255, np.uint8)
            # Black out (mask=0) all player bounding boxes
            for x1, y1, x2, y2 in boxes:
                m[max(0, int(y1)-15):min(H, int(y2)+15), max(0, int(x1)-15):min(W, int(x2)+15)] = 0

            # Find stable feature points exclusively on the pitch grass and stadium
            p = cv2.goodFeaturesToTrack(self.prev, 300, 0.01, 10, mask=m, blockSize=7)
            if p is not None and len(p) >= 10:
                # Track where the pitch turf moved between frames
                q, s, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p, None)
                ok = s.ravel().astype(bool)
                
                # Estimate the camera pan/zoom matrix M using RANSAC outlier rejection
                if ok.sum() >= 10 and (M := cv2.estimateAffinePartial2D(q[ok], p[ok], method=cv2.RANSAC)[0]) is not None:
                    # Accumulate camera movement into the global world matrix
                    self.cum = self.cum @ np.vstack([M, [0, 0, 1]]).astype(np.float32)

        self.prev = g

    def update(self, frame, boxes, ids):
        """Calculates real-world speed (km/h) for every player."""
        self._cam(frame, boxes)
        out = [None] * len(boxes)

        for i, (box, tid) in enumerate(zip(boxes, [int(t) for t in ids])):
            if tid < 0:
                continue

            x1, y1, x2, y2 = box
            bh = max(float(y2 - y1), 1.0)
            
            # 1. Dynamic scale: Meters-Per-Pixel based on player height
            mpp = self.h_m / bh

            # 2. Project player's feet position into stabilized world coordinates
            wx, wy, _ = self.cum @ np.array([(x1 + x2) / 2, y2, 1.0], np.float32)

            # 3. Rolling window buffer of recent positions
            h = self.hist.setdefault(tid, deque(maxlen=self.win))
            h.append((float(wx), float(wy), self.f, mpp))

            if len(h) < 3 or (dt := (h[-1][2] - h[0][2]) / self.fps) <= 0:
                continue

            # 4. Physical velocity calculation (pixels -> meters -> km/h)
            amp = sum(p[3] for p in h) / len(h)
            dist_meters = np.hypot(h[-1][0] - h[0][0], h[-1][1] - h[0][1]) * amp
            kmh = min(self.mx, (dist_meters / dt) * 3.6)

            # 5. EMA Smoothing & Deadband filter
            kmh = self.a * kmh + (1 - self.a) * self.sm.get(tid, kmh)
            self.sm[tid] = kmh = 0.0 if kmh < self.db else kmh
            out[i] = kmh

        self.f += 1
        return out


def draw_speed(img, boxes, speeds, team_ids=None, team_colors=None, min_show=1.0):
    """Renders the speed badge below each moving player's feet."""
    for i, (box, sp) in enumerate(zip(boxes, speeds)):
        # 1. Do not display badge if speed is unknown or player is standing still (< 1.0 km/h)
        if sp is None or sp < min_show:
            continue

        x1, y1, x2, y2 = map(int, box)
        txt = f"{sp:.1f} km/h"

        # 2. Measure text width & height to calculate padding for the badge
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        bw, bh = tw + 12, th + 8

        # 3. Center the badge horizontally beneath the player's feet (y2 + 10)
        bx, by = (x1 + x2) // 2 - bw // 2, y2 + 10

        # 4. Draw dark background container pill
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)

        # 5. Render sharp white speed text inside the badge
        cv2.putText(
            img, 
            txt, 
            (bx + 6, by + bh - 5), 
            cv2.FONT_HERSHEY_DUPLEX, 
            0.5, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )