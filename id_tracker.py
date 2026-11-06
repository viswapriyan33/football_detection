class IdTracker:
    def __init__(self, iou_thresh=0.3, max_age=20):
        # self.t: Minimum IoU overlap needed to consider it the same player (30%)
        # self.mx: Maximum frames a player can disappear before their ID is deleted (20 frames)
        # self.tr: Dictionary storing active tracks -> {track_id: [bounding_box, missed_frames]}
        # self.nid: Next available ID counter (0, 1, 2, ...)
        self.t, self.mx, self.tr, self.nid = iou_thresh, max_age, {}, 0

    def _iou(self, a, b):
        # Calculate Intersection Over Union (IoU) between bounding box a and b
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        
        # Overlapping intersection area
        i = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Combined union area
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
        
        return i / u if u > 0 else 0

    def update(self, boxes, mask):
        # ids: Final assigned ID for each detected box (-1 by default)
        # used: Set of track IDs already assigned in the current frame
        ids, used = [-1] * len(boxes), set()

        # Step 1: Match newly detected boxes to existing tracks
        for i, box in enumerate(boxes):
            if not mask[i]:  # Skip non-players (e.g. ball, referee)
                continue

            best, bv = -1, self.t  # Look for existing track with highest IoU > 0.3
            for tid, (tbox, _) in self.tr.items():
                if tid not in used and (v := self._iou(box, tbox)) > bv:
                    best, bv = tid, v

            # If no existing player overlaps enough, spawn a new ID
            if best < 0:
                best, self.nid = self.nid, self.nid + 1

            ids[i] = best
            used.add(best)
            self.tr[best] = [box, 0]  # Update track location, reset missed frames counter

        # Step 2: Age out and delete tracks for players who left the frame
        for tid in list(self.tr):
            if tid not in used:
                self.tr[tid][1] += 1
                if self.tr[tid][1] > self.mx:  # Exceeded 20 frames without being seen
                    del self.tr[tid]

        return ids