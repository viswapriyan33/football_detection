import math


class PossessionTracker:
    def __init__(self, proximity_px=80):
        # proximity_px: Maximum distance in pixels between player and ball to count as possession
        self.prox = proximity_px
        self.counts = [0, 0]  # [Team 0 frames, Team 1 frames]

    def update(self, b_box, p_boxes, t_ids):
        # If no ball is detected in this frame, nobody has possession
        if not b_box:
            return -1

        # Center point of the ball (bx, by)
        bx = (b_box[0] + b_box[2]) / 2
        by = (b_box[1] + b_box[3]) / 2

        # Find the single closest player to the ball
        dist, tid = min(
            (
                (
                    math.hypot(
                        bx - max(x1, min(x2, bx)),
                        by - max(y1, min(y2, by))
                    ),
                    t
                )
                for (x1, y1, x2, y2), t in zip(p_boxes, t_ids) if t >= 0
            ),
            default=(float('inf'), -1)
        )

        # If the closest player is within 80 pixels of the ball, award possession
        if dist <= self.prox:
            self.counts[tid] += 1
            return tid

        # Loose ball / pass in flight
        return -1

    def percentages(self):
        # Calculate possession percentage for both teams: [Team 0 %, Team 1 %]
        tot = sum(self.counts)
        return [100 * c / tot for c in self.counts] if tot else [50.0, 50.0]