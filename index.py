import os
import argparse
import cv2
import torch
from ultralytics import YOLO

# Import our custom modules
from custom_markers import (
    draw_annotations, 
    jersey_color, 
    TeamTracker, 
    PossessionTracker, 
    draw_possession
)
from id_tracker import IdTracker
from speed_tracker import SpeedTracker, draw_speed

# Optional flags: Set to True if you also want to measure the speed of the ball or referee
SPEED_BALL, SPEED_REF = False, False

# 1. Argument Parser
p = argparse.ArgumentParser(description="Advanced YOLO Football AI: Teams, Possession & Speed")
p.add_argument('--model', default='custom_model.pt', help="Path to your trained YOLO weights")
p.add_argument('--source', default='media/match_red_white.mp4', help="Path to video/image or webcam")
p.add_argument('--stream', action='store_true', help="Show live preview window")
p.add_argument('--device', default=None, help="cuda / cpu")
p.add_argument('--imgsz', type=int, default=640, help="Inference resolution (640 for speed)")
p.add_argument('--conf', type=float, default=0.25, help="Confidence threshold")
p.add_argument('--vid_stride', type=int, default=2, help="Frame step (2 for real match speed)")
a = p.parse_args()

# 2. Hardware selection (Windows CUDA vs CPU)
if a.device is None:
    a.device = '0' if torch.cuda.is_available() else 'cpu'

# 3. Handle inputs & file types
src = a.source.lstrip('usb') if a.source.startswith('usb') else a.source
is_img = src.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))

if not is_img and not a.stream:
    a.stream = input('Stream preview? [y/N]: ').strip().lower() == 'y'

# 4. Prepare output folder & video settings
os.makedirs('output', exist_ok=True)
fps = 30.0
if not is_img:
    c = cv2.VideoCapture(src)
    fps = c.get(cv2.CAP_PROP_FPS) or 30.0
    c.release()

writer, frames = None, 0
use_half = (a.device != 'cpu')

# 5. Initialize all 4 tracking engines outside the loop
tracker, pos_tracker = TeamTracker(), PossessionTracker()
spd_tracker, id_tracker = SpeedTracker(fps=fps), IdTracker()

# 6. Main Inference & Tracking Loop
try:
    for r in YOLO(a.model).predict(
        source=src, 
        stream=True, 
        device=a.device, 
        half=use_half, 
        imgsz=a.imgsz, 
        conf=a.conf, 
        vid_stride=a.vid_stride, 
        verbose=False
    ):
        img = r.orig_img.copy()

        # Extract bounding box coordinates, class IDs, and labels
        b = r.boxes.xyxy.cpu().numpy()
        c = r.boxes.cls.cpu().numpy()
        n = r.names

        # Step 1: Cluster players into Team 0 vs Team 1
        colors = [jersey_color(img, int(x[0]), int(x[1]), int(x[2]), int(x[3])) for x in b]
        team_ids, team_colors = tracker.assign(colors)

        # Step 2: Locate ball and update possession statistics
        ball = next((x for x, cid in zip(b, c) if 'ball' in n[int(cid)].lower()), None)
        pos_tracker.update(ball, b, team_ids)

        # Step 3: Filter which detected objects should be tracked for speed
        is_player = [
            (
                SPEED_BALL if ('ball' in (lbl := n[int(cid)].lower()) or 'football' in lbl)
                else SPEED_REF if 'referee' in lbl
                else team_ids[i] >= 0
            )
            for i, cid in enumerate(c)
        ]

        # Step 4: Track persistent player IDs across consecutive frames
        ids = id_tracker.update(b, is_player)

        # Step 5: Compute camera-compensated sprint speeds in km/h
        speeds = spd_tracker.update(img, b, ids)

        # Step 6: Render all broadcast overlays
        draw_annotations(img, b, c, n, team_ids, team_colors)         # Floating triangles & referee rings
        draw_possession(img, pos_tracker.percentages(), team_colors)  # Live possession bar
        draw_speed(img, b, speeds, team_ids, team_colors)             # Live speed badges under feet

        # If single image input: save and exit
        if is_img:
            cv2.imwrite('output/result.jpg', img)
            print('Saved -> output/result.jpg')
            break

        # Initialize video writer on first frame
        if not writer:
            writer = cv2.VideoWriter('output/result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (img.shape[1], img.shape[0]))
        
        writer.write(img)
        frames += 1

        # Live playback controls
        if a.stream:
            cv2.imshow('Detection', img)
            if cv2.waitKey(1) == ord('q'):
                break
        elif frames % 30 == 0:
            print(f'frames: {frames}', end='\r')

finally:
    # 7. Safe video finalization
    if writer:
        writer.release()
        print(f'\nVideo saved -> output/result.mp4 ({frames} frames)')
    cv2.destroyAllWindows()