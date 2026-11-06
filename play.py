import cv2
import time
import os
import sys

def main():
    # Allow user to specify video number: python play.py 1 or python play.py 2
    match_choice = sys.argv[1] if len(sys.argv) > 1 else "2"
    
    if match_choice == "1":
        # Match 1: Red vs White kits
        candidates = [
            "output/football_2min_h264.mp4",
            "output/football_2min_smooth.mp4",
            "output/smooth_demo.mp4"
        ]
        title_suffix = "Match 1 [Red vs White Kits]"
    else:
        # Match 2: Derby (Navy Blue vs White kits)
        candidates = [
            "output/derby_2min_smooth.mp4",
            "output/derby_annotated.mp4",
            "output/football_2min_h264.mp4",
            "output/smooth_demo.mp4"
        ]
        title_suffix = "Match 2 [Navy Blue vs White Kits]"

    video_path = None
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 1000:
            video_path = c
            break

    if not video_path:
        print("Error: No rendered output video found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    window_name = f"AI Football Tracking System - {title_suffix}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("=" * 65)
    print("⚽ AI Football Analytics - Smooth Real-Time Player")
    print("=" * 65)
    print(f"Playing: {video_path}")
    print(f"Theme: {title_suffix}")
    print(f"Speed: {fps:.1f} FPS (Smooth Broadcast Tempo, Zero Lag)")
    print("\nControls:")
    print(" • Press 'f' to toggle Fullscreen")
    print(" • Press 'Space' to Pause / Resume")
    print(" • Press 'q' to Quit")
    print(" • The video loops continuously for your screen recording!")
    print("=" * 65)

    is_fullscreen = False
    paused = False

    while True:
        if not paused:
            t_start = time.time()
            ret, frame = cap.read()
            
            # Loop seamlessly when reaching the end of clip
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.imshow(window_name, frame)

            # Precise frame-rate timing so playback is 100% natural and smooth
            elapsed = time.time() - t_start
            wait_time = max(1, int((frame_delay - elapsed) * 1000))
        else:
            wait_time = 50

        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # Spacebar
            paused = not paused
        elif key == ord('f'):  # Toggle Fullscreen
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
