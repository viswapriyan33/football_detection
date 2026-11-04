import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO Football Object Detection")
    parser.add_argument("--model", default="train/weights/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--source", default="media/test.png", help="Path to image/video or webcam (0 or usb)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image resolution (trained at 1000/1024)")
    args = parser.parse_args()

    # Handle webcam shortcut
    source = args.source.replace("usb", "") or "0" if args.source.startswith("usb") else args.source
    if str(source).isdigit():
        source = int(source)

    # Load model
    model = YOLO(args.model)

    # Run predictions with trained resolution and confidence threshold
    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        show=True,
        stream=True,
        save=True,
        save_dir="output",
    )

    for result in results:
        pass


if __name__ == "__main__":
    main()
