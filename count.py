import cv2
from ultralytics import YOLO, solutions

def main():
    # 1. Initialize YOLO-World model
    # We use YOLO-World for open-vocabulary (zero-shot) detection
    try:
        # Using medium YOLO-World for better zero-shot accuracy
        model = YOLO("yolov8m-worldv2.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Define the custom classes we are looking for
    # Added slightly more variations to ensure all sacks are detected
    custom_classes = ["sack", "gunny sack", "jute sack", "large white bag", "sandbag"]
    model.set_classes(custom_classes)

    # 2. Open the input video
    video_path = "Problem Statement Scenario3.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
        
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 3. Define the counting region
    # We'll set a vertical line in the middle of the frame
    # Modify this if the truck/warehouse layout requires a horizontal line or different position
    line_points = [(w // 2, 0), (w // 2, h)]

    # 4. Initialize Object Counter
    # Note: solutions.ObjectCounter API might differ slightly across ultralytics versions.
    # We will use the standard object counter.
    # `set_classes` will give them indices 0, 1, 2, 3, 4 in model.names
    classes_indices = list(range(len(custom_classes)))

    try:
        counter = solutions.ObjectCounter(
            region=line_points,
            classes=classes_indices,
            conf=0.03,
            iou=0.5,
        )
    except TypeError:
        # Fallback if region/reg_pts differ
        counter = solutions.ObjectCounter(
            reg_pts=line_points,
            classes=classes_indices,
            conf=0.03,
            iou=0.5,
        )

    # 5. Setup Video Writer for output
    output_path = "output_counted_bags.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Processing video {video_path}...")
    print(f"Resolution: {w}x{h}, FPS: {fps}")

    frame_count = 0
    # 6. Video processing loop
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...")

        # In Ultralytics 8.4+, ObjectCounter.process() handles both tracking and counting
        # Tracking config like conf and iou are passed during ObjectCounter initialization.
        im0 = counter.process(im0).plot_im
        
        # Write to output video
        video_writer.write(im0)

    # 7. Release resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete! Output saved to: {output_path}")

if __name__ == "__main__":
    main()
