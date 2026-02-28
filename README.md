# Bag Counting Video Analytics

This project analyzes video footage of workers moving gunny/jute sacks from trucks to warehouses and automatically tracks and counts the number of sacks. It outputs a new video file with bounded boxes, tracked movement lines, and a cumulative object counter.

## Approach & Technology
To achieve accurate object counting without needing a massive custom-labeled dataset of "sacks", this script utilizes **[YOLO-World](https://docs.ultralytics.com/models/yolo-world/)** by Ultralytics. 
YOLO-World is an open-vocabulary, zero-shot object detection model. It allows us to supply descriptive text prompts (e.g., `"sack", "gunny sack", "jute sack"`) instead of integer class IDs, and the model attempts to find objects matching that description out-of-the-box.

We pair the YOLO-World medium model (`yolov8m-worldv2.pt`) with Ultralytics' `solutions.ObjectCounter` module to handle tracking IDs (using ByteTrack) and region/line intersection counting.

### Prerequisites & Dependencies
Ensure you have Python 3.8+ installed.

1. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
2. **Install Required Packages:**
   This project relies on `ultralytics` for the YOLO models and ObjectCounter, and `opencv-python` for video writing.
   ```bash
   pip install ultralytics opencv-python shapely lapx
   ```
3. **Install OpenAI CLIP:**
   YOLO-World requires the `clip` module to encode the text prompts.
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

## How to Use
1. **Place your Input Video:** Place your video file inside the project directory (e.g., `Problem Statement Scenario3.mp4`).
2. **Configure the Script:** Open `count_sacks.py` and modify the following parameters if needed:
   * **`video_path`**: Upate the video file path to match your input file.
   * **`line_points`**: By default, the counting line is drawn vertically straight down the middle of the video. Depending on the camera angle of the new video, you might want to adjust the X/Y coordinates of `line_points` so they cross the path the workers take.
   * **Thresholds**: Inside the `ObjectCounter` initialization, you can tweak `conf` (Confidence Threshold - lower this if the model is missing sacks, increase it if there are false positives) and `iou` (Intersection over Union for the tracker).

3. **Run the Script:**
   ```bash
   python count_sacks.py
   ```
   The model will automatically download `yolov8m-worldv2.pt` on the first run. The script will output its progress frame-by-frame.

4. **Output:**
   Once finished, a new file named `output_counted_bags.mp4` will be created in the current directory showing the tracked objects and the real-time count in the top-right corner.
