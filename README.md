# Multi-Detection Tracker

A real-time computer vision application that detects and tracks hands, body pose, and objects using MediaPipe and OpenCV.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- 🖐️ **Hand Detection** - Track up to 2 hands with 21 landmarks each
- 🧍 **Pose Detection** - Full body skeleton tracking with 33 landmarks
- 📦 **Object Detection** - Detect 90+ common objects from COCO dataset
- ⚡ **Real-time Processing** - Live camera feed with FPS counter
- 🎮 **Interactive Controls** - Toggle each detection feature on/off
- 🎨 **Visual Feedback** - Color-coded overlays for different detections

## Demo

The application displays:
- Green/Blue overlays for hand landmarks and connections
- Yellow/Cyan skeleton for body pose
- Magenta bounding boxes for detected objects
- Real-time FPS counter
- Status indicators for each detection mode

## Requirements

- Python 3.8 or higher
- Webcam/Camera
- Linux/macOS/Windows

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/alet8319-ship-it/multi_detection_tracker.git
cd multi_detection_tracker
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Files

The application requires three MediaPipe model files. Run the provided script:

**Linux/macOS:**
```bash
chmod +x download_models.sh
./download_models.sh
```

**Windows:**
```bash
download_models.bat
```

**Or download manually:**

```bash
# Hand detection model
wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# Pose detection model
wget -O pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

# Object detection model
wget -O efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite
```

## Usage

### Running the Application

```bash
python main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `h` | Toggle hand detection ON/OFF |
| `p` | Toggle pose detection ON/OFF |
| `o` | Toggle object detection ON/OFF |
| `q` | Quit application |

### Expected Output

```
Multi-detection started!
Press 'h' to toggle hand detection
Press 'p' to toggle pose detection
Press 'o' to toggle object detection
Press 'q' to quit
```

## Project Structure

```
multi-detection-tracker/
├── main.py   # Main application
├── requirements.txt              # Python dependencies
├── download_models.sh            # Model download script (Linux/macOS)
├── download_models.bat           # Model download script (Windows)
├── README.md                     # This file
├── LICENSE                       # License file
├── .gitignore                   # Git ignore file
├── hand_landmarker.task         # Hand detection model (not in repo)
├── pose_landmarker_heavy.task   # Pose detection model (not in repo)
└── efficientdet_lite0.tflite    # Object detection model (not in repo)
```

## Troubleshooting

### Camera Not Opening
```
Error: Could not open camera
```
**Solution:** Check if your camera is being used by another application. Try changing camera index from `0` to `1` in the code.

### MediaPipe Import Error
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```
**Solution:** Install the correct MediaPipe version:
```bash
pip install mediapipe==0.10.31
```

### Model File Not Found
```
FileNotFoundError: hand_landmarker.task
```
**Solution:** Ensure all model files are downloaded and placed in the same directory as the script. Run the download script again.

### Low FPS
**Solution:** 
- Disable features you don't need using keyboard controls
- Close other applications
- Reduce `max_results` in object detection
- Use a lighter pose model (replace `pose_landmarker_heavy.task` with `pose_landmarker_lite.task`)

## Configuration

You can modify detection settings in the code:

```python
# Adjust number of hands (1-2)
num_hands=2

# Adjust confidence thresholds (0.0-1.0)
min_hand_detection_confidence=0.5
min_pose_detection_confidence=0.5

# Adjust object detection results (1-10)
max_results=5
score_threshold=0.5
```

## Dependencies

- **opencv-python** (4.8.0+) - Computer vision and camera interface
- **mediapipe** (0.10.30+) - Hand, pose, and object detection
- **numpy** (1.24.0+) - Array operations

## Performance Tips

1. **Optimize for your use case** - Disable unused detection features
2. **Lighting matters** - Ensure good lighting for better detection accuracy
3. **Camera quality** - Higher resolution cameras provide better results
4. **Background** - Plain backgrounds improve detection accuracy
5. **Distance** - Stand 1-2 meters from camera for best pose detection

## Detected Objects

The object detector can identify 90+ objects including:
- **People & Animals:** person, dog, cat, bird, horse
- **Vehicles:** car, motorcycle, bus, train, truck
- **Indoor Objects:** chair, table, laptop, keyboard, phone
- **Food:** banana, apple, sandwich, pizza, cake
- And many more from the COCO dataset!

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Google's ML solutions for live perception
- [OpenCV](https://opencv.org/) - Computer vision library
- [COCO Dataset](https://cocodataset.org/) - Object detection dataset

## Author

**Your Name**
- GitHub: [@alet8319-ship-it](https://github.com/alet8319-ship-it)
- Email: alet8319@gmail.com

## Support

If you encounter issues or have questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Search [existing issues](https://github.com/alet8319-ship-it/multi_detection_tracker/issues)
3. Create a [new issue](https://github.com/alet8319-ship-it/multi_detection_tracker/issues/new)

---

⭐ Star this repository if you find it helpful!# multi_detection_tracker
