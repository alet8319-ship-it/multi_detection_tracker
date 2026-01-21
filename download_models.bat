@echo off
echo Downloading MediaPipe model files...
echo.

echo [1/3] Downloading hand detection model...
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
echo Done: Hand model downloaded
echo.

echo [2/3] Downloading pose detection model...
curl -L -o pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
echo Done: Pose model downloaded
echo.

echo [3/3] Downloading object detection model...
curl -L -o efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite
echo Done: Object model downloaded
echo.

echo All models downloaded successfully!
echo You can now run: python multi_detection_tracker.py
pause