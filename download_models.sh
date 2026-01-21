#!/bin/bash

echo "Downloading MediaPipe model files..."
echo ""

# Hand detection model
echo "[1/3] Downloading hand detection model..."
wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
echo "✓ Hand model downloaded"
echo ""

# Pose detection model
echo "[2/3] Downloading pose detection model..."
wget -O pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
echo "✓ Pose model downloaded"
echo ""

# Object detection model
echo "[3/3] Downloading object detection model..."
wget -O efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite
echo "✓ Object model downloaded"
echo ""

echo "All models downloaded successfully!"
echo "You can now run: python multi_detection_tracker.py"