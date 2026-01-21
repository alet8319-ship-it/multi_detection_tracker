import cv2
import time
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_landmarker = HandLandmarker.create_from_options(hand_options)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_landmarker = PoseLandmarker.create_from_options(pose_options)

object_options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='efficientdet_lite0.tflite'),
    running_mode=VisionRunningMode.VIDEO,
    max_results=5,
    score_threshold=0.5
)
object_detector = ObjectDetector.create_from_options(object_options)

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), 
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32) 
]

prev_time = 0
fps = 0
frame_count = 0

show_hands = True
show_pose = True
show_objects = True

print("Multi-detection started!")
print("Press 'h' to toggle hand detection")
print("Press 'p' to toggle pose detection")
print("Press 'o' to toggle object detection")
print("Press 'q' to quit")

def draw_hand_landmarks(frame, hand_landmarks):
    """Draw hand landmarks and connections"""
    for hand in hand_landmarks:
        for landmark in hand:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            start = hand[start_idx]
            end = hand[end_idx]
            start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
            end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

def draw_pose_landmarks(frame, pose_landmarks):
    """Draw body pose landmarks and skeleton"""
    for pose in pose_landmarks:
        for landmark in pose:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(pose) and end_idx < len(pose):
                start = pose[start_idx]
                end = pose[end_idx]
                start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (255, 255, 0), 3)

def draw_objects(frame, detections):
    """Draw detected objects with bounding boxes"""
    h, w = frame.shape[:2]
    
    for detection in detections:
        bbox = detection.bounding_box
        x1 = int(bbox.origin_x)
        y1 = int(bbox.origin_y)
        x2 = int(bbox.origin_x + bbox.width)
        y2 = int(bbox.origin_y + bbox.height)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        category = detection.categories[0]
        label = category.category_name
        confidence = category.score
        
        label_text = f"{label}: {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (255, 0, 255), -1)
        
        cv2.putText(frame, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    frame_count += 1
    
    if show_hands:
        hand_results = hand_landmarker.detect_for_video(mp_image, frame_count)
        if hand_results.hand_landmarks:
            draw_hand_landmarks(frame, hand_results.hand_landmarks)
    
    if show_pose:
        pose_results = pose_landmarker.detect_for_video(mp_image, frame_count)
        if pose_results.pose_landmarks:
            draw_pose_landmarks(frame, pose_results.pose_landmarks)
    
    if show_objects:
        object_results = object_detector.detect_for_video(mp_image, frame_count)
        if object_results.detections:
            draw_objects(frame, object_results.detections)

    current_time = time.time()
    if current_time - prev_time > 0:
        fps = int(1 / (current_time - prev_time))
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    status_y = 60
    cv2.putText(frame, f"Hands: {'ON' if show_hands else 'OFF'}", (10, status_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if show_hands else (0, 0, 255), 2)
    cv2.putText(frame, f"Pose: {'ON' if show_pose else 'OFF'}", (10, status_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if show_pose else (0, 0, 255), 2)
    cv2.putText(frame, f"Objects: {'ON' if show_objects else 'OFF'}", (10, status_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if show_objects else (0, 0, 255), 2)

    cv2.imshow("Multi-Detection Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        show_hands = not show_hands
        print(f"Hand detection: {'ON' if show_hands else 'OFF'}")
    elif key == ord('p'):
        show_pose = not show_pose
        print(f"Pose detection: {'ON' if show_pose else 'OFF'}")
    elif key == ord('o'):
        show_objects = not show_objects
        print(f"Object detection: {'ON' if show_objects else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("Multi-detection tracking stopped.")