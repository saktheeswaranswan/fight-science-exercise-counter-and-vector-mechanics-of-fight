import cv2
import mediapipe as mp
import numpy as np
import os
import absl.logging
import math
import time
import json

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Set up video capture
cap = cv2.VideoCapture('Sequencebestmaa.mp4')

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

# Button controls
rotate_video = False
start_inference = False

# Set resolution for landscape video (16:9 aspect ratio)
screen_width, screen_height = 1280, 720

frame_idx = 0  # Frame counter
export_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Rotate video if button is pressed
    if rotate_video:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.resize(frame, (screen_width, screen_height))
    h, w, _ = frame.shape
    
    if start_inference:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            for person_id, person_landmarks in enumerate([results.pose_landmarks]):
                lm = person_landmarks.landmark
                keypoints = {
                    landmark.name: (int(lm[landmark].x * w), int(lm[landmark].y * h))
                    for landmark in mp_pose.PoseLandmark
                }
                
                export_data.append({"frame": frame_idx, "id": person_id, "keypoints": keypoints})
                
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                mp_drawing.draw_landmarks(frame, person_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=color))
                
                cv2.putText(frame, f"ID: {person_id}", (20, 50 + person_id * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow('Biomechanics Visualization', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        rotate_video = not rotate_video  # Toggle rotation
    elif key == ord('s'):
        start_inference = not start_inference  # Toggle inference
    
    frame_idx += 1
    
    if frame_idx % 100 == 0:
        with open("output_coordinates.json", "w") as outfile:
            json.dump(export_data, outfile, indent=4)

cap.release()
cv2.destroyAllWindows()

with open("output_coordinates.json", "w") as outfile:
    json.dump(export_data, outfile, indent=4)
