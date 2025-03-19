import cv2
import mediapipe as mp
import numpy as np
import os
import absl.logging
import math
import time

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Set up video capture
cap = cv2.VideoCapture('ffgrt.mp4')

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up video writer
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

# Force calculation
weight = 24  # kg
g = 9.81  # Gravity (m/s^2)
force_per_leg = (weight * g) / 2  # N per leg
impulse_force = 20  # N
impulse_duration = 0.5  # seconds
leg_impulse_time = 0
fist_impulse_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        keypoints = {landmark.name: (int(lm[landmark].x * w), int(lm[landmark].y * h)) for landmark in mp_pose.PoseLandmark}
        
        # Ensure keypoints stay within screen bounds
        for key in keypoints:
            keypoints[key] = (max(0, min(w - 1, keypoints[key][0])), max(0, min(h - 1, keypoints[key][1])))
        
        # Draw force resultant vector for legs
        for foot in ['LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']:
            foot_x, foot_y = keypoints[foot]
            resultant_force = int(force_per_leg / 180)
            if time.time() - leg_impulse_time < impulse_duration:
                resultant_force += impulse_force
            cv2.arrowedLine(frame, (foot_x, foot_y), (foot_x, max(0, foot_y - resultant_force)), (0, 0, 255), 3)
        
        # Hand force vectors (X, Y directions)
        for wrist in ['LEFT_WRIST', 'RIGHT_WRIST']:
            wrist_x, wrist_y = keypoints[wrist]
            impulse_offset = impulse_force if time.time() - fist_impulse_time < impulse_duration else 0
            cv2.arrowedLine(frame, (wrist_x, wrist_y), (min(w - 1, wrist_x + 30 + impulse_offset), wrist_y), (255, 0, 0), 3)  # X-direction
            cv2.arrowedLine(frame, (wrist_x, wrist_y), (wrist_x, max(0, wrist_y - 30 - impulse_offset)), (0, 255, 0), 3)  # Y-direction
        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Biomechanics Visualization', frame)
    out.write(frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        leg_impulse_time = time.time()
    elif key == ord('f'):
        fist_impulse_time = time.time()

cap.release()
out.release()
cv2.destroyAllWindows()

