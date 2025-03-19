import cv2
import mediapipe as mp
import numpy as np
import os
import absl.logging
import math
import time
import json  # Used for converting keypoints dict to string
import csv   # For CSV export
import tensorflow as tf

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Set resolution for vertical short video (9:16 aspect ratio, 420x640 resolution)
screen_width, screen_height = 420, 640

# Force calculation parameters
weight = 10  # kg
g = 9.81    # Gravity (m/s^2)
force_per_leg = (weight * g) / 2  # N per leg
impulse_force = 20  # N
impulse_duration = 0.5  # seconds
leg_impulse_time = 0.3
fist_impulse_time = 0.3

# List to store keypoints data for each processed frame
export_data = []
frame_idx = 0  # Frame counter

# Function to calculate angle between three points (in degrees)
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

# Function to draw angle arc at joint b between vectors from b->a and b->c
def draw_angle_arc(frame, a, b, c, angle):
    center = tuple(b)
    radius = 30
    start_angle = int(math.degrees(math.atan2(a[1] - b[1], a[0] - b[0])))
    end_angle = start_angle + int(angle)
    cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, (0, 255, 255), 2)

cap = cv2.VideoCapture('sdfgdfg.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate and resize the frame
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.resize(frame, (screen_width, screen_height))
    h, w, _ = frame.shape

    # Process frame with MediaPipe
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # Extract keypoints in current frame resolution
        keypoints = {
            landmark.name: (int(lm[landmark].x * w), int(lm[landmark].y * h))
            for landmark in mp_pose.PoseLandmark
        }
        
        # Save keypoints with frame index for CSV export
        export_data.append({"frame": frame_idx, "keypoints": keypoints})
        
        # Calculate joint angles (for left and right knees)
        joint_angles = {
            'Left Knee': (
                calculate_angle(keypoints['LEFT_HIP'], keypoints['LEFT_KNEE'], keypoints['LEFT_ANKLE']),
                'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'
            ),
            'Right Knee': (
                calculate_angle(keypoints['RIGHT_HIP'], keypoints['RIGHT_KNEE'], keypoints['RIGHT_ANKLE']),
                'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'
            ),
        }
        
        # Draw joint arcs and vector arrows for each knee
        for knee, (angle, a, b, c) in joint_angles.items():
            # Draw arc representing the angle
            draw_angle_arc(frame, keypoints[a], keypoints[b], keypoints[c], angle)
            # Draw vector arrow from knee to hip (red)
            cv2.arrowedLine(frame, tuple(keypoints[b]), tuple(keypoints[a]), (0, 0, 255), 2)
            # Draw vector arrow from knee to ankle (blue)
            cv2.arrowedLine(frame, tuple(keypoints[b]), tuple(keypoints[c]), (255, 0, 0), 2)
        
        # Draw force resultant vector for legs
        for foot in ['LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']:
            if foot in keypoints:
                foot_x, foot_y = keypoints[foot]
                resultant_force = int(force_per_leg * (joint_angles['Left Knee'][0] + joint_angles['Right Knee'][0]) / 180)
                if time.time() - leg_impulse_time < impulse_duration:
                    resultant_force += impulse_force
                cv2.arrowedLine(frame, (foot_x, foot_y), (foot_x, foot_y - resultant_force), (0, 0, 255), 3)
        
        # Draw hand force vectors (X and Y directions)
        for wrist in ['LEFT_WRIST', 'RIGHT_WRIST']:
            if wrist in keypoints:
                wrist_x, wrist_y = keypoints[wrist]
                impulse_offset = impulse_force if time.time() - fist_impulse_time < impulse_duration else 0
                cv2.arrowedLine(frame, (wrist_x, wrist_y), (wrist_x + 30 + impulse_offset, wrist_y), (255, 0, 0), 3)
                cv2.arrowedLine(frame, (wrist_x, wrist_y), (wrist_x, wrist_y - 30 - impulse_offset), (0, 255, 0), 3)
        
        # Kinetic linking: show moving rings for power transfer visualization
        for knee, (_, a, b, c) in joint_angles.items():
            mid_x = (keypoints[a][0] + keypoints[c][0]) // 2
            mid_y = (keypoints[a][1] + keypoints[c][1]) // 2
            cv2.circle(frame, (mid_x, mid_y), 10, (0, 255, 0), -1)
        
        # Display joint angles as text
        y_offset = 50
        for joint, (angle, _, _, _) in joint_angles.items():
            cv2.putText(frame, f"{joint}: {int(angle)} deg", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
        
        # (Optional) Draw landmarks and connections using MediaPipe drawing utility
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Biomechanics Visualization', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        leg_impulse_time = time.time()
    elif key == ord('f'):
        fist_impulse_time = time.time()
    
    frame_idx += 1  # Increment frame counter

    # Update CSV file every 100 frames
    if frame_idx % 100 == 0:
        with open("output_coordinates.csv", "w", newline="") as csvfile:
            fieldnames = ["frame", "keypoints"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in export_data:
                # Convert keypoints dict to a JSON string to preserve its structure in one CSV cell
                writer.writerow({"frame": data["frame"], "keypoints": json.dumps(data["keypoints"])})
                
cap.release()
cv2.destroyAllWindows()

# Final export of collected keypoints data to CSV file
with open("output_coordinates.csv", "w", newline="") as csvfile:
    fieldnames = ["frame", "keypoints"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in export_data:
        writer.writerow({"frame": data["frame"], "keypoints": json.dumps(data["keypoints"])})

