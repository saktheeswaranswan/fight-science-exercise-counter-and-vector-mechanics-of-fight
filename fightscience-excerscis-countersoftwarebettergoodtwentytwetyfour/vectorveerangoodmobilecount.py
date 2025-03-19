import cv2
import mediapipe as mp
import numpy as np
import json
import time
import csv

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('videoplayback.mp4')

# Screen resolution
screen_width, screen_height = 640, 840
horizontal_line_y = screen_height // 2  # Define horizontal line position
fist_cross_count = 0
prev_wrist_y = {'LEFT_WRIST': None, 'RIGHT_WRIST': None}
export_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (screen_width, screen_height))
    h, w, _ = frame.shape
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        keypoints = {landmark.name: (int(lm[landmark].x * w), int(lm[landmark].y * h)) for landmark in mp_pose.PoseLandmark}
        export_data.append(keypoints)
        
        # Draw key points
        for landmark, (x, y) in keypoints.items():
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw horizontal line
        cv2.line(frame, (0, horizontal_line_y), (screen_width, horizontal_line_y), (255, 255, 0), 2)
        
        for wrist in ['LEFT_WRIST', 'RIGHT_WRIST']:
            wrist_x, wrist_y = keypoints[wrist]
            
            # Check for crossings
            if prev_wrist_y[wrist] is not None and prev_wrist_y[wrist] < horizontal_line_y and wrist_y >= horizontal_line_y:
                fist_cross_count += 1
            
            prev_wrist_y[wrist] = wrist_y  # Update previous position
            
            cv2.circle(frame, (wrist_x, wrist_y), 10, (0, 0, 255), -1)  # Mark wrist position
        
        cv2.putText(frame, f'Fist Cross Count: {fist_cross_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Biomechanics Visualization', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Export JSON
with open("output_coorcdinates.json", "w") as json_file:
    json.dump(export_data, json_file, indent=4)

# Export CSV
csv_columns = [key for key in export_data[0]] if export_data else []
with open("output_coordccinates.csv", "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()
    csv_writer.writerows(export_data)
