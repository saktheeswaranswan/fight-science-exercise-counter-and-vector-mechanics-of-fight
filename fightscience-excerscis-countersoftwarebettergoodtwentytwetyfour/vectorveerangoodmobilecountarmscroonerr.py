import cv2
import mediapipe as mp
import numpy as np
import json
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
rotation_angle = 0  # Rotation angle in degrees

# Functions to modify line position and rotation
def move_line_up():
    global horizontal_line_y
    if horizontal_line_y > 10:
        horizontal_line_y -= 10

def move_line_down():
    global horizontal_line_y
    if horizontal_line_y < screen_height - 10:
        horizontal_line_y += 10

def rotate_frame(frame, angle):
    center = (screen_width // 2, screen_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, rotation_matrix, (screen_width, screen_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (screen_width, screen_height))
    frame = rotate_frame(frame, rotation_angle)  # Apply rotation
    h, w, _ = frame.shape
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        keypoints = {landmark.name: (int(lm[landmark].x * w), int(lm[landmark].y * h)) for landmark in mp_pose.PoseLandmark}
        export_data.append(keypoints)
        
        # Draw keypoints
        for landmark, (x, y) in keypoints.items():
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw horizontal line
        cv2.line(frame, (0, horizontal_line_y), (screen_width, horizontal_line_y), (255, 255, 0), 2)
        
        # Process wrist positions
        for wrist in ['LEFT_WRIST', 'RIGHT_WRIST']:
            wrist_x, wrist_y = keypoints[wrist]
            
            if prev_wrist_y[wrist] is not None and prev_wrist_y[wrist] < horizontal_line_y and wrist_y >= horizontal_line_y:
                fist_cross_count += 1
            
            prev_wrist_y[wrist] = wrist_y  # Update previous position
            cv2.circle(frame, (wrist_x, wrist_y), 10, (0, 0, 255), -1)
        
        cv2.putText(frame, f'Fist Cross Count: {fist_cross_count}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Biomechanics Visualization', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == 82 or key == 2490368:  # Up Arrow
        move_line_up()
    elif key == 84 or key == 2621440:  # Down Arrow
        move_line_down()
    elif key == ord('r'):  # Rotate
        rotation_angle = (rotation_angle + 10) % 360

cap.release()
cv2.destroyAllWindows()

# Export JSON
with open("output_coordinates.json", "w") as json_file:
    json.dump(export_data, json_file, indent=4)

# Export CSV
csv_columns = [key for key in export_data[0]] if export_data else []
with open("output_coordinates.csv", "w", newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()
    csv_writer.writerows(export_data)
