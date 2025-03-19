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
horizontal_fist_line_y = screen_height // 3  # Define fist tracking line
horizontal_foot_line_y = (2 * screen_height) // 3  # Define foot tracking line

fist_cross_count = 0
foot_cross_count = 0
prev_wrist_y = {'LEFT_WRIST': None, 'RIGHT_WRIST': None}
prev_ankle_y = {'LEFT_ANKLE': None, 'RIGHT_ANKLE': None}

export_data = []
rotation_angle = 0  # Rotation angle in degrees

# Functions to modify line position and rotation
def move_fist_line(up=True):
    global horizontal_fist_line_y
    if up and horizontal_fist_line_y > 10:
        horizontal_fist_line_y -= 10
    elif not up and horizontal_fist_line_y < screen_height - 10:
        horizontal_fist_line_y += 10

def move_foot_line(up=True):
    global horizontal_foot_line_y
    if up and horizontal_foot_line_y > 10:
        horizontal_foot_line_y -= 10
    elif not up and horizontal_foot_line_y < screen_height - 10:
        horizontal_foot_line_y += 10

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
        
        # Draw horizontal lines
        cv2.line(frame, (0, horizontal_fist_line_y), (screen_width, horizontal_fist_line_y), (255, 255, 0), 2)
        cv2.line(frame, (0, horizontal_foot_line_y), (screen_width, horizontal_foot_line_y), (0, 255, 255), 2)
        
        # Process wrist and foot crossings
        for wrist in ['LEFT_WRIST', 'RIGHT_WRIST']:
            wrist_x, wrist_y = keypoints[wrist]
            if prev_wrist_y[wrist] is not None and prev_wrist_y[wrist] < horizontal_fist_line_y and wrist_y >= horizontal_fist_line_y:
                fist_cross_count += 1
            prev_wrist_y[wrist] = wrist_y  # Update previous position
            cv2.circle(frame, (wrist_x, wrist_y), 10, (0, 0, 255), -1)

        for ankle in ['LEFT_ANKLE', 'RIGHT_ANKLE']:
            ankle_x, ankle_y = keypoints[ankle]
            if prev_ankle_y[ankle] is not None and prev_ankle_y[ankle] < horizontal_foot_line_y and ankle_y >= horizontal_foot_line_y:
                foot_cross_count += 1
            prev_ankle_y[ankle] = ankle_y  # Update previous position
            cv2.circle(frame, (ankle_x, ankle_y), 10, (255, 0, 0), -1)
        
        # Display counts
        cv2.putText(frame, f'Fist Cross Count: {fist_cross_count}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Foot Cross Count: {foot_cross_count}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Biomechanics Visualization', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('f'):  # Move fist tracking line
        move_fist_line(up=True)
    elif key == ord('v'):  # Move fist tracking line down
        move_fist_line(up=False)
    elif key == ord('l'):  # Move foot tracking line
        move_foot_line(up=True)
    elif key == ord('b'):  # Move foot tracking line down
        move_foot_line(up=False)
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
