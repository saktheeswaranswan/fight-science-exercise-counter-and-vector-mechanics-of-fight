import cv2
import mediapipe as mp
import numpy as np
import os
import absl.logging

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Set up video capture
cap = cv2.VideoCapture('ffgrt.mp4')

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

# Set resolution for vertical short video (9:16 aspect ratio, 640x420 resolution)
screen_width, screen_height = 1929, 1080  # Adjusted for a short vertical format

# Force calculation
weight = 24  # kg
force_per_leg = (weight * 9.81) / 2  # N per leg

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
        
        # Calculate joint angles
        joint_angles = {
            'Left Knee': calculate_angle(keypoints['LEFT_HIP'], keypoints['LEFT_KNEE'], keypoints['LEFT_ANKLE']),
            'Right Knee': calculate_angle(keypoints['RIGHT_HIP'], keypoints['RIGHT_KNEE'], keypoints['RIGHT_ANKLE']),
            'Left Elbow': calculate_angle(keypoints['LEFT_SHOULDER'], keypoints['LEFT_ELBOW'], keypoints['LEFT_WRIST']),
            'Right Elbow': calculate_angle(keypoints['RIGHT_SHOULDER'], keypoints['RIGHT_ELBOW'], keypoints['RIGHT_WRIST'])
        }
        
        # Draw force vectors at feet
        for foot in ['LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']:
            cv2.arrowedLine(frame, keypoints[foot], (keypoints[foot][0], keypoints[foot][1] - int(force_per_leg / 10)), (0, 0, 255), 3)
        
        # Draw velocity vectors for fists
        for wrist in ['LEFT_WRIST', 'RIGHT_WRIST']:
            cv2.arrowedLine(frame, keypoints[wrist], (keypoints[wrist][0] + 30, keypoints[wrist][1]), (255, 0, 0), 3)
            cv2.arrowedLine(frame, keypoints[wrist], (keypoints[wrist][0], keypoints[wrist][1] - 30), (0, 255, 0), 3)
        
        # Display joint angles dynamically
        y_offset = 50
        for joint, angle in joint_angles.items():
            cv2.putText(frame, f"{joint}: {int(angle)} deg", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Biomechanics Visualization', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
