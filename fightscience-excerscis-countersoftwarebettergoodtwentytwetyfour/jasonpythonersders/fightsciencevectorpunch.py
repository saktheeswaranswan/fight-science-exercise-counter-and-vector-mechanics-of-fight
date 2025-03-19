import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # End point
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Get required keypoints from head to toe
        keypoints = {
            "nose": (int(lm[mp_pose.PoseLandmark.NOSE].x * w), int(lm[mp_pose.PoseLandmark.NOSE].y * h)),
            "left_eye": (int(lm[mp_pose.PoseLandmark.LEFT_EYE].x * w), int(lm[mp_pose.PoseLandmark.LEFT_EYE].y * h)),
            "right_eye": (int(lm[mp_pose.PoseLandmark.RIGHT_EYE].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_EYE].y * h)),
            "left_ear": (int(lm[mp_pose.PoseLandmark.LEFT_EAR].x * w), int(lm[mp_pose.PoseLandmark.LEFT_EAR].y * h)),
            "right_ear": (int(lm[mp_pose.PoseLandmark.RIGHT_EAR].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_EAR].y * h)),
            "left_shoulder": (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w), int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)),
            "right_shoulder": (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)),
            "left_elbow": (int(lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * w), int(lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * h)),
            "right_elbow": (int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h)),
            "left_wrist": (int(lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w), int(lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h)),
            "right_wrist": (int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)),
            "left_hip": (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w), int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h)),
            "right_hip": (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h)),
            "left_knee": (int(lm[mp_pose.PoseLandmark.LEFT_KNEE].x * w), int(lm[mp_pose.PoseLandmark.LEFT_KNEE].y * h)),
            "right_knee": (int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h)),
            "left_ankle": (int(lm[mp_pose.PoseLandmark.LEFT_ANKLE].x * w), int(lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h)),
            "right_ankle": (int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h)),
            "left_foot": (int(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * w), int(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * h)),
            "right_foot": (int(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h))
        }
        
        # Draw keypoints
        for key, point in keypoints.items():
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
            cv2.putText(frame, key, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Biomechanics Visualization', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
